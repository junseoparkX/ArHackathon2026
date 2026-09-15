"""
Amazon Robotics Hackathon - Routing API

This module defines the routing API for the Amazon Robotics Hackathon.
Students will implement the drive_unit_next_move function in this module.

*****IMPORTANT*****
Team name: Cool ass team name
Email address: dzhang74@student.ubc.ca, egekayaa@student.ubc.ca, jinoh001@student.ubc.ca, jpark74@student.ubc.ca
*******************
"""

import heapq
import math
import copy
import time
from functools import lru_cache
from typing import Optional
from ar_hackathon.models.graph_state import GraphState
from ar_hackathon.models.pod import Pod

_FORECAST_CONF = 1.0
_FORECAST_PRIOR = 1.0
_FORECAST_MIN_HISTORY = 1
_FORECAST_FIXED_INTERVAL = 4
_FORECAST_DEST_MODE = "nearest"


_INF = float("inf")
_planner = None


class _Planner:
    """Cache graph searches and coordinate pickup queues across callbacks.

    Only information in GraphState is used: future pods and test files are
    deliberately unnecessary. Occupancy is read anew after each committed move.
    """

    def __init__(self, state, signature):
        self.signature = signature
        self.nodes = {node.id: node for node in state.nodes}
        self.adj = {node.id: [] for node in state.nodes}
        self.edges = list(state.edges)
        self.edge_index = {}
        # Match get_edge's first-match behavior, including directed edges.
        for index, edge in enumerate(state.edges):
            pairs = [(edge.from_node, edge.to_node)]
            if edge.bidirectional:
                pairs.append((edge.to_node, edge.from_node))
            for source, dest in pairs:
                if (source, dest) in self.edge_index:
                    continue
                self.edge_index[source, dest] = index
                if edge.capacity == 0 or self.nodes[dest].capacity == 0:
                    continue
                duration = max(1, math.ceil(edge.weight))
                self.adj[source].append((dest, duration, index))
        self.distances = {}
        self.time = -1
        self.last_id = None
        self.assignments = {}
        self.parking = {}
        self.reserved = set()
        self.positions = {}
        self.stalled = {}
        self.yielding = {}
        self.seen_arrivals = set()
        self.arrival_history = {}
        self.batch_waits = {}
        self.search_seconds = 0.0

    def distances_from(self, source):
        if source not in self.distances:
            result = {source: 0}
            queue = [(0, source)]
            while queue:
                distance, node = heapq.heappop(queue)
                if distance != result[node]:
                    continue
                for dest, duration, _ in self.adj.get(node, ()):
                    candidate = distance + duration
                    if candidate < result.get(dest, _INF):
                        result[dest] = candidate
                        heapq.heappush(queue, (candidate, dest))
            self.distances[source] = result
        return self.distances[source]

    def distance(self, source, dest):
        return self.distances_from(source).get(dest, _INF)

    def delivery_target(self, start, pods, now):
        """Maximize discounted delivery reward over a small set of stops.

        The recurrence discounts the whole remaining route on each leg. For
        unusually large loads use a bounded greedy choice instead of 2**n work.
        """
        values = {}
        for pod in pods:
            dest = pod.destination_station
            values[dest] = values.get(dest, 0.0) + math.exp(
                -min(700, max(0, now - pod.entry_time) / 50.0))
        stops = sorted(dest for dest in values
                       if self.distance(start, dest) < _INF)
        if not stops:
            return None
        if len(stops) > 7:
            return max(stops, key=lambda dest: values[dest] /
                       (1 + self.distance(start, dest)))

        @lru_cache(maxsize=None)
        def best(node, mask):
            reward, target = 0.0, None
            for index, dest in enumerate(stops):
                bit = 1 << index
                if not mask & bit:
                    continue
                distance = self.distance(node, dest)
                if distance == _INF:
                    continue
                tail, _ = best(dest, mask ^ bit)
                candidate = math.exp(-distance / 50.0) * (values[dest] + tail)
                if target is None or candidate > reward:
                    reward, target = candidate, dest
            return reward, target

        return best(start, (1 << len(stops)) - 1)[1]

    def plan_assignments(self, state):
        previous = self.assignments
        self.assignments = {}
        self.reserved = set()
        self.parking = {}
        queues = {}
        for pod in state.active_pods:
            if pod.carried_by is None and pod.current_node is not None:
                queues.setdefault(pod.current_node, []).append(pod)
        for pods in queues.values():
            pods.sort(key=lambda pod: (pod.entry_time, pod.id))
        available = [unit for unit in state.drive_units if not unit.carrying
                     and unit.capacity > 0]
        # Allocate actual queue prefixes: the engine, not the driver, chooses
        # which pod a robot picks up. In-transit robots can reserve future work.
        while available and queues:
            best = None
            for unit in available:
                start = (unit.transit_destination if unit.in_transit
                         else unit.current_node)
                remaining = (max(0, math.ceil(unit.transit_remaining_time))
                             if unit.in_transit else 0)
                for source, pods in queues.items():
                    pickup = remaining + self.distance(start, source)
                    batch = pods[:unit.capacity]
                    if pickup == _INF:
                        continue
                    value, delivery = 0.0, 0.0
                    for pod in batch:
                        leg = self.distance(source, pod.destination_station)
                        if leg < _INF:
                            value += math.exp(-min(700, max(0,
                                state.current_time_step - pod.entry_time +
                                pickup + leg) / 50.0))
                            delivery = max(delivery, leg)
                    if not value:
                        continue
                    merit = value / (2 + pickup + 0.5 * delivery)
                    if previous.get(unit.id) == source:
                        merit *= 1.08  # Avoid exchanging jobs for tiny gains.
                    candidate = (merit, -pickup, -unit.id, -source)
                    if best is None or candidate > best[0]:
                        best = (candidate, unit, source, batch)
            if best is None:
                break
            _, unit, source, batch = best
            self.assignments[unit.id] = source
            self.reserved.update(pod.id for pod in batch)
            del queues[source][:len(batch)]
            if not queues[source]:
                del queues[source]
            available.remove(unit)

        # Stage idle robots near storage so later arrivals need no round trip.
        # Spread them among sources; never use a station as a parking target.
        storage = sorted(node.id for node in state.nodes
                         if node.node_type == "storage" and node.capacity != 0)
        staged = {node: 0 for node in storage}
        for source in self.assignments.values():
            if source in staged:
                staged[source] += 1
        for unit in sorted(available, key=lambda item: item.id):
            start = unit.transit_destination if unit.in_transit else unit.current_node
            options = [node for node in storage
                       if self.distance(start, node) < _INF
                       and (self.nodes[node].capacity is None or
                            staged[node] < self.nodes[node].capacity)]
            if options:
                target = min(options, key=lambda node:
                             (self.distance(start, node) + 6 * staged[node], node))
                self.parking[unit.id] = target
                staged[target] += 1

    def occupancy(self, state):
        edge_release = {}
        node_release = {node: [] for node in self.nodes}
        for unit in state.drive_units:
            if unit.in_transit:
                remaining = max(1, math.ceil(unit.transit_remaining_time))
                # Respect occupancy semantics even when oppositely directed
                # edges or parallel edges appear in the input graph.
                for index, edge in enumerate(self.edges):
                    if edge.connects(unit.current_node, unit.transit_destination):
                        edge_release.setdefault(index, []).append(remaining)
                node_release[unit.transit_destination].append(remaining + 1)
            else:
                # A standing unit may depart next step. This is an estimate
                # for lookahead only; an occupied first hop is always rejected.
                delay = 1 + min(6, self.stalled.get(unit.id, 0))
                node_release[unit.current_node].append(delay)
        for releases in edge_release.values():
            releases.sort()
        for releases in node_release.values():
            releases.sort()
        return edge_release, node_release

    @staticmethod
    def opening(releases, capacity):
        if capacity is None or len(releases) < capacity:
            return 0
        if capacity <= 0:
            return _INF
        return releases[len(releases) - capacity]

    def route(self, unit, target, edge_release, node_release):
        """Earliest arrival with waits for currently reserved aisles/docks.

        Future traffic is replanned each tick. A wait can beat a long detour;
        the first action then remains None until that route actually opens.
        """
        start = unit.current_node
        if target is None or start == target:
            return None
        queue = [(0, start)]
        arrivals = {start: 0}
        actions = {start: None}
        while queue:
            arrival, node = heapq.heappop(queue)
            if arrival != arrivals[node]:
                continue
            if node == target:
                return actions[node]
            for dest, duration, index in self.adj.get(node, ()):
                edge = self.edges[index]
                edge_open = self.opening(edge_release.get(index, ()), edge.capacity)
                dock_open = self.opening(node_release[dest], self.nodes[dest].capacity)
                departure = max(arrival, edge_open, dock_open)
                candidate = departure + duration
                if candidate < arrivals.get(dest, _INF):
                    arrivals[dest] = candidate
                    actions[dest] = (dest if departure == 0 else None) if node == start else actions[node]
                    heapq.heappush(queue, (candidate, dest))
        return None

    def escape(self, unit, target, edge_release, node_release, state, pods_by_id):
        """Vacate a finite dock or break a persistent blocking cycle."""
        options = []
        for dest, duration, index in self.adj.get(unit.current_node, ()):
            if self.opening(edge_release.get(index, ()), self.edges[index].capacity):
                continue
            if self.opening(node_release[dest], self.nodes[dest].capacity):
                continue
            onward = self.distance(dest, target) if target is not None else 0
            if onward == _INF:
                continue
            node = self.nodes[dest]
            penalty = (12 if node.node_type == "station" else 0)
            penalty += 6 if node.capacity is not None else 0
            blocks_route = (node.capacity is not None and
                            self.needed_by_others(dest, unit.id, state, pods_by_id))
            options.append((blocks_route, penalty + duration + onward, dest))
        return min(options)[2] if options else None

    def needed_by_others(self, node, unit_id, state, pods_by_id):
        """Is this node on another robot's shortest route to visible work?"""
        for other in state.drive_units:
            if other.id == unit_id:
                continue
            start = (other.transit_destination if other.in_transit
                     else other.current_node)
            if start == node:
                # It already owns a slot; normal occupancy will protect it.
                continue
            carried = [pods_by_id[pod_id] for pod_id in other.carrying
                       if pod_id in pods_by_id]
            target = (self.delivery_target(start, carried, state.current_time_step)
                      if carried else self.assignments.get(other.id))
            if target is None:
                target = self.parking.get(other.id)
            if target is None:
                continue
            direct = self.distance(start, target)
            via = self.distance(start, node) + self.distance(node, target)
            if direct < _INF and via == direct:
                return True
        return False

    def move(self, unit, state):
        now = state.current_time_step
        if now != self.time:
            for other in state.drive_units:
                position = (other.current_node, other.in_transit,
                            other.transit_destination, tuple(other.carrying))
                if not other.in_transit and self.positions.get(other.id) == position:
                    self.stalled[other.id] = self.stalled.get(other.id, 0) + 1
                else:
                    self.stalled[other.id] = 0
                self.positions[other.id] = position
            self.plan_assignments(state)
            self.time = now
        self.last_id = unit.id
        pods_by_id = {pod.id: pod for pod in state.active_pods}
        carried = [pods_by_id[pod_id] for pod_id in unit.carrying
                   if pod_id in pods_by_id]
        target = self.delivery_target(unit.current_node, carried, now) if carried else self.assignments.get(unit.id)

        if carried and unit.has_capacity and target is not None:
            # Pick up unassigned batches only when the extra travel is short
            # and they are bound for a station already on the delivery tour.
            destinations = {pod.destination_station for pod in carried}
            direct = self.distance(unit.current_node, target)
            candidates = []
            queues = {}
            for pod in state.active_pods:
                if pod.carried_by is None and pod.current_node is not None:
                    queues.setdefault(pod.current_node, []).append(pod)
            for source, waiting in queues.items():
                waiting.sort(key=lambda pod: (pod.entry_time, pod.id))
                batch = waiting[:unit.capacity - len(carried)]
                if any(pod.id in self.reserved or pod.destination_station not in destinations
                       for pod in batch):
                    continue
                to_source = self.distance(unit.current_node, source)
                detour = to_source + self.distance(source, target) - direct
                if source != unit.current_node and detour <= min(3, direct * 0.2):
                    candidates.append((detour, to_source, source, batch))
            if candidates:
                _, _, target, batch = min(candidates, key=lambda item: item[:3])
                self.reserved.update(pod.id for pod in batch)

        if target is None:
            target = self.parking.get(unit.id)
        edge_release, node_release = self.occupancy(state)

        # After stepping aside, stay out of the way until the waiting robot
        # reserves the cleared node. Otherwise a lower-ID robot can immediately
        # reclaim the slot on every tick and starve the robot it just yielded to.
        yielding = self.yielding.get(unit.id)
        if yielding is not None:
            cleared, buffer = yielding
            buffer_blocks = (self.nodes[buffer].capacity is not None and
                             self.needed_by_others(buffer, unit.id, state, pods_by_id))
            if not buffer_blocks and self.needed_by_others(cleared, unit.id, state, pods_by_id):
                if unit.current_node == buffer:
                    return None
            else:
                del self.yielding[unit.id]

        current = self.nodes[unit.current_node]
        if (current.capacity is not None and not carried and
                (target is None or target == unit.current_node) and
                self.needed_by_others(unit.current_node, unit.id, state, pods_by_id)):
            result = self.escape(unit, None, edge_release, node_release, state, pods_by_id)
            if result is not None:
                self.yielding[unit.id] = (unit.current_node, result)
            return result
        result = self.route(unit, target, edge_release, node_release)
        if result is None and current.capacity is not None:
            # Empty docks must clear even when there are no visible jobs.
            if not carried and (target is None or current.node_type == "station"):
                result = self.escape(unit, target, edge_release, node_release, state, pods_by_id)
            elif self.stalled.get(unit.id, 0) >= 3 and target != unit.current_node:
                result = self.escape(unit, target, edge_release, node_release, state, pods_by_id)
                if result is not None:
                    self.yielding[unit.id] = (unit.current_node, result)
        return result


def drive_unit_next_move(drive_unit_id: int, state: GraphState) -> Optional[int]:
    """Return one legal adjacent node, or wait; pickups/deliveries are automatic."""
    global _planner
    unit = state.get_drive_unit(drive_unit_id)
    if unit is None or unit.in_transit:
        return None
    signature = (
        tuple((node.id, node.node_type, node.capacity) for node in state.nodes),
        tuple((edge.from_node, edge.to_node, edge.weight, edge.capacity,
               edge.bidirectional) for edge in state.edges),
        tuple((other.id, other.capacity) for other in state.drive_units),
    )
    if (_planner is None or _planner.signature != signature or
            state.current_time_step < _planner.time or
            (state.current_time_step == 0 and _planner.time == 0 and
             _planner.last_id is not None and drive_unit_id <= _planner.last_id)):
        _planner = _Planner(state, signature)
    _observe_arrivals(state, _planner)
    baseline = _planner.move(unit, state)
    if _should_batch_wait(unit, state, _planner, baseline):
        # A deliberate wait invalidates remaining actions from a joint plan.
        _planner.joint_actions = {}
        return None
    return _choose_move(unit, state, _planner, baseline)


def _observe_arrivals(state, planner):
    """Learn only arrival epochs actually seen through the public state."""
    units = {unit.id: unit for unit in state.drive_units}
    for pod in state.active_pods:
        if pod.id in planner.seen_arrivals:
            continue
        planner.seen_arrivals.add(pod.id)
        source = pod.current_node
        if source is None:
            carrier = units.get(pod.carried_by)
            if carrier is not None and not carrier.in_transit and pod.entry_time == state.current_time_step:
                source = carrier.current_node
        if source is not None:
            history = planner.arrival_history.setdefault(source, {})
            history.setdefault(pod.entry_time, []).append(pod.destination_station)
            if len(history) > 6:
                del history[min(history)]


def _should_batch_wait(unit, state, planner, normal):
    """Bounded batching based on observed arrivals, never a future schedule.

    A forecast can be wrong. Charge the wait against every carried pod and
    discount the estimated saved round trip by 50 percent before accepting it.
    The forecast expires at its first predicted epoch, so a missed arrival
    cannot cause indefinite waiting.
    """
    if normal is None or not unit.carrying or not unit.has_capacity:
        return False
    history = planner.arrival_history.get(unit.current_node, {})
    times = sorted(history)
    if len(times) < 2:
        return False
    gaps = [later - earlier for earlier, later in zip(times, times[1:])]
    interval = sorted(gaps)[len(gaps) // 2]
    if interval <= 0 or any(abs(gap - interval) > max(1, interval * 0.25) for gap in gaps[-3:]):
        return False
    delay = times[-1] + interval - state.current_time_step
    if not 0 < delay <= min(4, interval / 2):
        return False
    pod_map = {pod.id: pod for pod in state.active_pods}
    carried = [pod_map[pod_id] for pod_id in unit.carrying if pod_id in pod_map]
    target = planner.delivery_target(unit.current_node, carried, state.current_time_step)
    if target is None:
        return False
    reward = sum(math.exp(-min(700, max(0, state.current_time_step - pod.entry_time +
                     planner.distance(unit.current_node, pod.destination_station)) / 50.0))
                 for pod in carried)
    loss = reward * (1 - math.exp(-delay / 50.0))
    round_trip = (planner.distance(unit.current_node, target) +
                  planner.distance(target, unit.current_node))
    destinations = [dest for values in history.values() for dest in values]
    gain = sum(math.exp(-planner.distance(unit.current_node, dest) / 50.0) *
               (1 - math.exp(-max(0, round_trip - delay) / 50.0))
               for dest in destinations) / len(destinations)
    if 0.5 * gain <= loss:
        return False
    key = (unit.current_node, tuple(unit.carrying))
    previous_key, count, last_tick = planner.batch_waits.get(unit.id, (None, 0, -1))
    if previous_key != key:
        count = 0
    if count >= 4:
        return False
    if last_tick != state.current_time_step:
        count += 1
    planner.batch_waits[unit.id] = (key, count, state.current_time_step)
    return True


def _clone_planner(planner):
    clone = copy.copy(planner)
    for name in ('assignments', 'parking', 'positions', 'stalled', 'yielding'):
        setattr(clone, name, getattr(planner, name).copy())
    clone.reserved = planner.reserved.copy()
    clone.seen_arrivals = planner.seen_arrivals.copy()
    clone.arrival_history = {source: {tick: list(destinations) for tick, destinations in history.items()}
                            for source, history in planner.arrival_history.items()}
    clone.batch_waits = planner.batch_waits.copy()
    if hasattr(planner, 'joint_actions'):
        clone.joint_actions = planner.joint_actions.copy()
    return clone


def _resolve_pods(state):
    pods = {pod.id: pod for pod in state.active_pods}
    for unit in sorted(state.drive_units, key=lambda item: item.id):
        if unit.in_transit:
            continue
        for pod_id in list(unit.carrying):
            pod = pods.get(pod_id)
            if pod is not None and pod.destination_station == unit.current_node:
                unit.carrying.remove(pod_id)
                pod.carried_by = None
                pod.current_node = unit.current_node
                pod.delivery_time = state.current_time_step
                state.active_pods.remove(pod)
                state.delivered_pods.append(pod)
        waiting = sorted((pod for pod in state.active_pods
                          if pod.carried_by is None and
                          pod.current_node == unit.current_node),
                         key=lambda pod: (pod.entry_time, pod.id))
        for pod in waiting[:max(0, unit.capacity - len(unit.carrying))]:
            pod.carried_by = unit.id
            pod.current_node = None
            unit.carrying.append(pod.id)


def _commit_move(state, unit, dest):
    if dest is None or unit.in_transit or dest == unit.current_node:
        return False
    edge = state.get_edge(unit.current_node, dest)
    if edge is None:
        return False
    if edge.capacity is not None and state.edge_occupancy(unit.current_node, dest) >= edge.capacity:
        return False
    node = state.get_node(dest)
    if node is not None and node.capacity is not None and state.node_occupancy(dest) >= node.capacity:
        return False
    unit.in_transit = True
    unit.transit_destination = dest
    unit.transit_remaining_time = edge.weight
    return True


def _finish_tick(state, first_id, first_action, planner):
    """Continue from a callback, then advance and resolve the engine's tick.

    Lower IDs have already been polled in this state's tick. Passing None as
    first_id starts a later tick after its automatic pickup/delivery phase.
    """
    if first_id is not None and not hasattr(planner, 'recorded_actions'):
        planner.recorded_actions = {}
    for unit in sorted(state.drive_units, key=lambda item: item.id):
        if unit.in_transit or (first_id is not None and unit.id < first_id):
            continue
        action = first_action if unit.id == first_id else planner.move(unit, state)
        if getattr(planner, 'forced_time', -1) == state.current_time_step:
            action = planner.forced_actions.get(unit.id, action)
        target = getattr(planner, 'rollout_targets', {}).get(unit.id)
        if target is not None and unit.id != first_id:
            if unit.current_node == target:
                wait_until = getattr(planner, 'rollout_wait_until', {}).get(unit.id, -1)
                if state.current_time_step < wait_until and not unit.carrying:
                    action = None
                else:
                    del planner.rollout_targets[unit.id]
            else:
                edges, nodes = planner.occupancy(state)
                action = planner.route(unit, target, edges, nodes)
        committed = _commit_move(state, unit, action)
        if first_id is not None:
            planner.recorded_actions[unit.id] = (unit.current_node, tuple(unit.carrying), action if committed else None)
    for unit in state.drive_units:
        if unit.in_transit:
            unit.transit_remaining_time -= 1
            if unit.transit_remaining_time <= 0:
                unit.current_node = unit.transit_destination
                unit.in_transit = False
                unit.transit_destination = None
                unit.transit_remaining_time = 0
    _resolve_pods(state)
    state.current_time_step += 1


def _forecast_arrivals(state, planner, horizon):
    """Construct uncertain next batches solely from previously seen arrivals."""
    pending = []
    now = state.current_time_step
    existing = {pod.id for pod in state.active_pods}
    for source, history in planner.arrival_history.items():
        times = sorted(history)
        if len(times) < _FORECAST_MIN_HISTORY:
            continue
        gaps = [b - a for a, b in zip(times, times[1:])]
        destinations = history[times[-1]]
        if not destinations:
            continue
        if gaps:
            interval = sorted(gaps)[len(gaps) // 2]
            variation = sum(abs(gap - interval) for gap in gaps) / max(1, sum(gaps))
            confidence = _FORECAST_CONF * max(0.1, 1.0 - variation)
        else:
            distances = [planner.distance(source, dest) + planner.distance(dest, source)
                         for dest in destinations]
            finite = [distance for distance in distances if distance < _INF]
            if not finite:
                continue
            interval = max(1, round(_FORECAST_PRIOR * sum(finite) / len(finite)))
            confidence = _FORECAST_CONF * 0.5
        if _FORECAST_FIXED_INTERVAL:
            nearest = min((planner.distance(source, node.id) for node in state.nodes
                           if node.node_type == "station"), default=_INF)
            interval = max(1, round(nearest)) if nearest < _INF else interval
            confidence = _FORECAST_CONF
        if _FORECAST_DEST_MODE == "nearest":
            stations = [node.id for node in state.nodes if node.node_type == "station"]
            destinations = sorted(stations, key=lambda dest: (planner.distance(source, dest), dest))[:1]
        elif _FORECAST_DEST_MODE == "global":
            observed = {dest for epochs in planner.arrival_history.values()
                        for batch in epochs.values() for dest in batch}
            destinations = sorted(observed, key=lambda dest: (planner.distance(source, dest), dest))[:1]
        else:
            destinations = destinations[:1]
        if interval <= 0 or now >= times[-1] + 2 * interval:
            continue
        arrival = times[-1] + interval
        if arrival <= now:
            arrival += interval
            confidence *= 0.5
        if arrival > now + horizon:
            continue
        for index, dest in enumerate(destinations[:2]):
            ident = "__forecast_%s_%s_%s" % (source, arrival, index)
            while ident in existing:
                ident += "_"
            existing.add(ident)
            pending.append((arrival, Pod(ident, source, dest, arrival), confidence))
    return pending


def _rollout(state, unit_id, action, planner, horizon, deadline, target=None, extra_actions=None, capture=None):
    simulation = state.deep_copy()
    simulation.delivered_pods = []
    future = (_forecast_arrivals(state, planner, horizon)
              if (not state.get_drive_unit(unit_id).carrying and
                  planner.nodes[state.get_drive_unit(unit_id).current_node].capacity is None) else [])
    confidence = {pod.id: weight for _, pod, weight in future}
    policy = _clone_planner(planner)
    policy.rollout_targets = {unit_id: target} if target is not None else {}
    policy.rollout_wait_until = {unit_id: min(arrival for arrival, pod, _ in future
                                             if pod.current_node == target)} if (
        target is not None and any(pod.current_node == target for _, pod, _ in future)) else {}
    policy.forced_time = state.current_time_step
    policy.forced_actions = extra_actions or {}
    policy.recorded_actions = {}
    for step in range(horizon):
        if time.perf_counter() >= deadline:
            return None
        if step:
            for arrival, pod, _ in future:
                if arrival == simulation.current_time_step:
                    simulation.active_pods.append(pod)
            _resolve_pods(simulation)
        _finish_tick(simulation, unit_id if step == 0 else None,
                     action if step == 0 else None, policy)
        if not simulation.active_pods and not any(
                arrival >= simulation.current_time_step for arrival, _, _ in future):
            break
    if capture is not None:
        capture.update(policy.recorded_actions)
    reward = sum(confidence.get(pod.id, 1.0) * math.exp(-min(700, (pod.delivery_time - pod.entry_time) / 50.0))
                 for pod in simulation.delivered_pods)
    # Optimistic terminal delivery estimates keep long jobs visible to a short
    # horizon; only pods already known in the supplied state are evaluated.
    units = {unit.id: unit for unit in simulation.drive_units}
    for pod in simulation.active_pods:
        best = _INF
        candidates = ([units[pod.carried_by]] if pod.carried_by in units
                      else simulation.drive_units)
        for unit in candidates:
            position = unit.transit_destination if unit.in_transit else unit.current_node
            delay = max(0, math.ceil(unit.transit_remaining_time)) if unit.in_transit else 0
            if pod.carried_by is None:
                if not unit.has_capacity:
                    continue
                delay += policy.distance(position, pod.current_node)
                position = pod.current_node
            delay += policy.distance(position, pod.destination_station)
            best = min(best, delay)
        if best < _INF:
            duration = max(0, simulation.current_time_step + best - 1 - pod.entry_time)
            reward += confidence.get(pod.id, 1.0) * 0.95 * math.exp(-min(700, duration / 50.0))
    return reward


def _choose_move(unit, state, planner, baseline):
    # Keep total search cost bounded over an episode, independently of its
    # duration. The original lightweight policy remains the fallback.
    if planner.search_seconds >= 12.0:
        return baseline
    started = time.perf_counter()
    try:
        return _search_move(unit, state, planner, baseline)
    finally:
        planner.search_seconds += time.perf_counter() - started


def _search_move(unit, state, planner, baseline):
    pending = getattr(planner, 'joint_actions', {})
    if unit.carrying and getattr(planner, 'joint_time', -1) == state.current_time_step and unit.id in pending:
        position, carrying, action = pending.pop(unit.id)
        if position == unit.current_node and carrying == tuple(unit.carrying):
            edges, nodes = planner.occupancy(state)
            index = planner.edge_index.get((unit.current_node, action))
            if action is None or (index is not None and
                    not planner.opening(edges.get(index, ()), planner.edges[index].capacity) and
                    not planner.opening(nodes[action], planner.nodes[action].capacity)):
                return action
    if (not state.active_pods or len(state.nodes) > 45 or
            len(state.drive_units) > 8 or len(state.active_pods) > 40):
        return baseline
    edges, nodes = planner.occupancy(state)
    actions = [baseline]
    for dest, _, index in planner.adj.get(unit.current_node, ()):
        if (dest not in actions and
                not planner.opening(edges.get(index, ()), planner.edges[index].capacity) and
                not planner.opening(nodes[dest], planner.nodes[dest].capacity)):
            actions.append(dest)
    if None not in actions:
        actions.append(None)
    if len(actions) == 1:
        return baseline
    deadline = time.perf_counter() + 0.20
    horizon = 40
    best_score = _rollout(state, unit.id, baseline, planner, horizon, deadline)
    if best_score is None:
        return baseline
    best_action = baseline
    best_joint = None
    for action in actions[1:]:
        score = _rollout(state, unit.id, action, planner, horizon, deadline)
        if score is not None and score > best_score + 1e-8:
            best_score, best_action = score, action
    targets = set()
    for pod in state.active_pods:
        if pod.carried_by == unit.id:
            targets.add(pod.destination_station)
        elif pod.carried_by is None and unit.has_capacity:
            targets.add(pod.current_node)
    targets.discard(unit.current_node)
    targets.discard(None)
    if not unit.carrying:
        targets.update(pod.current_node for _, pod, _ in _forecast_arrivals(state, planner, horizon))
    for target in sorted(targets, key=lambda node: (planner.distance(unit.current_node, node), node))[:6]:
        action = planner.route(unit, target, edges, nodes)
        score = _rollout(state, unit.id, action, planner, horizon, deadline, target)
        if score is not None and score > best_score + 1e-8:
            best_score, best_action = score, action
    # Compare pairs of simultaneous decisions, retaining actual first-tick
    # choices so later callbacks can execute the plan in engine ID order.
    later = [other for other in state.drive_units
             if not other.in_transit and other.id > unit.id]
    for other in later[:3]:
        other_actions = [None] + [dest for dest, _, _ in planner.adj.get(other.current_node, ())]
        for action in actions:
            for other_action in other_actions:
                capture = {}
                score = _rollout(state, unit.id, action, planner, horizon, deadline,
                                 extra_actions={other.id: other_action}, capture=capture)
                if score is not None and score > best_score + 1e-8:
                    best_score, best_action, best_joint = score, action, capture
                if time.perf_counter() >= deadline:
                    break
            if time.perf_counter() >= deadline:
                break
        if time.perf_counter() >= deadline:
            break
    if best_joint is not None:
        best_joint.pop(unit.id, None)
        planner.joint_time = state.current_time_step
        planner.joint_actions = best_joint
    return best_action
