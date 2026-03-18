"""Network construction and Cell Transmission Model utilities.

Provides functions to build grid road networks, compute shortest paths,
and run macroscopic CTM traffic simulation steps. Designed to work
as the mock simulation backend when lightsim is not available.
"""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Link = dict[str, Any]
Node = dict[str, Any]
Network = dict[str, Any]
Route = list[tuple[str, str | None]]  # [(node_id, outgoing_link_id | None), ...]


# ---------------------------------------------------------------------------
# Network construction
# ---------------------------------------------------------------------------

def build_grid_network(
    rows: int = 4,
    cols: int = 4,
    link_length: float = 200.0,
    num_lanes: int = 2,
    v_free: float = 15.0,
    w: float = 5.0,
    k_jam: float = 0.15,
) -> Network:
    """Build a rectangular grid road network.

    Parameters
    ----------
    rows, cols : int
        Grid dimensions (number of intersections per side).
    link_length : float
        Length of each road link in metres.
    num_lanes : int
        Number of lanes per link.
    v_free : float
        Free-flow speed in m/s.
    w : float
        Backward wave speed in m/s (for CTM receiving function).
    k_jam : float
        Jam density in veh/m/lane.

    Returns
    -------
    Network
        Dictionary with keys ``"nodes"``, ``"links"``, ``"graph"``,
        ``"rows"``, ``"cols"`` and helper metadata.
    """
    G = nx.DiGraph()
    nodes: dict[str, Node] = {}
    links: dict[str, Link] = {}

    # --- Create intersection nodes ---
    for r in range(rows):
        for c in range(cols):
            nid = _node_id(r, c)
            is_boundary = r == 0 or r == rows - 1 or c == 0 or c == cols - 1
            nodes[nid] = {
                "id": nid,
                "row": r,
                "col": c,
                "is_boundary": is_boundary,
                "current_phase": 0,
                "num_phases": 4,  # N, E, S, W — simplified
                "incoming_links": [],
                "outgoing_links": [],
            }
            G.add_node(nid, **nodes[nid])

    # --- Create directed links (both directions for each grid edge) ---
    directions = [(0, 1, "E", "W"), (0, -1, "W", "E"), (1, 0, "S", "N"), (-1, 0, "N", "S")]

    capacity = v_free * k_jam * num_lanes  # veh/s (max flow per link)

    for r in range(rows):
        for c in range(cols):
            for dr, dc, fwd_dir, _bwd_dir in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    src = _node_id(r, c)
                    dst = _node_id(nr, nc)
                    lid = f"{src}->{dst}"
                    if lid not in links:
                        links[lid] = {
                            "id": lid,
                            "source": src,
                            "target": dst,
                            "length": link_length,
                            "num_lanes": num_lanes,
                            "v_free": v_free,
                            "w": w,
                            "k_jam": k_jam,
                            "capacity": capacity,
                            "density": 0.0,  # veh/m/lane — current state
                            "flow": 0.0,
                            "direction": fwd_dir,
                            "phase_index": _direction_to_phase(fwd_dir),
                        }
                        G.add_edge(src, dst, link_id=lid, weight=link_length)
                        nodes[src]["outgoing_links"].append(lid)
                        nodes[dst]["incoming_links"].append(lid)

    return {
        "nodes": nodes,
        "links": links,
        "graph": G,
        "rows": rows,
        "cols": cols,
        "link_length": link_length,
        "v_free": v_free,
        "w": w,
        "k_jam": k_jam,
        "num_lanes": num_lanes,
    }


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def compute_shortest_path(network: Network, origin: str, destination: str) -> Route:
    """Compute the shortest path through the network.

    Returns
    -------
    Route
        List of ``(node_id, outgoing_link_id)`` tuples along the path.
        The last entry has ``outgoing_link_id = None`` (destination reached).
    """
    G = network["graph"]
    try:
        node_path = nx.shortest_path(G, origin, destination, weight="weight")
    except nx.NetworkXNoPath:
        raise ValueError(f"No path from {origin} to {destination}")

    route: Route = []
    for i, node in enumerate(node_path):
        if i < len(node_path) - 1:
            next_node = node_path[i + 1]
            link_id = G.edges[node, next_node]["link_id"]
            route.append((node, link_id))
        else:
            route.append((node, None))
    return route


def get_route_intersections(network: Network, route: Route) -> list[str]:
    """Return the ordered list of intersection (node) IDs along a route."""
    return [node for node, _ in route]


def random_od_pair(
    network: Network,
    rng: np.random.Generator | None = None,
) -> tuple[str, str]:
    """Pick a random origin-destination pair on the network boundary.

    Both origin and destination are boundary nodes, and they are guaranteed
    to be distinct and connected.
    """
    if rng is None:
        rng = np.random.default_rng()

    boundary = [nid for nid, ndata in network["nodes"].items() if ndata["is_boundary"]]
    if len(boundary) < 2:
        raise ValueError("Network has fewer than 2 boundary nodes")

    G = network["graph"]
    for _ in range(200):
        origin, destination = rng.choice(boundary, size=2, replace=False)
        if nx.has_path(G, origin, destination):
            return str(origin), str(destination)

    raise RuntimeError("Could not find a connected OD pair after 200 attempts")


# ---------------------------------------------------------------------------
# Cell Transmission Model step
# ---------------------------------------------------------------------------

def ctm_step(network: Network, dt: float = 5.0) -> dict[str, float]:
    """Advance one Cell Transmission Model step across all links.

    The CTM computes *sending* and *receiving* flows at each cell boundary
    (here each link is a single cell) and updates densities accordingly.

    Parameters
    ----------
    network : Network
        The network dict (modified in-place — densities are updated).
    dt : float
        Time-step duration in seconds.

    Returns
    -------
    dict[str, float]
        Mapping from link_id to the flow that entered the link this step.
    """
    links = network["links"]
    nodes = network["nodes"]

    # --- 1. Compute sending & receiving for every link ---
    sending: dict[str, float] = {}
    receiving: dict[str, float] = {}

    for lid, lk in links.items():
        v_f = lk["v_free"]
        w = lk["w"]
        k_j = lk["k_jam"]
        n_lanes = lk["num_lanes"]
        density = lk["density"]  # veh/m/lane

        # Sending function: min(v_f * k, Q_max)
        s = v_f * density * n_lanes  # veh/s
        s = min(s, lk["capacity"])
        sending[lid] = s

        # Receiving function: min(w * (k_jam - k) * lanes, Q_max)
        r = w * (k_j - density) * n_lanes
        r = min(r, lk["capacity"])
        r = max(r, 0.0)
        receiving[lid] = r

    # --- 2. Compute actual flows through each node ---
    flow_in: dict[str, float] = {lid: 0.0 for lid in links}

    for nid, node in nodes.items():
        phase = node["current_phase"]
        for out_lid in node["outgoing_links"]:
            out_link = links[out_lid]
            recv = receiving[out_lid]

            # Sum sending from all incoming links that are green for this phase
            total_send = 0.0
            green_incoming: list[str] = []
            for in_lid in node["incoming_links"]:
                in_link = links[in_lid]
                if _movement_is_green(in_link, out_link, phase):
                    total_send += sending[in_lid]
                    green_incoming.append(in_lid)

            if total_send <= 0.0:
                continue

            # Actual flow is min(total sending, receiving)
            actual = min(total_send, recv)

            # Distribute proportionally among incoming links
            for in_lid in green_incoming:
                share = sending[in_lid] / total_send if total_send > 0 else 0.0
                flow_in[out_lid] += actual * share

    # --- 3. Update densities ---
    for lid, lk in links.items():
        length = lk["length"]
        n_lanes = lk["num_lanes"]
        inflow = flow_in[lid]
        outflow = sending[lid]  # approximation: what we computed as sendable

        # Density update: k_new = k_old + (inflow - outflow) * dt / (length * lanes)
        dk = (inflow - outflow) * dt / (length * n_lanes)
        new_density = lk["density"] + dk
        lk["density"] = float(np.clip(new_density, 0.0, lk["k_jam"]))
        lk["flow"] = inflow

    # --- 4. Add small random background demand on boundary links ---
    _inject_boundary_demand(network, dt)

    return flow_in


def reset_densities(network: Network, base_density: float = 0.02) -> None:
    """Reset all link densities to a uniform base value."""
    for lk in network["links"].values():
        lk["density"] = base_density
        lk["flow"] = 0.0


def get_total_queue_length(network: Network) -> float:
    """Sum of densities * lengths across all links (proxy for total queuing)."""
    total = 0.0
    for lk in network["links"].values():
        total += lk["density"] * lk["length"] * lk["num_lanes"]
    return total


# ---------------------------------------------------------------------------
# Signal helpers
# ---------------------------------------------------------------------------

def signal_is_green_for_link(network: Network, node_id: str, link_id: str) -> bool:
    """Check whether the current phase at *node_id* gives green to *link_id*.

    A simplified model: phase 0 = N/S through, phase 1 = E/W through,
    phase 2 = N/S left-turn, phase 3 = E/W left-turn.
    For the EV corridor we mainly care about through movements.
    """
    node = network["nodes"][node_id]
    link = network["links"][link_id]
    phase = node["current_phase"]
    return link["phase_index"] == phase


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _node_id(row: int, col: int) -> str:
    return f"n{row}_{col}"


def _direction_to_phase(direction: str) -> int:
    """Map link travel direction to the signal phase that serves it.

    Simplified 4-phase scheme:
        0 — Northbound / Southbound through
        1 — Eastbound / Westbound through
        2 — N/S left (not used heavily)
        3 — E/W left (not used heavily)
    """
    return {"N": 0, "S": 0, "E": 1, "W": 1}.get(direction, 0)


def _movement_is_green(
    in_link: Link,
    out_link: Link,
    phase: int,
) -> bool:
    """Determine if the movement from *in_link* to *out_link* is green.

    Simplified: a through movement is green when the phase matches the
    incoming link's direction group.
    """
    return in_link["phase_index"] == phase


def _inject_boundary_demand(network: Network, dt: float) -> None:
    """Add a trickle of vehicles on boundary entry links each step."""
    nodes = network["nodes"]
    links = network["links"]
    demand_rate = 0.05  # veh/s per boundary entry link

    for nid, node in nodes.items():
        if not node["is_boundary"]:
            continue
        for lid in node["outgoing_links"]:
            lk = links[lid]
            target = links[lid]["target"]
            # Only inject on links heading inward (target is not boundary,
            # or accept all for simplicity)
            added = demand_rate * dt / (lk["length"] * lk["num_lanes"])
            lk["density"] = float(np.clip(lk["density"] + added, 0.0, lk["k_jam"]))
