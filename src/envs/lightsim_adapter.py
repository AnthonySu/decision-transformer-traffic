"""Adapter to use LightSim as the traffic simulation backend.

When ``lightsim`` is installed, this module translates between LightSim's
Gymnasium / PettingZoo interface and our EV corridor environment's
internal network representation (see :mod:`src.envs.network_utils`).

If ``lightsim`` is **not** installed, :func:`is_lightsim_available` returns
``False`` and :class:`LightSimAdapter` raises on construction.  All other
helpers remain safe to import.
"""

from __future__ import annotations

import logging
from typing import Any

import networkx as nx
import numpy as np

from src.envs.network_utils import Network

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

def is_lightsim_available() -> bool:
    """Return ``True`` if the ``lightsim`` package can be imported."""
    try:
        import lightsim  # noqa: F401
        return True
    except ImportError:
        return False


def get_available_scenarios() -> list[str]:
    """List the scenario IDs that lightsim exposes.

    Returns an empty list when lightsim is not installed.
    """
    if not is_lightsim_available():
        return []

    try:
        import lightsim
        # lightsim.registry is the conventional place scenarios are listed.
        # Fall back to a hard-coded default if the attribute is missing.
        if hasattr(lightsim, "registry"):
            return list(lightsim.registry.keys())
        return ["grid-4x4-v0", "arterial-6-v0", "corridor-8-v0"]
    except Exception:
        logger.warning("lightsim is importable but scenario listing failed", exc_info=True)
        return []


# ---------------------------------------------------------------------------
# Adapter class
# ---------------------------------------------------------------------------

class LightSimAdapter:
    """Wraps a ``lightsim`` env to provide our :data:`Network`-compatible interface.

    The adapter exposes the same dictionary structure that the built-in CTM
    functions in :mod:`src.envs.network_utils` produce, so the EV corridor
    environments can swap backends transparently.

    Parameters
    ----------
    scenario : str
        LightSim scenario ID (e.g. ``"grid-4x4-v0"``).
    kwargs
        Forwarded to ``lightsim.make()``.
    """

    def __init__(self, scenario: str = "grid-4x4-v0", **kwargs: Any) -> None:
        try:
            import lightsim
        except ImportError as exc:
            raise ImportError(
                "lightsim is required for LightSimAdapter but is not installed. "
                "Install it with:  pip install lightsim"
            ) from exc

        self._scenario = scenario
        self._ls_env = lightsim.make(scenario, **kwargs)

        # --- Topology extraction ---
        # lightsim exposes its topology as a dict of nodes and edges.
        ls_topo = self._ls_env.topology

        # Build our canonical network dict from lightsim's topology
        self._network = self._build_network(ls_topo)

        # Maps between lightsim IDs and our canonical IDs
        self._ls_node_to_ours: dict[str, str] = {}
        self._our_node_to_ls: dict[str, str] = {}
        self._ls_link_to_ours: dict[str, str] = {}
        self._our_link_to_ls: dict[str, str] = {}
        self._build_id_maps(ls_topo)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def network(self) -> Network:
        """Return the network dict compatible with :mod:`network_utils`."""
        return self._network

    def reset(self, rng: np.random.Generator | None = None) -> None:
        """Reset the lightsim env and synchronise densities into our network."""
        seed = int(rng.integers(0, 2**31)) if rng is not None else None
        obs, _info = self._ls_env.reset(seed=seed)
        self._sync_state_from_lightsim(obs)

    def step(self, phase_actions: dict[str, int]) -> None:
        """Apply phase actions and advance one simulation step.

        Parameters
        ----------
        phase_actions : dict[str, int]
            Mapping from our node IDs to the desired signal phase.
        """
        ls_actions = self._translate_actions(phase_actions)
        obs, _reward, _term, _trunc, _info = self._ls_env.step(ls_actions)
        self._sync_state_from_lightsim(obs)

    def get_link_density(self, link_id: str) -> float:
        """Get current density on *link_id* (veh/m/lane)."""
        return float(self._network["links"][link_id]["density"])

    def set_phase(self, node_id: str, phase: int) -> None:
        """Set the signal phase at an intersection in our network dict.

        This does **not** push the phase to lightsim immediately; it will
        be sent on the next :meth:`step` call via ``phase_actions``.
        """
        node = self._network["nodes"][node_id]
        node["current_phase"] = phase % node["num_phases"]

    def get_obs(self) -> dict[str, Any]:
        """Return lightsim observations mapped to our format.

        Returns a dict with keys ``"link_densities"`` and
        ``"signal_phases"`` using our canonical IDs.
        """
        link_densities: dict[str, float] = {}
        for lid, lk in self._network["links"].items():
            link_densities[lid] = lk["density"]

        signal_phases: dict[str, int] = {}
        for nid, nd in self._network["nodes"].items():
            signal_phases[nid] = nd["current_phase"]

        return {
            "link_densities": link_densities,
            "signal_phases": signal_phases,
        }

    # ------------------------------------------------------------------
    # Network construction from lightsim topology
    # ------------------------------------------------------------------

    @staticmethod
    def _build_network(ls_topo: dict[str, Any]) -> Network:
        """Translate a lightsim topology dict into our :data:`Network` format.

        lightsim's topology is expected to contain:
        - ``nodes``: list of dicts with at least ``id``, ``x``, ``y``,
          and optionally ``num_phases``.
        - ``edges`` (or ``links``): list of dicts with ``id``, ``source``,
          ``target``, ``length``, ``lanes``, ``speed_limit``.
        """
        G = nx.DiGraph()
        nodes: dict[str, dict[str, Any]] = {}
        links: dict[str, dict[str, Any]] = {}

        # Default CTM parameters (can be overridden per-link if lightsim
        # provides them).
        default_v_free = 15.0   # m/s
        default_w = 5.0         # m/s
        default_k_jam = 0.15    # veh/m/lane
        default_length = 200.0  # m
        default_lanes = 2

        # -- Nodes --
        ls_nodes = ls_topo.get("nodes", [])
        for i, ln in enumerate(ls_nodes):
            nid = str(ln.get("id", f"n{i}"))
            num_phases = int(ln.get("num_phases", 4))
            is_boundary = bool(ln.get("is_boundary", False))
            nodes[nid] = {
                "id": nid,
                "row": ln.get("row", ln.get("y", 0)),
                "col": ln.get("col", ln.get("x", 0)),
                "is_boundary": is_boundary,
                "current_phase": 0,
                "num_phases": num_phases,
                "incoming_links": [],
                "outgoing_links": [],
            }
            G.add_node(nid)

        # -- Links / edges --
        ls_edges = ls_topo.get("edges", ls_topo.get("links", []))
        for j, le in enumerate(ls_edges):
            lid = str(le.get("id", f"link_{j}"))
            src = str(le["source"])
            dst = str(le["target"])
            length = float(le.get("length", default_length))
            num_lanes = int(le.get("lanes", le.get("num_lanes", default_lanes)))
            v_free = float(le.get("speed_limit", le.get("v_free", default_v_free)))
            w = float(le.get("w", default_w))
            k_jam = float(le.get("k_jam", default_k_jam))
            capacity = v_free * k_jam * num_lanes

            direction = str(le.get("direction", "E"))
            phase_index = {"N": 0, "S": 0, "E": 1, "W": 1}.get(direction, 0)

            links[lid] = {
                "id": lid,
                "source": src,
                "target": dst,
                "length": length,
                "num_lanes": num_lanes,
                "v_free": v_free,
                "w": w,
                "k_jam": k_jam,
                "capacity": capacity,
                "density": 0.0,
                "flow": 0.0,
                "direction": direction,
                "phase_index": phase_index,
            }
            G.add_edge(src, dst, link_id=lid, weight=length)
            if src in nodes:
                nodes[src]["outgoing_links"].append(lid)
            if dst in nodes:
                nodes[dst]["incoming_links"].append(lid)

        return {
            "nodes": nodes,
            "links": links,
            "graph": G,
            "rows": 0,  # not meaningful for arbitrary topologies
            "cols": 0,
            "link_length": default_length,
            "v_free": default_v_free,
            "w": default_w,
            "k_jam": default_k_jam,
            "num_lanes": default_lanes,
        }

    # ------------------------------------------------------------------
    # ID mapping
    # ------------------------------------------------------------------

    def _build_id_maps(self, ls_topo: dict[str, Any]) -> None:
        """Build bidirectional mappings between lightsim and our IDs.

        If lightsim uses the same string IDs we do, the maps are trivial
        identity mappings.  Otherwise each lightsim ID is paired with its
        counterpart in ``self._network``.
        """
        ls_nodes = ls_topo.get("nodes", [])
        our_node_ids = list(self._network["nodes"].keys())
        for i, ln in enumerate(ls_nodes):
            ls_id = str(ln.get("id", f"n{i}"))
            our_id = our_node_ids[i] if i < len(our_node_ids) else ls_id
            self._ls_node_to_ours[ls_id] = our_id
            self._our_node_to_ls[our_id] = ls_id

        ls_edges = ls_topo.get("edges", ls_topo.get("links", []))
        our_link_ids = list(self._network["links"].keys())
        for j, le in enumerate(ls_edges):
            ls_id = str(le.get("id", f"link_{j}"))
            our_id = our_link_ids[j] if j < len(our_link_ids) else ls_id
            self._ls_link_to_ours[ls_id] = our_id
            self._our_link_to_ls[our_id] = ls_id

    # ------------------------------------------------------------------
    # State synchronisation
    # ------------------------------------------------------------------

    def _sync_state_from_lightsim(self, obs: Any) -> None:
        """Pull densities and signal phases from a lightsim observation.

        The exact observation format depends on the lightsim version.
        We handle three common layouts:

        1. **dict** with ``"link_densities"`` and ``"signal_phases"`` keys.
        2. **dict** with ``"density"`` (flat array) and ``"phase"`` keys.
        3. **ndarray** — treated as a flat density vector ordered by link
           index (no signal info available; we keep our local phases).
        """
        if isinstance(obs, dict):
            # Layout 1: keyed by our/lightsim IDs
            if "link_densities" in obs:
                for ls_lid, density in obs["link_densities"].items():
                    our_lid = self._ls_link_to_ours.get(str(ls_lid), str(ls_lid))
                    if our_lid in self._network["links"]:
                        self._network["links"][our_lid]["density"] = float(density)
            # Layout 2: flat arrays
            elif "density" in obs:
                our_link_ids = list(self._network["links"].keys())
                for i, d in enumerate(obs["density"]):
                    if i < len(our_link_ids):
                        self._network["links"][our_link_ids[i]]["density"] = float(d)

            # Signal phases
            if "signal_phases" in obs:
                for ls_nid, phase in obs["signal_phases"].items():
                    our_nid = self._ls_node_to_ours.get(str(ls_nid), str(ls_nid))
                    if our_nid in self._network["nodes"]:
                        self._network["nodes"][our_nid]["current_phase"] = int(phase)
            elif "phase" in obs:
                our_node_ids = list(self._network["nodes"].keys())
                for i, p in enumerate(obs["phase"]):
                    if i < len(our_node_ids):
                        self._network["nodes"][our_node_ids[i]]["current_phase"] = int(p)

        elif hasattr(obs, "__len__"):
            # Layout 3: flat ndarray of densities
            our_link_ids = list(self._network["links"].keys())
            for i in range(min(len(obs), len(our_link_ids))):
                self._network["links"][our_link_ids[i]]["density"] = float(obs[i])

    # ------------------------------------------------------------------
    # Action translation
    # ------------------------------------------------------------------

    def _translate_actions(self, phase_actions: dict[str, int]) -> Any:
        """Convert our ``{node_id: phase}`` dict into lightsim's action format.

        lightsim environments typically accept either:
        - A dict mapping lightsim node IDs to phase ints, **or**
        - A flat numpy array ordered by node index.

        We try the dict form first; if the env rejects it we fall back to
        the array form.
        """
        # Build a dict keyed by lightsim node IDs
        ls_actions: dict[str, int] = {}
        for our_nid, phase in phase_actions.items():
            ls_nid = self._our_node_to_ls.get(our_nid, our_nid)
            ls_actions[ls_nid] = int(phase)

        # If the lightsim env has a dict-based action space, return the dict.
        # Otherwise convert to an array in the order lightsim expects.
        action_space = getattr(self._ls_env, "action_space", None)
        if action_space is not None and hasattr(action_space, "n"):
            # Discrete action space — flatten to array
            ordered_ls_nodes = [
                self._our_node_to_ls.get(nid, nid)
                for nid in self._network["nodes"]
            ]
            arr = np.zeros(len(ordered_ls_nodes), dtype=np.int64)
            for i, ls_nid in enumerate(ordered_ls_nodes):
                arr[i] = ls_actions.get(ls_nid, 0)
            return arr

        # Default: return the dict
        return ls_actions
