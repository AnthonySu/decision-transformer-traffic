"""Baseline methods for emergency vehicle corridor optimization."""

from src.baselines.fixed_time_evp import FixedTimeEVP
from src.baselines.greedy_preempt import GreedyPreemptPolicy

__all__ = ["FixedTimeEVP", "GreedyPreemptPolicy"]
