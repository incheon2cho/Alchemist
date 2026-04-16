"""External knowledge retrievers for Benchmark + Research agents.

Design principles:
  - Only FREE public APIs (no paid keys) — preserves Alchemist's $0-agent story.
  - Shared by both agents: Benchmark (model scouting) and Research (SoTA/techniques).
  - File-cached to avoid redundant calls within a session.
  - Policy filters (ImageNet-1K only, val-not-test) enforced at retriever level.
"""

from .arxiv import ArxivRetriever
from .hf_hub import HFHubRetriever

__all__ = ["ArxivRetriever", "HFHubRetriever"]
