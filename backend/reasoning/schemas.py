"""Structured I/O for the NEMESIS reasoning layer.

The GNN flags a *cluster* (a connected group of high-risk transactions). This
module defines the contract between the graph side and the LLM side:

  ClusterFeatures  — structural summary the GNN/graph produces for one cluster
  TypologyVerdict  — the structured judgement the LLM returns for that cluster

Keeping both as Pydantic models means the FastAPI layer (Phase 4) and the React
frontend render confidence scores and typology labels consistently, and the LLM
is constrained to valid, machine-readable output rather than free prose.

Typology note: the original project brief listed account-based typologies (mule
network, synthetic-identity farm, card-testing ring). The dataset pivoted to the
Elliptic *Bitcoin transaction* graph (homogeneous tx -> tx), so the typologies
here are the Bitcoin-laundering patterns that actually fit that topology. The
mechanism — structural features -> LLM -> typology + confidence + reasoning
chain — is unchanged.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Typology(str, Enum):
    """Bitcoin money-laundering patterns distinguishable by graph structure."""

    PEELING_CHAIN = "peeling_chain"
    """Long directed chain; each hop peels off a small amount and forwards the
    rest — classic layering to move value while shedding taint incrementally."""

    MIXING_TUMBLING = "mixing_tumbling"
    """Dense many-to-many core (high fan-in and fan-out) that obscures the
    input->output mapping — CoinJoin / tumbler behaviour."""

    CONSOLIDATION_CASHOUT = "consolidation_cashout"
    """Many inputs converging on one or few sink nodes — funds gathered for a
    single cash-out (e.g. to an exchange deposit)."""

    SCAM_PAYOUT_FANOUT = "scam_payout_fanout"
    """One source distributing to many recipients — scam/ponzi payout or
    dusting from a single controlled wallet."""

    LAYERING = "layering"
    """Multi-hop transfers with no single dominant shape, creating distance from
    the illicit source without an obvious peel/consolidate signature."""

    UNKNOWN_SUSPICIOUS = "unknown_suspicious"
    """Flagged as structurally anomalous but not matching a clean typology."""


class ClusterFeatures(BaseModel):
    """Structural summary of one flagged cluster — the LLM's only input."""

    cluster_id: str
    num_nodes: int
    num_edges: int
    density: float = Field(..., description="edges / possible directed edges")
    avg_illicit_prob: float = Field(..., description="mean GNN P(illicit) over cluster")
    max_illicit_prob: float
    mean_in_degree: float
    mean_out_degree: float
    max_in_degree: int = Field(..., description="fan-in hotspot — consolidation signal")
    max_out_degree: int = Field(..., description="fan-out hotspot — distribution signal")
    longest_chain: int = Field(..., description="longest directed path — peeling signal")
    reciprocity: float = Field(..., description="fraction of edges that are bidirectional")
    num_sources: int = Field(..., description="nodes with in-degree 0")
    num_sinks: int = Field(..., description="nodes with out-degree 0")
    time_span_steps: int = Field(..., description="distinct Elliptic time steps spanned")

    def to_prompt_block(self) -> str:
        """Render as a compact, LLM-friendly feature list."""
        return (
            f"cluster_id: {self.cluster_id}\n"
            f"nodes: {self.num_nodes}, edges: {self.num_edges}, density: {self.density:.3f}\n"
            f"GNN illicit probability — avg: {self.avg_illicit_prob:.2f}, "
            f"max: {self.max_illicit_prob:.2f}\n"
            f"degree — mean_in: {self.mean_in_degree:.2f}, mean_out: {self.mean_out_degree:.2f}, "
            f"max_in: {self.max_in_degree}, max_out: {self.max_out_degree}\n"
            f"longest_directed_chain: {self.longest_chain}\n"
            f"reciprocity: {self.reciprocity:.2f}\n"
            f"sources (in-deg 0): {self.num_sources}, sinks (out-deg 0): {self.num_sinks}\n"
            f"time_span: {self.time_span_steps} step(s)"
        )


class TypologyVerdict(BaseModel):
    """The LLM's structured judgement for a flagged cluster."""

    typology: Typology
    confidence: float = Field(..., ge=0.0, le=1.0)
    summary: str = Field(..., description="one-sentence plain-language verdict")
    reasoning_chain: list[str] = Field(
        ..., description="ordered structural observations that justify the typology"
    )
    recommended_action: str = Field(
        ..., description="what an analyst should do next (e.g. SAR filing, monitor)"
    )
