"""LangGraph reasoning pipeline — cluster structure -> typology verdict.

SPECTER-style StateGraph (reused pattern from ARGUS): a small, explicit graph of
reasoning steps rather than one opaque call. Two nodes:

  classify  — map structural features to a TypologyVerdict. Uses Groq
              llama-3.3-70b when GROQ_API_KEY is set; otherwise (or on any LLM
              error) falls back to a deterministic structural heuristic.
  validate  — guard the output: clamp confidence, ensure a non-empty reasoning
              chain, so downstream (API/frontend) always gets a well-formed verdict.

The offline heuristic means the whole system is runnable and testable without an
API key — drop a key in .env to get the richer LLM narration.
"""

from __future__ import annotations

import os
from typing import Optional, TypedDict

try:
    # Load GROQ_API_KEY (and other vars) from a local .env if present. Optional —
    # the pipeline falls back to the heuristic when no key is found either way.
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from langgraph.graph import END, StateGraph

from backend.reasoning.prompts import SYSTEM_PROMPT, build_user_prompt
from backend.reasoning.schemas import ClusterFeatures, Typology, TypologyVerdict

GROQ_MODEL = "llama-3.3-70b-versatile"


class ReasoningState(TypedDict):
    features: ClusterFeatures
    verdict: Optional[TypologyVerdict]
    used_llm: bool


# --------------------------------------------------------------------------- #
# Deterministic structural heuristic (offline fallback + LLM-failure fallback)
# --------------------------------------------------------------------------- #
def heuristic_verdict(f: ClusterFeatures) -> TypologyVerdict:
    """Classify from structure alone with transparent rules. Always succeeds."""
    chain_ratio = f.longest_chain / f.num_nodes if f.num_nodes else 0.0
    reasoning: list[str] = [
        f"Cluster of {f.num_nodes} transactions with mean GNN illicit probability "
        f"{f.avg_illicit_prob:.2f} (max {f.max_illicit_prob:.2f})."
    ]

    if chain_ratio >= 0.6 and f.max_out_degree <= 2:
        typ = Typology.PEELING_CHAIN
        conf = min(0.6 + 0.4 * chain_ratio, 0.95)
        reasoning.append(
            f"Longest directed chain is {f.longest_chain} of {f.num_nodes} nodes "
            f"({chain_ratio:.0%}) with max fan-out {f.max_out_degree} — value hops "
            "linearly forward, the signature of a peeling chain."
        )
        action = "Trace the chain end-to-end; identify the terminal cash-out address."
    elif f.max_in_degree >= 5 and f.num_sinks <= max(2, f.num_nodes // 10):
        typ = Typology.CONSOLIDATION_CASHOUT
        conf = min(0.55 + 0.05 * f.max_in_degree, 0.9)
        reasoning.append(
            f"One node absorbs {f.max_in_degree} inputs into {f.num_sinks} sink(s) — "
            "many sources funnel into a single consolidation/cash-out point."
        )
        action = "Prioritise the high-in-degree sink for exchange-deposit attribution."
    elif f.max_out_degree >= 5:
        typ = Typology.SCAM_PAYOUT_FANOUT
        conf = min(0.55 + 0.05 * f.max_out_degree, 0.9)
        reasoning.append(
            f"A node distributes to {f.max_out_degree} recipients — one wallet "
            "fanning out, consistent with a scam/ponzi payout."
        )
        action = "Check the source wallet against known scam-address intelligence."
    elif f.density >= 0.15 and f.max_in_degree >= 3 and f.max_out_degree >= 3:
        typ = Typology.MIXING_TUMBLING
        conf = min(0.5 + f.density, 0.9)
        reasoning.append(
            f"Dense core (density {f.density:.2f}, reciprocity {f.reciprocity:.2f}) with "
            f"high fan-in ({f.max_in_degree}) and fan-out ({f.max_out_degree}) — "
            "inputs and outputs are entangled to obscure provenance (mixing/tumbling)."
        )
        action = "Attempt input-output de-anonymisation via amount/timing correlation."
    else:
        typ = Typology.LAYERING
        conf = 0.5
        reasoning.append(
            f"Multi-hop structure (chain {f.longest_chain}, max_in {f.max_in_degree}, "
            f"max_out {f.max_out_degree}) with no single dominant shape — generic "
            "layering to distance funds from their source."
        )
        action = "Expand the cluster one hop and re-score to resolve the pattern."

    if f.time_span_steps > 1:
        reasoning.append(
            f"Activity spans {f.time_span_steps} time steps, consistent with a "
            "deliberate staged flow rather than a single burst."
        )

    return TypologyVerdict(
        typology=typ,
        confidence=round(conf, 2),
        summary=f"Structure matches {typ.value.replace('_', ' ')} "
        f"(confidence {conf:.0%}).",
        reasoning_chain=reasoning,
        recommended_action=action,
    )


# --------------------------------------------------------------------------- #
# LangGraph nodes
# --------------------------------------------------------------------------- #
def _llm_verdict(f: ClusterFeatures) -> TypologyVerdict:
    """Call Groq with structured output. Raises on any failure (caller falls back)."""
    from langchain_groq import ChatGroq

    llm = ChatGroq(model=GROQ_MODEL, temperature=0).with_structured_output(TypologyVerdict)
    messages = [
        ("system", SYSTEM_PROMPT),
        ("human", build_user_prompt(f.to_prompt_block())),
    ]
    return llm.invoke(messages)


def _classify(state: ReasoningState) -> ReasoningState:
    f = state["features"]
    if os.getenv("GROQ_API_KEY"):
        try:
            return {"features": f, "verdict": _llm_verdict(f), "used_llm": True}
        except Exception as exc:  # noqa: BLE001 — degrade gracefully, never crash detection
            print(f"[reasoning] LLM call failed ({exc}); using heuristic.")
    return {"features": f, "verdict": heuristic_verdict(f), "used_llm": False}


def _validate(state: ReasoningState) -> ReasoningState:
    verdict = state["verdict"]
    assert verdict is not None
    verdict.confidence = max(0.0, min(1.0, verdict.confidence))
    if not verdict.reasoning_chain:
        verdict.reasoning_chain = ["No structural detail available."]
    return {**state, "verdict": verdict}


def build_pipeline():
    """Compile the reasoning StateGraph."""
    graph = StateGraph(ReasoningState)
    graph.add_node("classify", _classify)
    graph.add_node("validate", _validate)
    graph.set_entry_point("classify")
    graph.add_edge("classify", "validate")
    graph.add_edge("validate", END)
    return graph.compile()


_PIPELINE = None


def classify_cluster_verbose(features: ClusterFeatures) -> tuple[TypologyVerdict, bool]:
    """Run one cluster through the pipeline; return (verdict, used_llm)."""
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = build_pipeline()
    result = _PIPELINE.invoke({"features": features, "verdict": None, "used_llm": False})
    return result["verdict"], result["used_llm"]


def classify_cluster(features: ClusterFeatures) -> TypologyVerdict:
    """Run one cluster through the reasoning pipeline and return its verdict."""
    return classify_cluster_verbose(features)[0]


if __name__ == "__main__":
    from pathlib import Path

    import torch

    from backend.reasoning.cluster import extract_clusters

    data = torch.load(Path("data/graphs/elliptic.pt"), weights_only=False)
    clusters = extract_clusters(data)
    print(f"{len(clusters)} clusters; reasoning over top 3:\n")
    for c in clusters[:3]:
        v = classify_cluster(c)
        print(f"=== {c.cluster_id} ({c.num_nodes}n/{c.num_edges}e) ===")
        print(f"  typology: {v.typology.value}  confidence: {v.confidence}")
        print(f"  summary: {v.summary}")
        for step in v.reasoning_chain:
            print(f"   - {step}")
        print(f"  action: {v.recommended_action}\n")
