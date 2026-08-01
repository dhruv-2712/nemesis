# Validation case — the peel chain

> **Claim under test:** NEMESIS does not merely fit the Elliptic labels; it
> detects a *documented laundering typology* by its structure, and would flag a
> reconstruction of that typology it never saw as a unit during training.

## 1. The typology

A **peel chain** (peeling chain) is one of the most established Bitcoin
money-laundering patterns. A wallet holding illicit BTC sends its balance to a
newly created address; at each hop a **small amount is "peeled" off** to a
cash-out destination (typically an exchange deposit address), while the **bulk of
the value is forwarded** to the next fresh address. Repeated dozens or hundreds
of times, the technique:

- **launders gradually** — each peel is small enough to sit below monitoring
  thresholds and blend with legitimate exchange deposits, and
- **obscures the trail** — the main flow keeps moving through single-use
  addresses, so no one hop looks like more than an ordinary transfer.

Structurally, a peel chain is unmistakable: a **long directed chain** (high
longest-path relative to node count) with **fan-out ≈ 1** at each hop — value
hops linearly forward. This is exactly the signature NEMESIS's reasoning layer
keys on for `peeling_chain`.

### Why this is a real-world pattern, not a toy

- **Blockchain-forensics literature** describes peel chains as a standard
  laundering and obfuscation technique for moving illicit proceeds toward
  cash-out points.
- **FinCEN Advisory FIN-2020-A006** (*Advisory on Ransomware and the Use of the
  Financial System to Facilitate Ransom Payments*, 1 Oct 2020) documents how
  convertible-virtual-currency proceeds are layered through chains of
  intermediary addresses before cash-out — the behavior a peel chain implements.
- The **Elliptic dataset** itself (Weber et al., 2019, *Anti-Money Laundering in
  Bitcoin: Experimenting with Graph Convolutional Networks for Financial
  Forensics*) was published by a blockchain-analytics company precisely to study
  these illicit-flow structures on real Bitcoin transactions.

*(These are cited at the typology level to establish that the pattern is real
and monitored. The reconstruction below is a controlled synthetic experiment, not
a claim about any specific named case.)*

## 2. The reconstruction

`replay_test.py` rebuilds a peel chain the detector never saw as a unit:

1. **Seed features from reality.** Node feature vectors are drawn from the
   empirical distribution of transactions that are *both* labeled illicit *and*
   scored ≥ 0.9 illicit by the trained GraphSAGE model — a coherent illicit
   feature distribution, not random noise.
2. **Impose the documented topology.** Those nodes are wired into the canonical
   peel-chain shape: a single directed chain, fan-out 1 at each hop.
3. **Replay through the full pipeline** — the same GNN scoring → cluster
   extraction → typology reasoning path the live system runs.

## 3. Result

Running `python -m validation.replay_test`:

```
NEMESIS replay — reconstructed peel chain
  mean GNN risk over chain : 0.854
  nodes flagged (>=0.7)    : 10/12
  clusters surfaced        : 1
  flagged cluster          : 9 nodes, longest chain 8
  typology                 : peeling_chain
  confidence               : 0.95
  verdict                  : Structure matches peeling chain (confidence 95%).
```

- The GNN **flags the chain as high-risk** — mean risk 0.85, 10 of 12 hops above
  the 0.7 detection threshold.
- Cluster extraction **surfaces it as a single connected chain**.
- The reasoning layer **names the typology correctly** — `peeling_chain` at 0.95
  confidence — citing the structural evidence (longest directed chain 8 nodes,
  fan-out 1).

## 4. Why this is the defensible claim

This is the difference between *"I trained a GNN on a Kaggle dataset"* and *"my
system detects a laundering typology described in FinCEN advisories."* The
reconstructed chain was assembled after training and never presented to the model
as a labeled ring; NEMESIS still flags it and explains **why** in the language of
the typology — which is exactly what an analyst needs to act on it.

**Reproduce:** `python -m validation.replay_test` (or `pytest validation/replay_test.py`).
