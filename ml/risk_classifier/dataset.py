"""Risk-classifier dataset, built from CUAD.

CUAD (Contract Understanding Atticus Dataset) is 13,155 clauses extracted from
real commercial contracts by practising attorneys, labelled with 41 clause
types. It is the standard academic benchmark for contract clause understanding.

CUAD labels clause *type*, not risk, so the mapping below is the modelling
decision that makes it usable here — and the one most worth arguing with. It is
recorded explicitly rather than buried in a notebook so a lawyer can disagree
with a specific line.

The mapping asks one question of each clause type: **if this clause is
one-sided or missing a limit, what is the exposure?**

- HIGH   — uncapped or unbounded exposure, loss of control over the entity or
           its IP, or a restraint on trade. These are the terms that get a deal
           blocked in review.
- MEDIUM — real obligations with bounded exposure: money, term, territory.
           Negotiable, not existential.
- LOW    — administrative and definitional terms. Wrong ones cause friction,
           not liability.

A caveat that matters for how the numbers are read: this is US commercial
contract data, and the risk weighting reflects contract-drafting exposure
generally rather than Indian statutory violation specifically. The classifier
scores *clause severity*; the statutory grounding comes from retrieval and the
citation verifier, which are Indian-law-specific.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from enum import StrEnum

import httpx

CUAD_DATASET = "dvgodoy/CUAD_v1_Contract_Understanding_clause_classification"
PARQUET_API = "https://datasets-server.huggingface.co/parquet"


class RiskLevel(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


#: CUAD clause type -> risk level. See the module docstring for the rationale.
RISK_BY_CLAUSE_TYPE: dict[str, RiskLevel] = {
    # Unbounded or unquantifiable exposure.
    "Uncapped Liability": RiskLevel.HIGH,
    "Unlimited/All-You-Can-Eat-License": RiskLevel.HIGH,
    "Irrevocable Or Perpetual License": RiskLevel.HIGH,
    "Liquidated Damages": RiskLevel.HIGH,
    # Loss of control over the entity or its core assets.
    "Change Of Control": RiskLevel.HIGH,
    "Ip Ownership Assignment": RiskLevel.HIGH,
    "Joint Ip Ownership": RiskLevel.HIGH,
    "Source Code Escrow": RiskLevel.HIGH,
    # Restraints on trade — the terms most likely to be unenforceable, and in
    # India specifically void under s.27 of the Contract Act.
    "Non-Compete": RiskLevel.HIGH,
    "Exclusivity": RiskLevel.HIGH,
    "Competitive Restriction Exception": RiskLevel.HIGH,
    "No-Solicit Of Customers": RiskLevel.HIGH,
    "Most Favored Nation": RiskLevel.HIGH,
    # Bounded but material obligations.
    "Cap On Liability": RiskLevel.MEDIUM,
    "Minimum Commitment": RiskLevel.MEDIUM,
    "Revenue/Profit Sharing": RiskLevel.MEDIUM,
    "Price Restrictions": RiskLevel.MEDIUM,
    "Volume Restriction": RiskLevel.MEDIUM,
    "Insurance": RiskLevel.MEDIUM,
    "Audit Rights": RiskLevel.MEDIUM,
    "Anti-Assignment": RiskLevel.MEDIUM,
    "Termination For Convenience": RiskLevel.MEDIUM,
    "Post-Termination Services": RiskLevel.MEDIUM,
    "Rofr/Rofo/Rofn": RiskLevel.MEDIUM,
    "Covenant Not To Sue": RiskLevel.MEDIUM,
    "No-Solicit Of Employees": RiskLevel.MEDIUM,
    "Non-Disparagement": RiskLevel.MEDIUM,
    "Third Party Beneficiary": RiskLevel.MEDIUM,
    "License Grant": RiskLevel.MEDIUM,
    "Non-Transferable License": RiskLevel.MEDIUM,
    "Affiliate License-Licensee": RiskLevel.MEDIUM,
    "Affiliate License-Licensor": RiskLevel.MEDIUM,
    "Warranty Duration": RiskLevel.MEDIUM,
    "Governing Law": RiskLevel.MEDIUM,
    # Administrative and definitional.
    "Parties": RiskLevel.LOW,
    "Document Name": RiskLevel.LOW,
    "Agreement Date": RiskLevel.LOW,
    "Effective Date": RiskLevel.LOW,
    "Expiration Date": RiskLevel.LOW,
    "Renewal Term": RiskLevel.LOW,
    "Notice Period To Terminate Renewal": RiskLevel.LOW,
}


@dataclass(slots=True)
class LabelledClause:
    text: str
    risk: RiskLevel
    clause_type: str


def _parquet_url(dataset: str) -> str:
    response = httpx.get(PARQUET_API, params={"dataset": dataset}, timeout=60)
    response.raise_for_status()
    files = response.json()["parquet_files"]
    if not files:
        raise RuntimeError(f"no parquet export for {dataset}")
    return str(files[0]["url"])


def load_cuad(*, min_chars: int = 40, max_chars: int = 2000) -> list[LabelledClause]:
    """Download CUAD and map it onto risk levels.

    Very short spans are dropped: CUAD marks some answers as a date or a party
    name, which carries no clause semantics for a model to learn from. Very long
    ones are truncated by the tokenizer anyway and skew the length distribution.
    """
    import pandas as pd

    raw = httpx.get(_parquet_url(CUAD_DATASET), timeout=300, follow_redirects=True)
    raw.raise_for_status()
    frame = pd.read_parquet(io.BytesIO(raw.content))

    clauses: list[LabelledClause] = []
    unmapped: set[str] = set()

    for row in frame.itertuples():
        label = str(row.label)
        risk = RISK_BY_CLAUSE_TYPE.get(label)
        if risk is None:
            unmapped.add(label)
            continue

        text = " ".join(str(row.clause).split())
        if not (min_chars <= len(text) <= max_chars):
            continue

        clauses.append(LabelledClause(text=text, risk=risk, clause_type=label))

    if unmapped:
        # Loud rather than silent: an unmapped label means the dataset changed
        # and the mapping above needs a decision, not a default.
        raise RuntimeError(
            f"CUAD labels missing from RISK_BY_CLAUSE_TYPE: {sorted(unmapped)}"
        )

    return clauses
