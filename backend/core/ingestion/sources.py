"""Registry of Indian statutory sources.

Acts are fetched from India Code (indiacode.nic.in), the Government of India's
official repository, and from the relevant ministry sites. These are public
PDFs with no API key, no rate limit, and no per-document billing — which is why
the PRD's original Indian Kanoon plan was dropped.

Sources are declared as data rather than code so adding an Act is a one-line
change, and so ingestion, eval, and the API can all enumerate the same corpus.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class LegalDomain(StrEnum):
    """What an Act governs — used to scope retrieval to relevant law.

    A contract-compliance question should not have to rank against securities
    regulation, so the domain narrows the candidate set before ranking.
    """

    DATA_PROTECTION = "data_protection"
    CONTRACT = "contract"
    CORPORATE = "corporate"
    TECHNOLOGY = "technology"
    CONSUMER = "consumer"
    EMPLOYMENT = "employment"
    DISPUTE_RESOLUTION = "dispute_resolution"
    INTELLECTUAL_PROPERTY = "intellectual_property"
    COMPETITION = "competition"
    TAXATION = "taxation"


#: India Code is a DSpace instance. An Act's *handle* is a stable identifier;
#: the PDF path underneath it is not — the sequence number changes when a
#: bitstream is revised, and several Acts carry Hindi and English versions
#: under the same handle. Storing the handle and resolving the PDF at fetch
#: time is therefore the only form that survives; hardcoded PDF URLs were
#: already broken for 9 of the 12 Acts when first checked.
INDIA_CODE_HANDLE = "https://www.indiacode.nic.in/handle/123456789"


@dataclass(frozen=True, slots=True)
class StatuteSource:
    key: str
    title: str
    source_ref: str
    year: int
    act_number: str
    domain: LegalDomain
    #: DSpace handle id on India Code; the PDF is resolved from this at fetch
    #: time. Mutually exclusive with `url`.
    handle: int | None = None
    #: Direct PDF URL, for sources not hosted on India Code.
    url: str | None = None
    #: Ordering hint for the demo corpus: 1 is ingested first.
    priority: int = 2

    def __post_init__(self) -> None:
        if not (self.handle or self.url):
            raise ValueError(f"{self.key}: needs a handle or a url")

    @property
    def handle_url(self) -> str | None:
        return f"{INDIA_CODE_HANDLE}/{self.handle}" if self.handle else None

    @property
    def meta(self) -> dict[str, object]:
        return {
            "year": self.year,
            "act_number": self.act_number,
            "domain": self.domain.value,
            "jurisdiction": "India",
        }


#: Central Acts most relevant to contract compliance auditing. Priority 1 is
#: the set ComplianceGuard needs to be useful at all; the rest broaden coverage.
STATUTES: tuple[StatuteSource, ...] = (
    StatuteSource(
        key="dpdp",
        title="Digital Personal Data Protection Act, 2023",
        source_ref="DPDP-2023",
        url="https://www.meity.gov.in/static/uploads/2024/06/2bf1f0e9f04e6fb4f8fef35e82c42aa5.pdf",
        year=2023,
        act_number="22 of 2023",
        domain=LegalDomain.DATA_PROTECTION,
        priority=1,
    ),
    StatuteSource(
        key="contract",
        title="Indian Contract Act, 1872",
        source_ref="CONTRACT-1872",
        handle=2187,
        year=1872,
        act_number="9 of 1872",
        domain=LegalDomain.CONTRACT,
        priority=1,
    ),
    StatuteSource(
        key="it",
        title="Information Technology Act, 2000",
        source_ref="IT-2000",
        handle=1999,
        year=2000,
        act_number="21 of 2000",
        domain=LegalDomain.TECHNOLOGY,
        priority=1,
    ),
    StatuteSource(
        key="companies",
        title="Companies Act, 2013",
        source_ref="COMPANIES-2013",
        handle=2114,
        year=2013,
        act_number="18 of 2013",
        domain=LegalDomain.CORPORATE,
        priority=1,
    ),
    StatuteSource(
        key="consumer",
        title="Consumer Protection Act, 2019",
        source_ref="CONSUMER-2019",
        handle=15256,
        year=2019,
        act_number="35 of 2019",
        domain=LegalDomain.CONSUMER,
        priority=1,
    ),
    StatuteSource(
        key="arbitration",
        title="Arbitration and Conciliation Act, 1996",
        source_ref="ARBITRATION-1996",
        handle=1978,
        year=1996,
        act_number="26 of 1996",
        domain=LegalDomain.DISPUTE_RESOLUTION,
        priority=1,
    ),
    StatuteSource(
        key="copyright",
        title="Copyright Act, 1957",
        source_ref="COPYRIGHT-1957",
        handle=1367,
        year=1957,
        act_number="14 of 1957",
        domain=LegalDomain.INTELLECTUAL_PROPERTY,
        priority=2,
    ),
    StatuteSource(
        key="trade-marks",
        title="Trade Marks Act, 1999",
        source_ref="TRADEMARKS-1999",
        handle=1993,
        year=1999,
        act_number="47 of 1999",
        domain=LegalDomain.INTELLECTUAL_PROPERTY,
        priority=2,
    ),
    StatuteSource(
        key="competition",
        title="Competition Act, 2002",
        source_ref="COMPETITION-2002",
        handle=2010,
        year=2002,
        act_number="12 of 2003",
        domain=LegalDomain.COMPETITION,
        priority=2,
    ),
    StatuteSource(
        key="sale-of-goods",
        title="Sale of Goods Act, 1930",
        source_ref="SALE-OF-GOODS-1930",
        handle=2390,
        year=1930,
        act_number="3 of 1930",
        domain=LegalDomain.CONTRACT,
        priority=2,
    ),
)

BY_KEY: dict[str, StatuteSource] = {s.key: s for s in STATUTES}


def by_priority(max_priority: int = 3) -> list[StatuteSource]:
    return sorted(
        (s for s in STATUTES if s.priority <= max_priority),
        key=lambda s: (s.priority, s.key),
    )
