import Link from "next/link";

const MODULES = [
  {
    href: "/compliance",
    name: "ComplianceGuard",
    tagline: "Audit a contract against Indian statutes",
    detail:
      "Upload or paste a contract. Each clause is checked against the ingested Acts, and every flag cites the exact provision it relies on.",
  },
  {
    href: "/caselens",
    name: "CaseLens",
    tagline: "Find precedents for a fact pattern",
    detail:
      "Describe the facts and the position you are arguing. Retrieved judgments are assessed for whether they support or undermine it.",
  },
];

export default function Home() {
  return (
    <div className="mx-auto max-w-3xl py-8">
      <h1 className="text-3xl font-semibold tracking-tight">
        Grounded AI for Indian law
      </h1>
      <p className="mt-3 max-w-2xl text-muted">
        Two modules over one retrieval core. Every claim either cites a
        provision that was actually retrieved and quotes it verbatim, or it is
        discarded before you see it.
      </p>

      <div className="mt-10 grid gap-4 sm:grid-cols-2">
        {MODULES.map((module) => (
          <Link
            key={module.href}
            href={module.href}
            className="surface flex flex-col gap-2 p-5 transition-colors hover:border-[var(--color-brand-300)]"
          >
            <h2 className="font-medium">{module.name}</h2>
            <p className="text-sm font-medium text-[var(--color-brand-600)]">
              {module.tagline}
            </p>
            <p className="text-sm text-muted">{module.detail}</p>
            <span className="mt-auto pt-3 text-sm font-medium text-[var(--color-brand-600)]">
              Open →
            </span>
          </Link>
        ))}
      </div>

      <section className="mt-12">
        <h2 className="text-xs font-semibold tracking-wide uppercase text-muted">
          How grounding works
        </h2>
        <ol className="mt-4 space-y-3 text-sm">
          <li className="flex gap-3">
            <span className="font-mono text-xs text-muted">01</span>
            <span>
              Retrieval combines dense vectors with lexical search, so a
              question phrased in plain English still reaches the provision that
              governs it.
            </span>
          </li>
          <li className="flex gap-3">
            <span className="font-mono text-xs text-muted">02</span>
            <span>
              A critic judges whether the retrieved law actually addresses the
              clause, and reformulates the search when it does not.
            </span>
          </li>
          <li className="flex gap-3">
            <span className="font-mono text-xs text-muted">03</span>
            <span>
              Every citation is checked mechanically: the chunk must have been
              retrieved for this run, and the quote must appear in it. Claims
              that fail go back to be re-grounded, then are discarded.
            </span>
          </li>
        </ol>
      </section>
    </div>
  );
}
