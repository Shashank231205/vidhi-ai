import Link from "next/link";
import { Label, VerifiedMark } from "@/components/ui";

const STAGES = ["Retrieve", "Critic", "Analyze", "Verify"];

const MODULES = [
  {
    href: "/compliance",
    index: "01",
    name: "ComplianceGuard",
    description: "Audit contracts clause by clause against Indian statutes.",
    steps: "Upload · Trace · Verify",
  },
  {
    href: "/caselens",
    index: "02",
    name: "CaseLens",
    description: "Rank precedents by stance, authority and verified source text.",
    steps: "Frame · Research · Argue",
  },
];

export default function Home() {
  return (
    <div className="py-6">
      <div className="grid gap-12 lg:grid-cols-[1fr_22rem] lg:gap-16">
        <div>
          <Label>Legal research, with receipts</Label>

          <h1 className="mt-5 max-w-2xl text-4xl leading-[1.06] sm:text-5xl">
            Reasoning that holds up under cross-examination.
          </h1>

          <p className="mt-6 max-w-xl text-muted">
            VidhiAI traces every conclusion through retrieval, critique,
            analysis and mechanical verification against Indian statutes and
            judicial sources.
          </p>

          {/* The pipeline, named. It is the product's actual argument, so it
              belongs above the fold rather than in a features list. */}
          <ol className="mt-8 flex flex-wrap gap-2">
            {STAGES.map((stage, index) => {
              const last = index === STAGES.length - 1;
              return (
                <li
                  key={stage}
                  className="flex items-center gap-1.5 rounded border px-3 py-1.5 font-mono text-2xs tracking-wider uppercase"
                  style={
                    last
                      ? {
                          background: "var(--color-verified)",
                          borderColor: "var(--color-verified)",
                          color: "white",
                        }
                      : { background: "var(--surface)" }
                  }
                >
                  <span className="opacity-55">0{index + 1}</span>
                  {last && <span aria-hidden>✓</span>}
                  {stage}
                </li>
              );
            })}
          </ol>
        </div>

        {/* A worked example of provenance, not decoration: this is exactly the
            shape of what the trace emits during a real run. */}
        <aside
          className="h-fit rounded-[var(--radius-card)] border p-5"
          style={{ background: "var(--surface-sunken)" }}
        >
          <Label>Live provenance</Label>

          <ul className="mt-4 space-y-2.5 font-mono text-xs">
            <li className="flex gap-2.5 text-muted">
              <span style={{ color: "var(--accent)" }}>↳</span>
              DPDP Act 2023 · §8 retrieved
            </li>
            <li className="flex gap-2.5 text-muted">
              <span style={{ color: "var(--accent)" }}>↳</span>
              official source chunk matched
            </li>
            <li className="flex gap-2.5" style={{ color: "var(--color-verified)" }}>
              <span aria-hidden>✓</span>
              quote verified against source
            </li>
          </ul>

          <div className="mt-5 flex items-center justify-between border-t pt-4">
            <Label>Verified source</Label>
            <VerifiedMark />
          </div>
        </aside>
      </div>

      <div className="mt-14 grid gap-4 sm:grid-cols-2">
        {MODULES.map((module) => (
          <Link
            key={module.href}
            href={module.href}
            className="group rounded-[var(--radius-card)] border p-6 transition-colors hover:border-[var(--border-strong)]"
            style={{ background: "var(--surface-sunken)" }}
          >
            <Label>Module {module.index}</Label>
            <h2 className="mt-3 text-xl">{module.name}</h2>
            <p className="mt-2 text-sm text-muted">{module.description}</p>
            <p
              className="mt-6 font-mono text-2xs tracking-wider uppercase"
              style={{ color: "var(--accent)" }}
            >
              {module.steps} →
            </p>
          </Link>
        ))}
      </div>
    </div>
  );
}
