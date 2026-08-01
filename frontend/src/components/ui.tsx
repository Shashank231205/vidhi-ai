import type { ReactNode } from "react";
import type { Risk, Stance } from "@/lib/api";

/**
 * Shared primitives.
 *
 * Every verdict indicator pairs colour with a text label. Risk and stance are
 * the load-bearing signals here, and colour alone fails for roughly one in
 * twelve men — and in greyscale, which legal work still prints in.
 */

const RISK: Record<Risk, { label: string; hue: string }> = {
  high: { label: "High risk", hue: "var(--color-risk-high)" },
  medium: { label: "Medium risk", hue: "var(--color-risk-medium)" },
  low: { label: "Low risk", hue: "var(--color-risk-low)" },
};

export function RiskBadge({ risk, source }: { risk: Risk; source?: string }) {
  const { label, hue } = RISK[risk];
  return (
    <span
      className="inline-flex items-center gap-1.5 rounded px-2 py-1 font-mono text-2xs font-medium tracking-wider uppercase"
      style={{
        color: risk === "high" ? "white" : hue,
        background:
          risk === "high" ? hue : `color-mix(in oklch, ${hue} 12%, transparent)`,
      }}
    >
      {label}
      {/* Surfaced because the classifier and the LLM sometimes disagree, and a
          reviewer should know which one produced this level. */}
      {source?.startsWith("classifier") && (
        <span className="font-normal opacity-70">· model</span>
      )}
    </span>
  );
}

const STANCE: Record<Stance, { label: string; hue: string }> = {
  supports: { label: "Supports", hue: "var(--color-supports)" },
  undermines: { label: "Undermines", hue: "var(--color-undermines)" },
  neutral: { label: "Neutral", hue: "var(--muted)" },
};

export function StanceBadge({ stance }: { stance: Stance }) {
  const { label, hue } = STANCE[stance];
  const solid = stance !== "neutral";
  return (
    <span
      className="inline-flex items-center rounded px-2.5 py-1 font-mono text-2xs font-medium tracking-wider uppercase"
      style={{
        color: solid ? "white" : hue,
        background: solid ? hue : "var(--surface-sunken)",
      }}
    >
      {label}
    </span>
  );
}

/** The verification mark. Used only where a citation actually passed the check. */
export function VerifiedMark({ label }: { label?: string }) {
  return (
    <span
      className="inline-flex items-center gap-1.5"
      style={{ color: "var(--color-verified)" }}
      title="Quote matched against the retrieved source text"
    >
      <svg
        width="16"
        height="16"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.8"
        aria-hidden
      >
        <circle cx="12" cy="12" r="9" />
        <path d="m8.5 12.5 2.5 2.5 4.5-5" />
      </svg>
      {label && (
        <span className="font-mono text-2xs tracking-wider uppercase">{label}</span>
      )}
    </span>
  );
}

export function Button({
  children,
  variant = "primary",
  ...props
}: React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "primary" | "secondary" | "ghost";
}) {
  const base =
    "inline-flex items-center justify-center gap-2 rounded px-4 py-2 text-sm transition-colors duration-150 disabled:cursor-not-allowed disabled:opacity-45";
  const variants = {
    primary:
      "bg-[var(--color-gold-600)] text-white hover:bg-[var(--color-gold-700)]",
    secondary:
      "border border-[var(--border-strong)] bg-[var(--surface)] hover:bg-[var(--surface-sunken)]",
    ghost: "text-muted hover:text-[var(--foreground)]",
  };
  return (
    <button className={`${base} ${variants[variant]}`} {...props}>
      {children}
    </button>
  );
}

/** A small uppercase monospace label. The main texture of this design. */
export function Label({ children }: { children: ReactNode }) {
  return <p className="label-mono">{children}</p>;
}

export function Panel({
  children,
  className = "",
  sunken = false,
}: {
  children: ReactNode;
  className?: string;
  sunken?: boolean;
}) {
  return (
    <div
      className={`rounded-[var(--radius-card)] border ${className}`}
      style={{ background: sunken ? "var(--surface-sunken)" : "var(--surface)" }}
    >
      {children}
    </div>
  );
}

/** Empty states carry an instruction. A blank panel reads as a failure. */
export function EmptyState({
  title,
  description,
  action,
}: {
  title: string;
  description: string;
  action?: ReactNode;
}) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 px-6 py-16 text-center">
      <h3 className="font-serif text-xl">{title}</h3>
      <p className="max-w-sm text-sm text-muted">{description}</p>
      {action}
    </div>
  );
}

export function ErrorState({
  title = "Something went wrong",
  message,
  onRetry,
}: {
  title?: string;
  message: string;
  onRetry?: () => void;
}) {
  return (
    <div
      role="alert"
      className="rounded-[var(--radius-card)] border p-5"
      style={{
        borderColor: "var(--color-risk-high)",
        background: "color-mix(in oklch, var(--color-risk-high) 7%, transparent)",
      }}
    >
      <h3 className="font-serif text-lg">{title}</h3>
      <p className="mt-1.5 text-sm text-muted">{message}</p>
      {onRetry && (
        <div className="mt-4">
          <Button variant="secondary" onClick={onRetry}>
            Retry
          </Button>
        </div>
      )}
    </div>
  );
}

export function Skeleton({ className = "" }: { className?: string }) {
  return <div className={`skeleton rounded ${className}`} aria-hidden />;
}
