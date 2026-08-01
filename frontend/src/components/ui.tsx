import type { ReactNode } from "react";
import type { Risk, Stance } from "@/lib/api";

/**
 * Shared primitives.
 *
 * Every status indicator pairs colour with a text label. Risk and stance are
 * the load-bearing signals in this UI, and colour alone fails for roughly one
 * in twelve men — and in greyscale print, which legal work still produces.
 */

const RISK_STYLES: Record<Risk, { bg: string; fg: string; label: string }> = {
  high: { bg: "var(--risk-high-bg)", fg: "var(--color-risk-high)", label: "High risk" },
  medium: {
    bg: "var(--risk-medium-bg)",
    fg: "var(--color-risk-medium)",
    label: "Medium risk",
  },
  low: { bg: "var(--risk-low-bg)", fg: "var(--color-risk-low)", label: "Low risk" },
};

export function RiskBadge({ risk, source }: { risk: Risk; source?: string }) {
  const style = RISK_STYLES[risk];
  return (
    <span
      className="inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-2xs font-semibold tracking-wide uppercase"
      style={{ background: style.bg, color: style.fg }}
    >
      <span
        aria-hidden
        className="size-1.5 rounded-full"
        style={{ background: style.fg }}
      />
      {style.label}
      {/* Surfaced because the classifier and the LLM disagree sometimes, and
          a reviewer should know which one is speaking. */}
      {source?.startsWith("classifier") && (
        <span className="font-normal normal-case opacity-70">· model</span>
      )}
    </span>
  );
}

const STANCE_STYLES: Record<Stance, { bg: string; fg: string; label: string }> = {
  supports: {
    bg: "var(--supports-bg)",
    fg: "var(--color-supports)",
    label: "Supports",
  },
  undermines: {
    bg: "var(--undermines-bg)",
    fg: "var(--color-undermines)",
    label: "Undermines",
  },
  neutral: {
    bg: "var(--surface-sunken)",
    fg: "var(--muted)",
    label: "Neutral",
  },
};

export function StanceBadge({
  stance,
  confidence,
}: {
  stance: Stance;
  confidence?: number;
}) {
  const style = STANCE_STYLES[stance];
  return (
    <span
      className="inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-2xs font-semibold tracking-wide uppercase"
      style={{ background: style.bg, color: style.fg }}
    >
      <span
        aria-hidden
        className="size-1.5 rounded-full"
        style={{ background: style.fg }}
      />
      {style.label}
      {confidence !== undefined && (
        <span className="font-normal normal-case opacity-70">
          · {Math.round(confidence * 100)}%
        </span>
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
    "inline-flex items-center justify-center gap-2 rounded-lg px-4 py-2 text-sm font-medium transition-colors duration-150 disabled:cursor-not-allowed disabled:opacity-50";
  const variants = {
    primary:
      "bg-[var(--color-brand-600)] text-white hover:bg-[var(--color-brand-700)]",
    secondary:
      "border border-[var(--border)] bg-[var(--surface)] hover:bg-[var(--surface-sunken)]",
    ghost: "hover:bg-[var(--surface-sunken)]",
  };
  return (
    <button className={`${base} ${variants[variant]}`} {...props}>
      {children}
    </button>
  );
}

export function Card({
  children,
  className = "",
}: {
  children: ReactNode;
  className?: string;
}) {
  return <div className={`surface p-5 ${className}`}>{children}</div>;
}

/**
 * Empty states carry an instruction, not just an absence. A blank panel makes
 * the user wonder whether something failed.
 */
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
    <div className="flex flex-col items-center justify-center gap-3 px-6 py-14 text-center">
      <p className="text-lg font-medium">{title}</p>
      <p className="max-w-md text-sm text-muted">{description}</p>
      {action}
    </div>
  );
}

export function ErrorState({
  message,
  onRetry,
}: {
  message: string;
  onRetry?: () => void;
}) {
  return (
    <div
      role="alert"
      className="flex flex-col items-start gap-3 rounded-lg border p-4"
      style={{
        background: "var(--risk-high-bg)",
        borderColor: "var(--color-risk-high)",
      }}
    >
      <p className="text-sm" style={{ color: "var(--color-risk-high)" }}>
        {message}
      </p>
      {onRetry && (
        <Button variant="secondary" onClick={onRetry}>
          Try again
        </Button>
      )}
    </div>
  );
}

export function Skeleton({ className = "" }: { className?: string }) {
  return <div className={`skeleton rounded ${className}`} aria-hidden />;
}

/**
 * A cited source. Clicking is not wired to a hover-card yet; the citation text
 * itself is the guarantee, since it is only ever rendered for findings whose
 * quote the backend verified against the retrieved chunk.
 */
export function Citation({ children }: { children: ReactNode }) {
  return (
    <cite className="text-xs font-medium not-italic text-[var(--color-brand-600)]">
      {children}
    </cite>
  );
}

export function Quote({ children }: { children: ReactNode }) {
  return (
    <blockquote
      className="legal-text border-l-2 py-1 pl-3 text-muted"
      style={{ borderColor: "var(--color-brand-300)" }}
    >
      {children}
    </blockquote>
  );
}
