"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";

const MODULES = [
  { href: "/compliance", name: "ComplianceGuard" },
  { href: "/caselens", name: "CaseLens" },
];

function ThemeToggle() {
  // The inline script in the root layout applies the stored theme before
  // paint, so the DOM is the source of truth. State exists only to re-render
  // this control; reading localStorage during render would differ between
  // server and client.
  const [theme, setTheme] = useState<"light" | "dark" | null>(null);

  const toggle = () => {
    const current =
      theme ??
      (document.documentElement.dataset.theme as "light" | "dark" | undefined) ??
      (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");

    const next = current === "dark" ? "light" : "dark";
    setTheme(next);
    localStorage.setItem("theme", next);
    document.documentElement.dataset.theme = next;
  };

  return (
    <button
      onClick={toggle}
      className="rounded p-1.5 text-muted transition-colors hover:text-[var(--foreground)]"
      aria-label="Toggle colour theme"
      title="Toggle colour theme"
    >
      <svg
        width="15"
        height="15"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.6"
        aria-hidden
      >
        <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
      </svg>
    </button>
  );
}

/** The mark. A shield, because the product's claim is that nothing unverified gets through. */
function Wordmark() {
  return (
    <Link href="/" className="flex items-center gap-2.5">
      <span
        className="grid size-7 place-items-center rounded font-serif text-sm text-white"
        style={{ background: "var(--color-gold-600)" }}
        aria-hidden
      >
        V
      </span>
      <span className="font-serif text-lg leading-none">VidhiAI</span>
    </Link>
  );
}

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const active = MODULES.find((m) => pathname.startsWith(m.href));

  return (
    <div className="flex min-h-full flex-col">
      <header className="sticky top-0 z-20 border-b bg-[var(--surface)]">
        <div className="mx-auto flex h-14 max-w-[1400px] items-center gap-4 px-5 sm:px-8">
          <Wordmark />

          {/* The active module reads as a path, not a tab bar — this is a
              workspace, and the breadcrumb says where you are in it. */}
          {active && (
            <>
              <span className="text-muted" aria-hidden>
                /
              </span>
              <span className="label-mono">{active.name}</span>
            </>
          )}

          <nav className="ml-auto flex items-center gap-1" aria-label="Modules">
            {MODULES.map((module) => {
              const isActive = pathname.startsWith(module.href);
              return (
                <Link
                  key={module.href}
                  href={module.href}
                  aria-current={isActive ? "page" : undefined}
                  className={`rounded px-2.5 py-1.5 text-sm transition-colors ${
                    isActive
                      ? "text-[var(--foreground)]"
                      : "text-muted hover:text-[var(--foreground)]"
                  }`}
                >
                  {module.name}
                </Link>
              );
            })}
            <span className="mx-1 h-4 w-px bg-[var(--border)]" aria-hidden />
            <ThemeToggle />
          </nav>
        </div>
      </header>

      <main className="mx-auto w-full max-w-[1400px] flex-1 px-5 py-8 sm:px-8">
        {children}
      </main>

      <footer className="border-t px-5 py-3.5 sm:px-8">
        <div className="mx-auto flex max-w-[1400px] flex-wrap items-center justify-between gap-2">
          <span className="label-mono">VidhiAI / India</span>
          <span className="label-mono">
            Decision support · not legal advice
          </span>
        </div>
      </footer>
    </div>
  );
}
