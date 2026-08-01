"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";

const MODULES = [
  {
    href: "/compliance",
    name: "ComplianceGuard",
    description: "Audit contracts against Indian statutes",
  },
  {
    href: "/caselens",
    name: "CaseLens",
    description: "Research precedents for a fact pattern",
  },
];

function ThemeToggle() {
  // The inline script in the root layout has already applied the stored theme
  // to <html> before paint, so the DOM is the source of truth here. State only
  // exists to re-render the control; it is read from the DOM rather than from
  // localStorage during render, which would differ between server and client.
  const [theme, setTheme] = useState<"light" | "dark" | null>(null);

  const toggle = () => {
    const current =
      theme ??
      (document.documentElement.dataset.theme as "light" | "dark" | undefined) ??
      (window.matchMedia("(prefers-color-scheme: dark)").matches
        ? "dark"
        : "light");

    const next = current === "dark" ? "light" : "dark";

    setTheme(next);
    localStorage.setItem("theme", next);
    document.documentElement.dataset.theme = next;
  };

  return (
    <button
      onClick={toggle}
      className="rounded-lg p-2 text-muted transition-colors hover:bg-[var(--surface-sunken)]"
      aria-label="Toggle colour theme"
      title="Toggle colour theme"
    >
      <svg
        width="16"
        height="16"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        aria-hidden
      >
        <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
      </svg>
    </button>
  );
}

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();

  return (
    <div className="flex min-h-full flex-col">
      <header className="sticky top-0 z-10 border-b bg-[var(--surface)]/85 backdrop-blur-sm">
        <div className="mx-auto flex h-14 max-w-6xl items-center gap-6 px-4 sm:px-6">
          <Link href="/" className="flex items-baseline gap-2 font-semibold">
            VidhiAI
            <span className="text-2xs font-normal text-muted">Indian law</span>
          </Link>

          <nav className="flex items-center gap-1" aria-label="Modules">
            {MODULES.map((module) => {
              const active = pathname.startsWith(module.href);
              return (
                <Link
                  key={module.href}
                  href={module.href}
                  aria-current={active ? "page" : undefined}
                  className={`rounded-lg px-3 py-1.5 text-sm transition-colors ${
                    active
                      ? "bg-[var(--surface-sunken)] font-medium"
                      : "text-muted hover:bg-[var(--surface-sunken)]"
                  }`}
                >
                  {module.name}
                </Link>
              );
            })}
          </nav>

          <div className="ml-auto flex items-center gap-1">
            <ThemeToggle />
          </div>
        </div>
      </header>

      <main className="mx-auto w-full max-w-6xl flex-1 px-4 py-8 sm:px-6">
        {children}
      </main>

      <footer className="border-t px-4 py-4 text-center text-xs text-muted sm:px-6">
        Decision support, not legal advice. Every citation is verified against
        the retrieved source text.
      </footer>
    </div>
  );
}
