import type { Metadata } from "next";
import { IBM_Plex_Mono, Instrument_Serif, Inter } from "next/font/google";
import { AppShell } from "@/components/AppShell";
import "./globals.css";

/**
 * Three faces, each doing one job:
 *
 * - **Instrument Serif** for display. High-contrast and editorial — it reads as
 *   a document rather than a dashboard, which is the point.
 * - **Inter** for interface prose. Neutral at small sizes where the serif's
 *   contrast becomes noise.
 * - **IBM Plex Mono** for anything the system asserts about its own work:
 *   chunk ids, hashes, trace events. Monospace is the signal that you are
 *   looking at machinery rather than at law.
 */
const display = Instrument_Serif({
  variable: "--font-display",
  subsets: ["latin"],
  weight: "400",
});

const body = Inter({ variable: "--font-body", subsets: ["latin"] });

const mono = IBM_Plex_Mono({
  variable: "--font-mono-stack",
  subsets: ["latin"],
  weight: ["400", "500"],
});

export const metadata: Metadata = {
  title: "VidhiAI — grounded legal reasoning for Indian law",
  description:
    "Audit contracts against Indian statutes and research case law. Every citation is verified against its source text before it is shown.",
};

/**
 * Applies the stored theme before first paint.
 *
 * Without this the page renders in the system theme and then swaps, which is a
 * visible flash for anyone who chose the non-default. It must be inline and
 * blocking: a deferred script runs after first paint, which is the thing being
 * avoided.
 */
const THEME_SCRIPT = `
try {
  var t = localStorage.getItem('theme');
  if (t === 'dark' || t === 'light') document.documentElement.dataset.theme = t;
} catch (e) {}
`;

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${display.variable} ${body.variable} ${mono.variable} h-full`}
      suppressHydrationWarning
    >
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME_SCRIPT }} />
      </head>
      <body className="min-h-full">
        <AppShell>{children}</AppShell>
      </body>
    </html>
  );
}
