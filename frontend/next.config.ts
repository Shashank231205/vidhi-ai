import type { NextConfig } from "next";

/**
 * The backend runs as a separate process, but the browser only ever talks to
 * this origin: `/api/*` is proxied through to it.
 *
 * Two things fall out of that, both worth having:
 *
 * - **One URL.** In development you open localhost:3000 and everything works;
 *   there is no second port to remember or to get wrong.
 * - **No CORS.** Same-origin requests need no preflight and no allow-list, so
 *   the deployed frontend does not have to be registered with the API.
 *
 * The destination is configured, not hardcoded — in production it points at
 * the HuggingFace Space hosting the API (Vercel cannot host it: ~3GB of models
 * against a 250MB function limit, and audits that run for minutes against a
 * 10s timeout).
 */

const API_ORIGIN = (
  process.env.API_ORIGIN ?? "http://127.0.0.1:8010"
).replace(/\/$/, "");

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${API_ORIGIN}/:path*`,
      },
    ];
  },
};

export default nextConfig;
