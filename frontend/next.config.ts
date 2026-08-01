import type { NextConfig } from "next";

/**
 * The browser only ever talks to this origin. `/api/*` is proxied to the
 * backend by the route handler in `src/app/api/[...path]/route.ts` — not by a
 * `rewrites` entry, because rewrites buffer the whole response and that
 * defeats SSE.
 *
 * Configure the backend with `API_ORIGIN`. It is deliberately not
 * `NEXT_PUBLIC_`: the proxy runs server-side, so the backend URL never reaches
 * the browser and the API is not directly addressable from it.
 */
const nextConfig: NextConfig = {};

export default nextConfig;
