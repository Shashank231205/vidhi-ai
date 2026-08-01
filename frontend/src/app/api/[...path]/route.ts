/**
 * Proxy to the backend API.
 *
 * This exists instead of a `rewrites` entry because rewrites buffer the whole
 * response before forwarding it. That is invisible for JSON, but it defeats
 * the point of SSE: measured against the sample contract, the first trace
 * event arrived at 49.7s — exactly the total run time — so the UI showed
 * "Starting…" for the entire audit and then rendered every event at once.
 * Piping the body through explicitly delivers the first event in ~60ms.
 *
 * Everything else the proxy buys us still holds: the browser sees one origin,
 * so there is no CORS to configure, and the backend URL never reaches the
 * client.
 */

const API_ORIGIN = (process.env.API_ORIGIN ?? "http://127.0.0.1:8010").replace(
  /\/$/,
  "",
);

/** Hop-by-hop headers must not be forwarded; the runtime sets its own. */
const STRIPPED = new Set([
  "connection",
  "content-encoding",
  "content-length",
  "host",
  "keep-alive",
  "transfer-encoding",
]);

function forwardHeaders(source: Headers): Headers {
  const headers = new Headers();
  source.forEach((value, key) => {
    if (!STRIPPED.has(key.toLowerCase())) headers.set(key, value);
  });
  return headers;
}

async function proxy(request: Request, path: string[]): Promise<Response> {
  const incoming = new URL(request.url);
  const target = `${API_ORIGIN}/${path.join("/")}${incoming.search}`;

  let response: Response;
  try {
    response = await fetch(target, {
      method: request.method,
      headers: forwardHeaders(request.headers),
      body: request.body,
      // Required by undici whenever a request carries a streaming body.
      ...(request.body ? { duplex: "half" } : {}),
      // No caching layer between the client and an agent run.
      cache: "no-store",
      redirect: "manual",
    });
  } catch {
    return Response.json(
      {
        detail:
          "Could not reach the API. If it is deployed on a free tier it may be waking up; retry in a moment.",
      },
      { status: 502 },
    );
  }

  const headers = forwardHeaders(response.headers);
  // Tell any intermediary — a CDN, or Vercel's own edge — not to buffer.
  // Without this the stream can be re-collected downstream of us.
  if (headers.get("content-type")?.includes("text/event-stream")) {
    headers.set("cache-control", "no-cache, no-transform");
    headers.set("x-accel-buffering", "no");
  }

  // response.body is passed through unread, so chunks reach the client as the
  // backend emits them.
  return new Response(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers,
  });
}

type Context = { params: Promise<{ path: string[] }> };

export async function GET(request: Request, context: Context) {
  return proxy(request, (await context.params).path);
}

export async function POST(request: Request, context: Context) {
  return proxy(request, (await context.params).path);
}

export async function PUT(request: Request, context: Context) {
  return proxy(request, (await context.params).path);
}

export async function PATCH(request: Request, context: Context) {
  return proxy(request, (await context.params).path);
}

export async function DELETE(request: Request, context: Context) {
  return proxy(request, (await context.params).path);
}
