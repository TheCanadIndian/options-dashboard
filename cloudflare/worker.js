// Runs the scanner without the PC: every 15 minutes in US market hours it starts the GitHub
// Actions scan, and it keeps the scanner's state (journal, accounts, caches) in R2 between runs.
//
//   GET  /state   the latest state archive          (Authorization: Bearer STATE_TOKEN)
//   PUT  /state   replace it; the old one is kept as state-prev.tgz

const LATEST = "state.tgz";
const PREVIOUS = "state-prev.tgz";

export default {
  async scheduled(controller, env, ctx) {
    if (!inMarketWindow(new Date(controller.scheduledTime))) return;
    const r = await fetch(`https://api.github.com/repos/${env.REPO}/actions/workflows/scan.yml/dispatches`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${env.DISPATCH_TOKEN}`,
        Accept: "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "options-scan-trigger",
      },
      body: JSON.stringify({ ref: "main" }),
    });
    if (!r.ok) throw new Error(`GitHub dispatch failed: ${r.status} ${await r.text()}`);
  },

  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.pathname !== "/state") return new Response("not found", { status: 404 });
    if (request.headers.get("Authorization") !== `Bearer ${env.STATE_TOKEN}`) {
      return new Response("unauthorised", { status: 401 });
    }
    if (request.method === "GET") {
      const obj = await env.STATE.get(url.searchParams.get("previous") ? PREVIOUS : LATEST);
      if (!obj) return new Response("no state yet", { status: 404 });
      return new Response(obj.body, { headers: { "Content-Type": "application/gzip", "Last-Modified": obj.uploaded.toUTCString() } });
    }
    if (request.method === "PUT") {
      const current = await env.STATE.get(LATEST);
      if (current) await env.STATE.put(PREVIOUS, current.body);
      await env.STATE.put(LATEST, request.body);
      return new Response("saved");
    }
    return new Response("method not allowed", { status: 405 });
  },
};

// Monday to Friday, 9:30 to 16:00 New York time. Holidays are left to the scanner, which does
// nothing when the market is closed.
function inMarketWindow(now) {
  const parts = Object.fromEntries(
    new Intl.DateTimeFormat("en-US", {
      timeZone: "America/New_York", weekday: "short", hour: "2-digit", minute: "2-digit", hourCycle: "h23",
    }).formatToParts(now).map((p) => [p.type, p.value]),
  );
  if (parts.weekday === "Sat" || parts.weekday === "Sun") return false;
  const minutes = Number(parts.hour) * 60 + Number(parts.minute);
  return minutes >= 9 * 60 + 30 && minutes <= 16 * 60;
}
