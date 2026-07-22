# Integrations

Modelship speaks the OpenAI API (`/v1/chat/completions`, `/v1/responses`,
`/v1/embeddings`, `/v1/audio/*`, `/v1/images/generations`), so anything that
already talks to OpenAI — an SDK, a chat UI, a low-code agent builder — can
point at Modelship instead by changing its base URL.

All of these guides assume a running Modelship gateway reachable at
`http://<host>:8000` (see [Quickstart](../quickstart.md) if you don't have
one yet), and use `Authorization: Bearer <key>` for auth if `MSHIP_API_KEYS`
is set — otherwise any non-empty string works as the API key, since most
OpenAI-compatible clients require *some* value in that field even when the
server doesn't check it.

- [OpenAI SDK](openai-sdk.md) — Python and JS/TS, the direct integration path
- [Open WebUI](open-webui.md) — a self-hosted chat UI
- [Dify](dify.md) — a low-code agent/app builder
- [n8n](n8n.md) — workflow automation
- [Responses-speaking agents](responses-agents.md) — agent frameworks and
  custom loops built on `/v1/responses` specifically
