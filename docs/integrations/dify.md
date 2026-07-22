# Dify

[Dify](https://github.com/langgenius/dify) supports adding a custom
**OpenAI-API-compatible** model provider, which is how it talks to Modelship.

1. In Dify, go to **Settings → Model Provider** and add an
   **OpenAI-API-compatible** provider (Dify lists this as a distinct provider
   type from "OpenAI").
2. Set the **API Base** to `http://<modelship-host>:8000/v1`.
3. Set the **API Key** to a real key if `MSHIP_API_KEYS` is configured, or any
   non-empty placeholder if not.
4. Add a model entry for each model name in your `models.yaml` (Dify needs
   the model registered explicitly per provider — it doesn't auto-discover
   via `GET /v1/models`), matching its type (chat, embeddings, etc.) to the
   corresponding Modelship deployment.

Once registered, use that model like any other inside Dify's chatflow/agent
builder. Streaming, embeddings (for Dify's knowledge-base retrieval), and
tool calling all route through the same `/v1` endpoints Modelship already
exposes.
