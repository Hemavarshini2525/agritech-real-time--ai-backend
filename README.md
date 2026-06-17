# Real-Time-AI-Crop-Advisory-System-for-Farmers

## Gemini / AI forwarding endpoint

The backend exposes a simple endpoint to forward small prompts/queries to a Gemini-compatible API and return the response to the frontend.

Environment variables required:

- `GEMINI_API_KEY` — API key or bearer token used for authentication.
- `GEMINI_MODEL` — Gemini model name, for example `gemini-1.5`.
- `GEMINI_API_URL` — optional custom Gemini API base URL.

Example request:

```bash
curl -X POST http://localhost:8000/ai-query \
	-H "Content-Type: application/json" \
	-d '{"query":"What crops suit sandy loam soil with pH 6.5?"}'
```

The backend sends the prompt text to Gemini using the official SDK and returns the model response. If `GEMINI_API_URL` is set, the backend uses it as the SDK base URL; otherwise it uses the SDK default endpoint.