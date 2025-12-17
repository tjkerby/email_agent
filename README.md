## Email Agent

This repo shows how to pipe an email thread through LangChain while using a local Ollama
model (default: `llama3`) to draft polished replies.

### Prerequisites
- Python 3.12+
- [Ollama](https://ollama.com/download) running locally (`ollama serve`)
- Model pulled locally, e.g. `ollama pull llama3`
- `langchain-ollama` Python package (installed automatically via `pip install -e .`)

### Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

### Run the sample flow
```bash
python main.py
```
The script prints a starter email from the scenario, prompts you to type the student's
reply (finish input with a line containing just `.`), then grades it and shows the AI
manager's response plus the JSON payload that would be stored.

### Streamlit web app
```bash
streamlit run app.py
```
Then open the provided local URL. Pick a scenario from the dropdown, review its context,
draft your student email in the text area, and click **Grade my email** to see rubric
scores, qualitative feedback, the AI counterpart's reply, and the Supabase-ready JSON.

### Changing the grading rubric
- Rubric definitions live in `rubrics/` as JSON or YAML files. Each file should contain a
	root object with optional `name`/`description` fields and an `items` list. Every item
	needs `name`, `description`, and an optional `max_score` (defaults to 5).
- The CLI entry point (`python main.py`) now loads `rubrics/default.json`. Edit or
	replace that file to change the CLI grading behavior.
- The Streamlit app automatically lists every rubric file in `rubrics/`. Drop additional
	JSON/YAML files into that directory and they become selectable in the sidebar.

Example rubric snippet:

```json
{
	"name": "communication_basics",
	"description": "Checks for tone, clarity, ownership, and follow-up.",
	"items": [
		{"name": "Tone", "description": "Respectful and empathetic", "max_score": 5},
		{"name": "Clarity", "description": "Clear ask or next step"}
	]
}
```

### Use the responder in your own code
```python
from email_agent import EmailMessage, EmailResponder

thread = [
		EmailMessage(sender="alex@acme.com", subject="Status?", body="Any update on the report?"),
]

responder = EmailResponder(model="llama3", temperature=0.1)
reply = responder.respond(thread, instructions="Confirm delivery tomorrow morning.")
print(reply)
```

#### Tips
- Pass `base_url="http://localhost:11434"` if Ollama is bound to a non-default address.
- Use `.respond_async` inside async frameworks (FastAPI, FastAPI background tasks, etc.).
- Use `.batch_respond` when you want to score multiple threads in parallel to reduce
	network round-trips to Ollama.
