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
