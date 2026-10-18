# Datheon_Python_-_Generative_AI_-_LangChain_Internship
# LLM Document Extractor

Turns unstructured invoices and resumes into **strictly validated JSON** using Google Gemini or OpenAI, parameterised prompt templates with few-shot examples, and Pydantic schemas.

## How it works

```
document ──► PromptTemplate (system + few-shot user prompt, JSON schema embedded)
         ──► LLM provider (Gemini / OpenAI, JSON mode, temperature 0)
         ──► parse_json (strips code fences / stray text)
         ──► Pydantic validation (types, formats, cross-field maths, no extra keys)
              ├─ valid   → typed Invoice / Resume object
              └─ invalid → errors fed back to the model, retry (default 2 retries)
```

Validation goes beyond types: line totals must equal quantity × unit price, the subtotal must equal the sum of lines, total = subtotal + tax, currency must be an ISO code, emails must be valid, and unknown fields are rejected.

## Project structure

| File | Purpose |
|---|---|
| `llm_extractor/schemas.py` | Pydantic models `Invoice`, `Resume` with validators |
| `llm_extractor/prompts.py` | `PromptTemplate` class, `$placeholder` templates, few-shot examples |
| `llm_extractor/providers.py` | `GeminiProvider`, `OpenAIProvider`, `MockProvider` behind one interface |
| `llm_extractor/extractor.py` | Validate-and-retry loop |
| `main.py` | Command-line interface |
| `tests/` | Offline tests using the mock provider |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # then add your API key
```

Get a free Gemini key from Google AI Studio, or an OpenAI key from the OpenAI platform.

## Usage

```bash
python main.py invoice samples/invoice.txt --provider gemini
python main.py resume  samples/resume.txt  --provider openai -v
```

From Python:

```python
from llm_extractor import analyse, get_provider

invoice = analyse(open("samples/invoice.txt").read(), "invoice", get_provider("gemini"))
print(invoice.total, invoice.currency)
```

## Tests (no API key needed)

```bash
pytest -q
```

## Adding a new document type

1. Add a Pydantic model to `schemas.py`.
2. Add a `FewShotExample` and a branch in `build_template()` in `prompts.py`.

## Notes

- Uses the current `google-genai` SDK; the older `google-generativeai` package is deprecated.
- Model names are configurable via `GEMINI_MODEL` / `OPENAI_MODEL` in `.env`.
- Never commit your `.env` file.
