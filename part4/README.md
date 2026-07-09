# Part 4 — LLM-Powered Feature
## Track B: Tabular Record Batch Scoring — Road Accident Risk Review

**Chosen Track: (B) Tabular Record Batch Scoring**

## What is this feature?

Given accident records from the cleaned dataset, this feature calls an LLM
once per record to score it against a safety-review rubric, returning a
structured, validated JSON assessment — the kind of output a safety review
dashboard could consume directly, rather than free-form text a human would
need to manually interpret.

## Files

| File | Purpose |
|---|---|
| `llm_client.py` | `call_llm()` — reusable function to call the Groq API |
| `guardrails.py` | PII detection guardrail (regex-based) |
| `prompts.py` | System prompt (rubric + few-shot example) and user prompt builder |
| `schema.py` | JSON schema for the risk assessment output |
| `batch_scoring.py` | Main pipeline — runs the 3-record demonstration |
| `temperature_comparison.py` | Temperature A/B comparison script |
| `.env.example` | Documents the required environment variable |

## How to run

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file (copy `.env.example`, rename, add your real key):
```
GROQ_API_KEY=your_actual_key_here
```

Run the main pipeline:
```
python batch_scoring.py
```

Run the temperature comparison:
```
python temperature_comparison.py
```

---

## 1. `call_llm` Function

```python
def call_llm(system_prompt, user_prompt, temperature=0.0, max_tokens=512):
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    response = requests.post(URL, headers=headers, json=payload)
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        return None
    return response.json()['choices'][0]['message']['content']
```

**Demonstrated with test prompt:** `call_llm("You are a helpful assistant.", "Reply with only the word: hello")` → returned `"hello"`, confirming the function connects to the API and returns a visible response.

The API key is read from the `GROQ_API_KEY` environment variable via
`python-dotenv` — it is never hardcoded anywhere in the codebase.

---

## 2. System Prompt & User Prompt Template (verbatim)

**System prompt:**
```
You are a road accident risk assessment system for a safety review team. 
Given a JSON record describing an accident's conditions, score it against this rubric:

- risk_tier: "low", "medium", or "high" — based on severity of conditions (alcohol involvement, poor lighting/weather, high speed relative to road type increase risk)
- flag_for_review: true if risk_tier is "high" OR if alcohol was involved, else false
- primary_signal: the single most concerning factor in this record (a short phrase)
- confidence: "low", "medium", or "high" — how confident you are in this assessment given the available fields
- recommended_action: one concrete, short recommendation for a safety reviewer

Respond with ONLY a valid JSON object with exactly these 5 fields, no other text.

Example:
Input: {"Weather Conditions": "Clear", "Lighting Conditions": "Daylight", "Alcohol Involvement": "No", "Speed Limit (km/h)": 40, "Road Type": "Urban Road"}
Output: {"risk_tier": "low", "flag_for_review": false, "primary_signal": "Favorable conditions across the board", "confidence": "high", "recommended_action": "No immediate action needed; routine monitoring sufficient"}
```

**User prompt template:**
```
Input: {record}
```
where `{record}` is a Python dict of the accident's Weather Conditions,
Lighting Conditions, Alcohol Involvement, Speed Limit, and Road Type.

**Why temperature=0:** structured scoring should be consistent and
repeatable — a safety review system should return the same assessment for
the same input every time it runs. Temperature=0 always selects the
highest-probability next token at each generation step, producing
deterministic, predictable output suited to this structured data task.

---

## 3. JSON Schema

```python
RISK_ASSESSMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "risk_tier": {"type": "string", "enum": ["low", "medium", "high"]},
        "flag_for_review": {"type": "boolean"},
        "primary_signal": {"type": "string"},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "recommended_action": {"type": "string"}
    },
    "required": ["risk_tier", "flag_for_review", "primary_signal", "confidence", "recommended_action"]
}
```

Every LLM response is parsed with `json.loads()` inside a
`try/except json.JSONDecodeError` block, then validated against this
schema using `jsonschema.validate()` inside a
`try/except jsonschema.ValidationError` block. On any failure, a fallback
dict (all 5 fields set to `None`) is returned and the error is logged to
the console.

---

## 4. PII Guardrail

```python
def has_pii(text):
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    phone_pattern = r'\b\d{10}\b|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b'
    return bool(re.search(email_pattern, text) or re.search(phone_pattern, text))
```

Run before every `call_llm()` call — if PII is detected, the LLM is never
called and `"Input blocked: PII detected."` is printed.

**Test results:**
| Input | PII Detected | Result |
|---|---|---|
| "Contact driver at john.doe@email.com for details" | True | Blocked |
| "Weather was rainy, speed limit 80 km/h" | False | Proceeded to LLM call |

---

## 5. Batch Scoring Demonstration (3 Records)

| Input Record | LLM Assessment JSON | Validation Status |
|---|---|---|
| Weather=Hazy, Lighting=Dark, Alcohol=Yes, Speed=61km/h, Road=National Highway | `{"risk_tier": "high", "flag_for_review": true, "primary_signal": "Alcohol involvement and hazardous conditions", "confidence": "medium", "recommended_action": "Conduct thorough investigation and review of safety protocols"}` | PASS |
| Weather=Hazy, Lighting=Dusk, Alcohol=Yes, Speed=92km/h, Road=Urban Road | `{"risk_tier": "high", "flag_for_review": true, "primary_signal": "Alcohol involvement and high speed", "confidence": "high", "recommended_action": "Review and consider enforcement action"}` | PASS |
| Weather=Foggy, Lighting=Dawn, Alcohol=No, Speed=120km/h, Road=National Highway | `{"risk_tier": "high", "flag_for_review": true, "primary_signal": "High speed on national highway during poor visibility", "confidence": "medium", "recommended_action": "Review speed management strategies for national highways during low-light conditions"}` | PASS |

All 3 records passed schema validation on the first attempt (no retries
triggered). All 3 were scored "high" risk — this reflects that the
sampled rows happen to have genuinely concerning conditions (alcohol
involvement or poor visibility at high speed), not a bias in the LLM
toward always predicting "high."

---

## 6. Temperature A/B Comparison (temp=0 vs temp=0.7)

| Input | Output (temp=0) | Output (temp=0.7) | Key Difference |
|---|---|---|---|
| Hazy/Dark/Alcohol=Yes/61km/h/Highway | confidence="medium", action="Conduct thorough investigation..." | confidence="high", action="Immediate review and potential intervention necessary" | Same risk_tier/flag; confidence and wording differ, temp=0.7 slightly more urgent tone |
| Hazy/Dusk/Alcohol=Yes/92km/h/Urban | confidence="high", signal="Alcohol involvement and high speed" | confidence="medium", signal="Alcohol involvement and hazardous driving conditions" | Same risk_tier/flag; confidence flipped, phrasing varies |
| Foggy/Dawn/Alcohol=No/120km/h/Highway | action="Review speed management strategies for national highways..." | action="Conduct a thorough review of driver behavior and road conditions" | Same risk_tier/flag/confidence; recommended_action wording differs |

**Why temperature=0 is more deterministic:** the model always picks the
single highest-probability next token at each generation step — given the
same input, this produces the same (or near-identical) output every time,
since no randomness is introduced into token selection.

**Why temperature=0.7 introduces variability:** the model samples from a
broader probability distribution over possible next tokens rather than
always choosing the top candidate, allowing lower-probability (but still
reasonable) phrasings to be selected. This produced varied wording and
occasionally different confidence levels across the 3 test records, even
though the core risk assessment (`risk_tier`, `flag_for_review`) remained
stable in every case.

---

## Design Decisions

- **`jsonschema` over `pydantic`**: used per the rubric's explicit
  specification for this part, providing a lightweight, declarative schema
  definition separate from Python class-based validation.
- **Groq API** (`llama-3.1-8b-instant`): free tier, no billing setup
  required, reused from the same account set up in earlier parts of this
  project.
- **One API call per record** in the main pipeline (not a duplicate
  display call), ensuring the "Raw LLM Response" shown to the user and the
  validated assessment are always the exact same response.

## Known Limitations

- The rubric criteria in the system prompt are authored heuristics (e.g.,
  "alcohol involvement or poor lighting increase risk"), not learned from
  the dataset — Parts 1–3 established this dataset's recorded outcomes
  don't actually correlate with these factors, so this feature should be
  understood as a general-purpose rubric-based reviewer, not a
  data-validated risk predictor.
- Tested on a small sample (3 records); broader validation across more
  records and edge cases (e.g., missing fields) was not performed.