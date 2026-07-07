# Part 4 — LLM System with Production Guardrails
## Road Safety Advisor — End-to-End Application

## What is this project about?

This part turns the core LLM logic from Part 3 into a complete, runnable
application — the **Road Safety Advisor**. A user can describe accident
conditions (weather, lighting, alcohol involvement, speed, road type)
through a simple command-line interface, and the system returns an
AI-generated risk assessment.

The focus of this part isn't just "does it work" but "does it work
*reliably*" — handling bad input gracefully, recovering from failed API
calls, respecting rate limits, and keeping a record of what happened. These
are the concerns that separate a classroom script from a system someone
could actually depend on.

---

## What's inside this folder?

| File | What it does |
|---|---|
| `app.py` | Main entry point — runs the interactive command-line application |
| `validators.py` | Checks and cleans user input before it's sent anywhere |
| `llm_client.py` | Talks to the Groq LLM API — handles retries, timeouts, rate limiting, and usage tracking |
| `schema.py` | Defines the exact JSON structure the AI's response must follow |
| `logger_setup.py` | Sets up logging to both the console and a file |
| `test_validators.py` | Automated tests confirming the input validation works correctly |
| `logs/app.log` | Generated automatically when the app runs — a record of every request |
| `.env.example` | Documents the required environment variable name (no real key inside) |
| `requirements.txt` | Python packages needed to run this project |

---

## How to run this yourself

1. **Get a free Groq API key** at [console.groq.com/keys](https://console.groq.com/keys)
   (no credit card required).

2. **Set up your environment**:
   ```
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Create your own `.env` file** (copy `.env.example`, rename to `.env`,
   and paste in your real key):
   ```
   GROQ_API_KEY=your_actual_key_here
   ```

4. **Run the application**:
   ```
   python app.py
   ```
   Answer the prompts (weather, lighting, alcohol involvement, speed, road
   type) to get a risk assessment. Type `quit` at any prompt to exit.

5. **Run the automated tests** (optional, confirms validation logic works):
   ```
   python test_validators.py
   ```

---

## System scope & user flow

**What this system does:** Takes a description of road/accident conditions
from a user and returns a structured AI risk assessment (risk level,
contributing factors, explanation, and a safety recommendation).

**User flow:**
1. User is prompted for 5 pieces of information about the conditions.
2. The system validates this input before doing anything else.
3. If valid, the system calls the LLM (with retries and a timeout built in)
   and displays the structured result.
4. If invalid, the system explains what was wrong and asks again — it never
   crashes on bad input.
5. This repeats until the user types `quit`, at which point a summary of
   the session's API usage is shown.

**What this system explicitly does *not* do:** it does not store any
personal data, does not make automated decisions on anyone's behalf, and
does not claim to be a substitute for official road safety guidance — it's
an educational risk-awareness tool.

---

## Guardrails implemented (and why each one matters)

### 1. Input validation & sanitization (`validators.py`)
Every field is checked against an allowed set of values (e.g. weather must
be one of Clear/Rainy/Foggy/Stormy/Hazy) or a sensible numeric range (speed
between 1–200 km/h). Text input is normalized (trimmed, title-cased) before
checking, so `"rainy "` and `"Rainy"` are both accepted.

**Why:** Sending unvalidated user input straight to an LLM risks confusing
or wasted API calls, unpredictable responses, and in worse cases, prompt
injection. Catching bad input early is cheaper and safer than handling it
after the fact.

### 2. Retries with exponential backoff (`llm_client.py`, via `tenacity`)
If the LLM returns malformed JSON, or the request fails, the system
automatically retries up to 3 times, waiting slightly longer between each
attempt (2s, then 4s, then 8s).

**Why:** LLM APIs occasionally return imperfect output or hit transient
network issues. Failing immediately on the first hiccup would make the
system unreliable; blind, instant retries could make rate-limiting worse.
Backoff strikes a balance.

### 3. Timeouts
Every API call has a 15-second timeout.

**Why:** Without this, a slow or hanging API response could freeze the
whole application indefinitely. A timeout guarantees the system always
either succeeds or fails within a bounded time.

### 4. Rate limiting
The client enforces a minimum 2-second gap between consecutive API calls.

**Why:** This respects the API provider's usage limits and avoids the
system accidentally hammering the API with rapid-fire requests, which could
get the API key throttled or blocked.

### 5. Logging (`logger_setup.py`)
Every request, successful response, validation failure, and error is logged
with a timestamp — both to the console and to `logs/app.log`.

**Why:** If something goes wrong days later, logs are how you find out what
happened, when, and why — without them, debugging a deployed system is
largely guesswork.

### 6. Usage tracking
Each API response includes token usage data, which the system accumulates
and reports at the end of a session (total requests, total tokens).

**Why:** LLM APIs are usage-billed. In a real deployment, tracking usage is
essential for cost monitoring and catching runaway usage early.

### 7. Graceful failure handling
Validation errors, API failures, and unexpected exceptions are all caught
explicitly — none of them crash the program. The user sees a clear message
and can simply try again.

**Why:** A production system should never crash outright on a recoverable
error; it should fail safely and let the user (or operator) recover.

---

## Testing

`test_validators.py` contains 6 automated test cases covering: valid input,
an invalid category value, non-numeric input, an out-of-range value, a
missing required field, and correct normalization of lowercase input. All
6 tests pass, confirming the validation logic behaves as intended across
both valid and invalid scenarios.

---

## Design decisions

- **Command-line interface, not a web app** — kept the interaction simple
  and dependency-light, since the focus of this part is the guardrails
  themselves, not UI complexity.
- **`tenacity` for retries** instead of a hand-rolled retry loop, since it's
  a well-tested, widely-used library for exactly this purpose in production
  Python code.
- **In-memory rate limiting/usage tracking** — resets each time the program
  restarts. For a longer-running or multi-user service, this would need to
  move to persistent storage (e.g. a database or Redis), noted as a
  limitation below.

## Known Limitations

- Rate limiting and usage tracking are in-memory only — they reset if the
  program restarts, and wouldn't correctly coordinate limits across multiple
  simultaneous users if this were deployed as a shared service.
- The system assumes a single interactive user at a time; it isn't built as
  a concurrent, multi-user server.
- Logs are stored locally as plain text; a production deployment would
  typically ship these to a centralized logging service.