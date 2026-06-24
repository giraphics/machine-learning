# AI Compliance Checker — Beginner's Guide

This tool lets you upload any PDF or Word document and check it against a list of
compliance rules you write in plain English. An AI model reads each document and
tells you whether each rule passes, fails, or is uncertain — and quotes the exact
text that supports its decision.

You can choose from **four AI providers** — including a free local option:

| Provider | Cost | Best for |
|---|---|---|
| **MiniMax M3** | ~$0.03–0.08 per audit | Best reasoning quality |
| **Groq** | Free tier (14,400 req/day) | Fast, free cloud option |
| **Google Gemini** | Free tier (1,500 req/day) | Good quality, free |
| **Ollama (local)** | Free forever | Privacy, no internet needed |

---

## What You Need Before Starting

| Requirement | What it is | How to get it |
|---|---|---|
| **Python 3.11+** | The programming language this tool runs on | https://www.python.org/downloads/ |
| **An AI API key** | Only needed for cloud providers (not Ollama) | See provider table above |
| **A terminal** | A text-based window to type commands | Windows: press `Win + R`, type `cmd`, press Enter |

---

## Step-by-Step Setup (Do This Once)

### Step 1 — Open your terminal and go to the project folder

```
cd C:\dev\giraphics\machine-learning\MiniMax\compliance-checker
```

> **What this does:** "cd" means "change directory" — it moves you into the project folder,
> like navigating into a folder in Windows Explorer.

---

### Step 2 — Create a virtual environment

```
python -m venv venv
```

> **What this does:** Creates an isolated Python environment just for this project.
> Think of it as a clean room where only this project's packages live.

Then activate it:

**Windows:**
```
venv\Scripts\activate
```

**Mac/Linux:**
```
source venv/bin/activate
```

> You'll know it worked when you see `(venv)` at the start of your terminal line.

---

### Step 3 — Install the required packages

```
pip install -r requirements.txt
```

> **What this does:** Reads `requirements.txt` and downloads all the Python libraries
> this tool needs (the AI SDK, PDF reader, web interface, etc.).
> This may take 1–2 minutes on first run.

---

### Step 4 — Set up your AI provider

Copy the example environment file:

```
copy .env.example .env
```

Open the new `.env` file in Notepad:

```
notepad .env
```

Fill in **one** of the keys depending on which provider you want to use, then save and close:

```
# Option A — MiniMax M3 (paid, ~$0.03/audit)
MINIMAX_API_KEY=your_key_here

# Option B — Groq (free tier)
GROQ_API_KEY=your_key_here

# Option C — Google Gemini (free tier)
GOOGLE_API_KEY=your_key_here

# Option D — Ollama (free, local) — no key needed, skip this step entirely
#   Just install Ollama and run: ollama pull qwen2.5:14b
```

> **What is an API key?**
> An API key is like a unique password that tells the AI service "this request
> comes from you". Keep it private — don't share it or commit it to git.

> **Recommendation for beginners:** Start with **Groq** (free, fast, no credit card needed).
> Sign up at https://console.groq.com/ and copy your API key into `GROQ_API_KEY`.

---

### Step 5 — Run the app

```
streamlit run app.py
```

> **What this does:** Starts the web app on your computer. Streamlit will open a browser
> window automatically (usually at http://localhost:8501).

You should see the **AI Compliance Checker** dashboard in your browser.

---

## How to Use the Tool

### Tab 1: Manage Rules

This is where you define *what* compliance means for your documents.

1. Click the **Manage Rules** tab
2. Type a **profile name** (e.g. `digipen_module_profile`)
3. Write your rules — one rule per line, in plain English:
   ```
   The document must include a module code and title.
   The document must list at least two learning outcomes.
   The document must state an attendance policy.
   ```
4. Click **Save Profile**

> A profile already exists for DigiPen module profiles — it will appear in the dropdown
> when you open the app.

---

### Tab 2: Run Audit

1. Select your compliance profile from the dropdown
2. Upload one or more PDF or DOCX documents
3. Select your AI provider from the **sidebar dropdown**
4. Click **Run Audit**

The AI will check every rule against every document. This takes ~5–10 seconds per rule with cloud providers, or ~15–30 seconds with Ollama depending on your hardware.

---

### Tab 3: Results

- **Summary matrix** — how many rules passed/failed per document at a glance
- **Detailed findings** — expand any document to see:
  - ✅ Pass — rule is satisfied
  - ❌ Fail — rule is not satisfied
  - ⚠️ Uncertain — AI wasn't sure (needs human review)
  - 📌 Evidence — the exact quote from the document
  - 💬 Reason — one-sentence explanation
- **Export CSV** — download all results as a spreadsheet

---

## Understanding What the AI Does

When you run an audit, the tool sends this to MiniMax M3 for every rule:

```
REQUIREMENT: The document must include an attendance policy.

DOCUMENT: [contents of your uploaded file]

Does this document satisfy the requirement? Reply with pass/fail/uncertain,
a confidence score, the relevant quote, and a one-sentence reason.
```

MiniMax M3 reads the document and responds with structured JSON that the tool
displays as the colour-coded results.

**Important:** The AI can make mistakes. Always have a human review:
- Any "uncertain" result
- Any "fail" result before acting on it
- High-stakes compliance decisions

---

## Folder Structure

```
compliance-checker/
│
├── app.py                  ← Main application (run this)
├── requirements.txt        ← Python packages needed
├── .env                    ← Your API key (create from .env.example)
├── .env.example            ← Template for .env
│
├── core/
│   ├── document_parser.py  ← Extracts text from PDF/DOCX files
│   ├── compliance_engine.py← Calls MiniMax M3 API for each rule
│   └── rule_profiles.py    ← Saves/loads rule profiles
│
├── profiles/
│   └── digipen_module_profile.json  ← Pre-built rules for DigiPen syllabi
│
└── results/                ← (Optional) save your CSV exports here
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` again |
| `(venv)` not showing | Run `venv\Scripts\activate` again |
| Browser doesn't open | Go to http://localhost:8501 manually |
| "API key invalid" error | Check your `.env` file — no extra spaces around the `=` sign |
| Ollama "connection refused" | Make sure Ollama is running — open a terminal and type `ollama serve` |
| Ollama model not found | Run `ollama pull qwen2.5:14b` in a terminal first |
| PDF shows no text | The PDF may be scanned; install PyMuPDF: `pip install PyMuPDF` |
| Slow audit | Normal — each rule is one AI call. 15 rules × 3 docs = 45 AI calls |

---

## Adding More Rules or Profiles

You can create as many profiles as you like — one for DigiPen syllabi, one for
HR policies, one for grant applications. Each is just a JSON file in the `profiles/`
folder.

You can also edit profile JSON files directly in Notepad if you prefer.

---

## Costs

MiniMax M3 charges per token (units of text). A typical 13-page syllabus audit
with 15 rules costs roughly **$0.01–$0.05 USD**. Check current pricing at
https://www.minimaxi.com/pricing.
