# The Price Is Right

The Price Is Right is a Streamlit app that helps identify good product deals. It compares the sale price with an estimated fair value, calculates the possible discount, and ranks the best opportunities.

The app uses the Groq API for fast price estimates. It also works without an API key by using a simple local fallback estimator.

## Run Locally

From PowerShell, run:

```powershell
cd D:\dev\llm_engineering\week8\Project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Create `.env` from `.env.example`, then add your Groq key:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-8b-instant
```

Start the app:

```powershell
python -m streamlit run main.py
```

The app opens at `http://localhost:8501`.

Click **Run deal scan** to estimate the value of the saved products and rank them by discount.

Without a Groq key, the app uses the local fallback estimator.


## Files

```text
Project/
├── main.py                 # Streamlit app and Groq workflow
├── data/
│   └── deals.json          # Saved deals
├── requirements.txt        # Python packages
├── .env.example            # Environment variable template
└── README.md               # Instructions
```

## Important Configuration

`GROQ_API_KEY` enables Groq-powered estimates.

`GROQ_MODEL` is optional. The default is `llama-3.1-8b-instant`.

 The current app is the simple deployable version of that project.
