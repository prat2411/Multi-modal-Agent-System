# Product Price Estimator

I built this project to estimate the price of a product from its title and description. It uses a Groq-hosted language model, with a small Flask backend and a Streamlit interface.

## Run it locally

1. Install Python 3.9 or newer.
2. Install the packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Add your Groq key:

   ```powershell
   $env:GROQ_API_KEY = "your-groq-api-key"
   ```

4. Start the Streamlit app:

   ```bash
   streamlit run app.py
   ```

Enter a product title and description to get an estimate.

## Flask API

Start the backend with:

```bash
python backend.py
```

The API runs at `http://localhost:5000`.

```bash
curl http://localhost:5000/health
```

```bash
curl -X POST http://localhost:5000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"title\":\"Used iPhone 15\",\"description\":\"Excellent condition, 256GB\"}"
```

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Create an app at [share.streamlit.io](https://share.streamlit.io).
3. Select this repository and set the main file to `app.py`.
4. Add `GROQ_API_KEY` in the app's Secrets settings.

Streamlit runs the interface. The Flask API can be run locally or deployed separately.

## Project files

```text
app.py                  Streamlit interface
backend.py              Flask API and prediction logic
pricer/evaluator.py     Groq model wrapper
train.py                Optional LoRA/QLoRA training
prepare_data.py         Optional dataset preparation
requirements.txt        Python packages
```

Training is optional. The deployed app uses Groq for inference, so it does not need a GPU.