import json
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from groq import Groq

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "deals.json"
load_dotenv(ROOT / ".env")


def setting(name, default=""):
    value = os.getenv(name)
    if value:
        return value
    secret_files = [
        Path.home() / ".streamlit" / "secrets.toml",
        ROOT / ".streamlit" / "secrets.toml",
    ]
    if not any(path.exists() for path in secret_files):
        return default
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


class DealStore:
    def __init__(self, path):
        self.path = Path(path)

    def load(self):
        with self.path.open(encoding="utf-8") as file:
            return json.load(file)

    def save(self, deals):
        self.path.write_text(json.dumps(deals, indent=2), encoding="utf-8")

    def estimate_with_groq(self, deals):
        api_key = setting("GROQ_API_KEY")
        if not api_key:
            return self.run_local_estimates(deals)

        model = setting("GROQ_MODEL", "llama-3.3-70b-versatile")
        client = Groq(api_key=api_key)
        descriptions = [item["deal"]["product_description"] for item in deals]
        prompt = (
            "Estimate the fair market value in USD for each product. "
            "Return only a JSON object with an 'estimates' array of numbers "
            "in the same order as the products. Products:\n"
            + json.dumps(descriptions)
        )
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "You are a careful product price estimation assistant.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        estimates = json.loads(response.choices[0].message.content)["estimates"]
        if len(estimates) != len(deals):
            raise ValueError("Groq returned the wrong number of estimates")
        for deal, estimate in zip(deals, estimates):
            deal["estimate"] = round(float(estimate), 2)
            deal["discount"] = round(deal["estimate"] - deal["deal"]["price"], 2)
        return deals

    def run_local_estimates(self, deals):
        for deal in deals:
            deal["estimate"] = estimate_value(deal["deal"]["product_description"])
            deal["discount"] = round(deal["estimate"] - deal["deal"]["price"], 2)
        deals.sort(key=lambda item: item["discount"], reverse=True)
        return deals


def estimate_value(description):
    text = description.lower()
    value = 300.0
    if "iphone" in text:
        value = 930.0
    elif "watch" in text:
        value = 775.0
    elif "laptop" in text or "computer" in text:
        value = 850.0
    elif "headphone" in text or "microphone" in text:
        value = 280.0
    return value


store = DealStore(DATA_FILE)


def money(value):
    return f"${value:,.2f}"


st.set_page_config(page_title="The Price Is Right", page_icon="$", layout="wide")
st.title("The Price Is Right")
st.caption("A Groq-powered deal intelligence dashboard based on Multimodal agent workflow.")

if "deals" not in st.session_state:
    st.session_state.deals = store.load()

if st.button("Run deal scan", type="primary"):
    with st.spinner("Estimating true value with Groq..."):
        try:
            st.session_state.deals = store.estimate_with_groq(st.session_state.deals)
            store.save(st.session_state.deals)
            st.success("Scan complete. Opportunities are ranked by discount.")
        except Exception as error:
            st.error(f"Groq estimation failed: {error}")

deals = sorted(st.session_state.deals, key=lambda item: item.get("discount", 0), reverse=True)
discounts = [float(item.get("discount", 0)) for item in deals]
summary = st.columns(3)
summary[0].metric("Opportunities", len(deals))
summary[1].metric("Best discount", money(max(discounts, default=0)))
summary[2].metric(
    "Average discount", money(sum(discounts) / len(discounts) if discounts else 0)
)

st.subheader("Saved opportunities")
for item in deals:
    product = item["deal"]
    with st.container():
        left, right = st.columns([4, 1])
        with left:
            st.markdown(f"**{product['product_description']}**")
            st.caption(f"Deal price: {money(product['price'])} | Estimated value: {money(item['estimate'])}")
            st.link_button("View source", product["url"])
        with right:
            st.metric("Discount", money(item["discount"]))
