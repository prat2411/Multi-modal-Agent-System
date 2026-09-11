"""Streamlit app for Groq-powered price estimation."""

from __future__ import annotations

import streamlit as st

from backend import estimate_price


st.set_page_config(page_title="Price Estimator", page_icon="$", layout="centered")
st.title("Product Price Estimator")
st.write("Enter a product and get an estimated price from a Groq-powered model.")

title = st.text_input("Product title", placeholder="Vintage Seiko watch")
description = st.text_area(
    "Description",
    placeholder="Automatic movement, stainless steel case, lightly used.",
)

if st.button("Estimate price", type="primary"):
    if not title.strip():
        st.warning("Please enter a product title.")
    else:
        try:
            result = estimate_price(title, description)
            st.metric("Estimated price", result["price"])
            st.caption(result["response"])
        except Exception as error:
            st.error(str(error))
