import streamlit as st
import os
import requests

API_URL = os.getenv(API_URL = "http://65.2.129.23:8000")

st.set_page_config(page_title="Fraud Detection", page_icon="💳")

st.title("💳 Fraud Detection System")
st.markdown("Enter transaction details to check if it's fraudulent.")

# Inputs
col1, col2 = st.columns(2)

with col1:
    time = st.number_input("⏱ Time", min_value=0.0, value=10000.0)

with col2:
    amount = st.number_input("💰 Amount", min_value=0.0, value=100.0)

input_data = {
    "Time": time,
    "Amount": amount
}

# Predict
if st.button("🔍 Predict", use_container_width=True):
    st.divider()

    try:
        with st.spinner("Analyzing transaction..."):
            response = requests.post(
                f"{API_URL}/predict",
                json=input_data,
                timeout=10
            )
            response.raise_for_status()

        result = response.json()

        prob = result.get("fraud_probability", 0)
        pred = result.get("fraud_prediction", False)

        st.subheader("📊 Result")
        st.metric("Fraud Probability", f"{prob:.4f}")

        # Risk interpretation
        if prob < 0.1:
            st.info("🟢 Low risk transaction")
        elif prob < 0.3:
            st.warning("🟡 Medium risk transaction")
        else:
            st.error("🔴 High risk transaction")

        if pred:
            st.error("🚨 Fraud Detected")
        else:
            st.success("✅ Legit Transaction")

    except requests.exceptions.ConnectionError:
        st.error(f"❌ Cannot connect to API at {API_URL}")
    except requests.exceptions.Timeout:
        st.error("❌ API request timed out")
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ API error: {e.response.status_code}")
    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")