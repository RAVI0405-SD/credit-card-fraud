import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

st.markdown("""
    <style>
        .result-card {
            padding: 20px; 
            border-radius: 10px; 
            background: white;
            box-shadow: 0 3px 10px rgba(0,0,0,0.1);
            margin-bottom: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# ----------------------------
# TITLE SECTION
# ----------------------------
st.title("💳 Credit Card Fraud Detection System")
st.write("Predict fraudulent transactions using a trained machine-learning model.")

# ----------------------------
# SIDEBAR INPUT MODE
# ----------------------------
st.sidebar.header("⚙️ Input Options")
input_mode = st.sidebar.radio(
    "Select Input Type",
    ["Manual Input", "Upload CSV File"]
)

# ----------------------------
# LOAD MODEL
# ----------------------------
model = joblib.load("savedModels/model.joblib")

# ----------------------------
# FUNCTION: MANUAL USER INPUT
# ----------------------------
def manual_input():
    st.sidebar.subheader("Transaction Features")

    sliders = {}
    for v in range(1, 29):
        sliders[f"V{v}"] = st.sidebar.slider(
            f"V{v}",
            -5.0, 5.0, 0.0
        )

    amount = st.sidebar.number_input(
        "Amount",
        min_value=0.0,
        max_value=10000.0,
        value=100.0
    )

    data = {**sliders, "Amount": amount}
    return pd.DataFrame(data, index=[0])

# ----------------------------
# GET INPUT DATA
# ----------------------------
if input_mode == "Upload CSV File":
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        st.success("CSV uploaded successfully!")
    else:
        st.warning("Waiting for CSV upload...")
        input_df = None
else:  # Manual Input
    input_df = manual_input()

# ----------------------------
# SHOW DATA
# ----------------------------
st.subheader("📄 Input Data Preview")
if input_df is not None:
    st.dataframe(input_df, use_container_width=True)
else:
    st.info("Please upload a CSV file or enter values manually to continue.")

# ----------------------------
# PREDICT & DISPLAY RESULTS
# ----------------------------
if input_df is not None:
    prediction = model.predict(input_df)

    # ---------- MANUAL MODE: single transaction UI ----------
    if input_mode == "Manual Input":
        pred = int(prediction[0])
        prob = float(model.predict_proba(input_df)[0][1])  # probability of fraud (0–1)
        prob_percent = prob * 100

        st.subheader("🧾 Prediction Result")

        if pred == 1:
            box_color = "#FF4C4C"
            label = "⚠️ Fraudulent Transaction Detected"
        else:
            box_color = "#4CAF50"
            label = "✅ Genuine Transaction"

        # Result card
        st.markdown(
            f"""
            <div class="result-card" style="border-left: 10px solid {box_color}">
                <h3 style="color:{box_color}; margin: 0;">{label}</h3>
                <p style="margin-top:10px;"><b>Fraud Probability:</b> {prob_percent:.2f}%</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Risk bar (how close it is to fraudulent)
        st.markdown("#### 🔍 How close is it to being fraudulent?")
        st.write(f"Risk level: **{prob_percent:.2f}%**")
        st.progress(int(prob_percent))   # 0–100 bar

    # ---------- CSV MODE: summary stats UI ----------
    else:
        fraud_count = int(np.sum(prediction == 1))
        genuine_count = int(np.sum(prediction == 0))
        total = len(prediction)

        col1, col2 = st.columns(2)

        # Result message
        with col1:
            st.subheader("🧾 Prediction Result")
            if fraud_count > 0:
                st.error(f"Fraudulent transactions detected: {fraud_count}")
            else:
                st.success("All transactions are genuine")

        # Summary stats
        with col2:
            st.subheader("📊 Summary Statistics")
            st.metric("Total Transactions", total)
            st.metric("Genuine Transactions", genuine_count)
            st.metric("Fraudulent Transactions", fraud_count)


