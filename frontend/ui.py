import streamlit as st
import requests

API_URL = "http://localhost:8000/predict"


st.title("Churn Predictor")


with st.form(key="churn-form"):
    st.write("Enter Below Details: ")

    Tenure = st.number_input("Tenure", min_value=0, step=1)
    Usage_Frequency = st.number_input("Usage Frequency", min_value=0, step=1)
    Support_Calls = st.number_input("Support Calls", min_value=0, step=1)
    Payment_Delay = st.number_input("Payment Delay", min_value=0, step=1)
    Subscription_Type = st.selectbox(
        "Subscription Type", ("Basic", "Standard", "Premium"), index=None
    )
    Contract_Length = st.selectbox(
        "Contract Length", ("Monthly", "Annual", "Quarterly"), index=None
    )
    Total_Spend = st.number_input("Total Spend", min_value=0)
    Last_Interaction = st.number_input("Last Interaction", min_value=0, step=1)

    submit = st.form_submit_button("submit")


if submit:
    if Subscription_Type is None or Contract_Length is None:
        st.error("Select a type")
    else:
        try:
            with st.spinner(
                "Analyzing Churn Probability....",
            ):
                response = requests.post(
                    API_URL,
                    json={
                        "Tenure": Tenure,
                        "Usage Frequency": Usage_Frequency,
                        "Support Calls": Support_Calls,
                        "Payment Delay": Payment_Delay,
                        "Subscription Type": Subscription_Type,
                        "Contract Length": Contract_Length,
                        "Total Spend": Total_Spend,
                        "Last Interaction": Last_Interaction,
                    },
                    timeout=10,
                )

                prediction = response.json()["ChurnPrediction"]

                st.write(prediction)

                if prediction == 1:
                    st.write("High Churn Probability")
                else:
                    st.write("Low Churn Probability")

        except requests.RequestException as error:
            st.error(f"API request error: {error}")

else:
    st.error("Form error")
