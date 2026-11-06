import streamlit as st
import joblib
import pandas as pd
import os

# Page configuration
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered"
)

# Application title
st.title("🚢 Titanic Survival Predictor")

st.write(
    "Enter passenger details below to predict "
    "the estimated chance of survival."
)
# Load the trained machine learning model

  
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_model():
    model_path = os.path.join(BASE_DIR, "model.joblib")
    try:
        return joblib.load(model_path)
    except FileNotFoundError:
        st.error("Model file is missing. Please check the project files.")
        st.stop()
    except Exception:
        st.error("Unable to load the model. Please try again later.")
        st.stop()

model = load_model()
# Passenger details
st.subheader("Enter Passenger Details")

pclass = st.selectbox(
    "Ticket Class",
    [1, 2, 3]
)

sex = st.radio(
    "Sex",
    ["Female", "Male"]
)

age = st.slider(
    "Age",
    1, 80, 30
)

fare = st.slider(
    "Ticket Fare",
    0, 500, 30
)
# Prediction button
if st.button("Predict Survival"):

    # Convert passenger details into model input
    sex_value = 1 if sex == "Female" else 0

    input_data = pd.DataFrame(
        [[pclass, sex_value, age, fare]],
        columns=["pclass", "sex", "age", "fare"]
    )

    # Make prediction
    prediction = model.predict(input_data)

    # Calculate survival probability
    probability = model.predict_proba(input_data)[0][1]

    # Display result
    st.metric(
        "Estimated Chance of Survival",
        f"{probability:.0%}"
    )

    if prediction[0] == 1:
        st.success("The model predicts that the passenger may survive!")
    else:
        st.error("The model predicts that the passenger may not survive.")
# Educational disclaimer
st.divider()

st.caption(
    "This machine learning model was trained using "
    "historical Titanic passenger data. "
    "It is an educational project and its predictions "
    "are not guaranteed to be accurate."
)
