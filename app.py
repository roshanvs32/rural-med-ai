import streamlit as st
import pickle
import numpy as np

# Load trained model
model = pickle.load(open("model/disease_model.pkl", "rb"))

st.title("Rural Health AI Assistant")
st.write("Enter patient symptoms below:")

# Symptom checkboxes
fever = st.checkbox("Fever")
cough = st.checkbox("Cough")
rash = st.checkbox("Rash")
joint_pain = st.checkbox("Joint Pain")
vomiting = st.checkbox("Vomiting")
diarrhea = st.checkbox("Diarrhea")
headache = st.checkbox("Headache")
breathing = st.checkbox("Breathing Difficulty")

# Prediction button
if st.button("Analyze Patient"):

    symptoms = np.array([[
        int(fever),
        int(cough),
        int(rash),
        int(joint_pain),
        int(vomiting),
        int(diarrhea),
        int(headache),
        int(breathing)
    ]])

    # Predict disease
    prediction = model.predict(symptoms)

    # Predict probabilities
    probabilities = model.predict_proba(symptoms)

    # Show top predicted disease
    st.subheader("Primary Prediction:")
    st.success(prediction[0])

    # Sort diseases by probability
    disease_probs = list(zip(model.classes_, probabilities[0]))
    disease_probs = sorted(disease_probs, key=lambda x: x[1], reverse=True)

    # Show only diseases with non-zero probability
    st.subheader("Most Likely Diseases:")
    for disease, prob in disease_probs:
        if prob > 0:
            st.write(f"{disease}: {prob*100:.2f}%")

    # Urgency detection
    if breathing and fever:
        st.error("⚠ Emergency Detected: Possible severe respiratory infection. Refer patient to hospital immediately.")
    elif diarrhea and vomiting:
        st.warning("⚠ Moderate Risk: Possible dehydration. Monitor patient and consider referral.")
    else:
        st.success("Low Immediate Risk. Monitor symptoms and follow medical guidance.")

# Medical disclaimer
st.error("⚠ This tool is for educational purposes only. Always consult a qualified medical professional.")