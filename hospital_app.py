import streamlit as st
import pandas as pd
import pickle

with open("hospital_model_bryan.pkl","rb") as f:
    bundle = pickle.load(f)
    st.write("Connected")

model = bundle["model"]
scaler = bundle["scaler"]

features = bundle["features"]
cols_to_scale = bundle["cols_to_scale"]

dept_map_inv = bundle["dept_map_inv"]

gender_map = bundle["gender_map"]
temp_map = bundle["temp_map"]
hr_map = bundle["hr_map"]
dur_map = bundle["dur_map"]
cc_map = bundle["cc_map"]

DEPT_INFO = {
  "Respiratory Medicine": {
    "icon" : "🫁",
    "desc" : "Specializes is conditions affecting the lungs and airways",
    "next" : [
      "visit level 2, Wing B", 
      "Estimated wait: 15-25 minutes", 
      "Please wear mask"
    ]
  },
   "Cardiology Medicine": {
    "icon" : "❤️",
    "desc" : "Specializes in heart and cardiovascular conditions",
    "next" : [
      "visit level 3, Wing A", 
      "Estimated wait: 20-30 minutes", 
      "Please bring previous ECG reports"
    ]
  },
 "Gastroenterology Medicine": {
    "icon" : "🫄",
    "desc" : "Specializes in digestive system conditions",
    "next" : [
      "visit level 1, Wing C", 
      "Estimated wait: 20-30 minutes"
    ]
  },
"Neurology Medicine": {
    "icon" : "🧠",
    "desc" : "Specializes in brain and nervous system conditions",
    "next" : [
      "visit level 1, Wing C", 
      "Estimated wait: 30 minutes - 60 minutes",
      "Bring current medication list"
    ]
  },
"General Medicine": {
    "icon" : "🩺",
    "desc" : "General Health Consultation",
    "next" : [
      "visit level 1, Wing A"
    ]
  },
"Dermatology Medicine": {
    "icon" : "👩",
    "desc" : "Specializes in skin conditions",
    "next" : [
      "visit level 2, Wing D"
    ]
  }
}

st.title("🏥 Smart Hospital Patient Navigator Bryan")

st.write("Fill in the patient's information below")

st.header("Patient Information")

age = st.number_input(
  "Age",
  min_value=1,
  max_value=120,
  value=30
)
gender = st.selectbox(
  "Gender", 
  ["Female", "Male"]
)

st.header("Symptoms")

col1, col2 = st.columns(2)

with col1: 
  fever = st.checkbox("Fever 🥵")
  cough = st.checkbox("Cough 😷")
  headache = st.checkbox("Headache 💆‍♂️")
  chest_pain = st.checkbox("Chest Pain 💔")
  stomach_pain = st.checkbox("Stomach Pain 🤢")

with col2: 
  shortness_breath = st.checkbox("Shortness of Breath 😮‍💨")
  nausea_vomiting = st.checkbox("Nausea/Vomiting 🤮")
  dizziness = st.checkbox("Dizziness 😵‍💫")
  skin_rash = st.checkbox("Skin Rash 🔴")

st.header("Patient Condition")
temperature_level = st.selectbox(
  "Temperature", 
  options = list(temp_map.keys())
)

heart_rate_level = st.selectbox(
  "Heart Rate", 
  options = list(hr_map.keys())
)

duration = st.selectbox(
  "Duration of Symptoms", 
  options = list(dur_map.keys())
)

chief_complaint = st.selectbox(
  "Chief Complaint", 
  options = list(cc_map.keys())
)

st.header("Medical History")
hypertension = st.checkbox("Hypertension")
heart_disease = st.checkbox("Heart Disease")
asthma = st.checkbox("Asthma")

predict_button = st.button("Predict Department")

if predict_button:
    patient = pd.DataFrame ([{
        "age" : age, 
        "gender" : gender_map[gender], 
        "fever" : int(fever), 
        "cough" : int(cough), 
        "headache" : int(headache), 
        "chest_pain" : int(chest_pain), 
        "stomach_pain" : int(stomach_pain),
        "shortness_breath" : int(shortness_breath),
        "nausea_vomiting" : int(nausea_vomiting),
        "dizziness" : int(dizziness),
        "skin_rash" : int(skin_rash),

        "temperature_level": temp_map[temperature_level], 
        "heart_rate_level": hr_map[heart_rate_level],
        "duration": dur_map[duration],

        "asthma" : int(asthma),
        "hypertension" : int(hypertension),
        "heart_disease" : int(heart_disease),

        "chief_complaint" : cc_map[chief_complaint]
    }])

    patient_scaled = patient.copy()

    patient_scaled[cols_to_scale] = scaler.transform(
        patient[cols_to_scale]
    )

    prediction = model.predict(
        patient_scaled[features]
    )[0]

    probability = model.predict_proba(
        patient_scaled[features]
    )[0]

    # Fixed: Changed dept_map_inv -> depth_map_inv and departments -> department
    department = dept_map_inv[prediction]

    confidence = probability[prediction] * 100

    # Fixed: Brought entire output display block INSIDE the button block
    st.divider()
    st.header("Prediction Result")
    info = DEPT_INFO.get(department)

    if info:
        # Fixed: Changed Department -> department and adjusted quote layout inside f-string
        st.success(f"{info['icon']} Recommended Department: {department}")
        st.write(f"**Confidence:** {confidence:.1f}%")
        st.write("Description: ")
        st.write(info["desc"])
        st.write("What should the patient do?")

        for step in info["next"]: 
            st.write(f"- {step}")
    else:
        st.success(f"Recommended Department: {department}")
        st.write(f"Confidence : {confidence:.1f}%")

    st.warning("This A.I. Recommendation is only for educational purposes")
