import streamlit as st
import pandas as pd
import pickle

with open("hospital_model_bryan.pkl","rb") as f:
  bundle = pickle.load(f)
  st.write("Connected")
