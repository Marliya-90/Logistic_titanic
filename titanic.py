import streamlit as st
import joblib

model = joblib.load("titanic_model.pkl")


import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np

st.title("🚢 Titanic Survival Prediction")

# Load dataset
df = pd.read_csv("Titanic-Dataset.csv")

# Select all 6 features + target
df = df[['Pclass','Sex','Age','SibSp','Parch','Fare','Survived']].dropna()

# Encode Sex
df['Sex'] = df['Sex'].map({'male':0, 'female':1})

X = df[['Pclass','Sex','Age','SibSp','Parch','Fare']]
y = df['Survived']

# Train model
model = LogisticRegression()
model.fit(X, y)

# Streamlit inputs
pclass = st.selectbox("Passenger Class", [1,2,3])
sex = st.selectbox("Sex", ["Male","Female"])
age = st.number_input("Age", 0, 100, 25)
sibsp = st.number_input("Siblings/Spouses aboard", 0, 10, 0)
parch = st.number_input("Parents/Children aboard", 0, 10, 0)
fare = st.number_input("Fare", 0.0, 600.0, 32.0)

# Convert Sex to numeric
sex_val = 0 if sex=="Male" else 1

# Create input array
input_data = np.array([[pclass, sex_val, age, sibsp, parch, fare]])

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.success("Passenger Survived ✅")
    else:
        st.error("Passenger Did NOT Survive ❌")
