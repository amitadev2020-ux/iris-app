import streamlit as st
import numpy as np
import pickle

# Load the trained model
with open("iris_model.pkl", "rb") as f:
    model = pickle.load(f)

# Title
st.title("🌸 Iris Flower Classifier")
st.write("Enter the flower measurements below:")

# Input sliders
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.4, 3.0)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.2)

# Create feature array
features = np.array([[sepal_length,
                      sepal_width,
                      petal_length,
                      petal_width]])

# Predict button
if st.button("Predict"):
    prediction = model.predict(features)

    class_names = [
        "Setosa",
        "Versicolor",
        "Virginica"
    ]

    st.success(f"The predicted Iris species is: **{class_names[prediction[0]]}**")

