import streamlit as st
import pickle
import numpy as np
 
#load model
with open ('iris_model.pkl', 'rb') as f:
  model = pickle.load(f)
  
st.title("Iris Flower classifier")
st.write("Enter the fectures below:")

sepal_length = st.slider("Sepal length (cm)", 4.0,8.0,5.8)
sepal_width = st.slider("Sepal width (cm)", 2.0,4.4,3.0)
petal_length = st.slider("Petal length (cm)", 1.0,7.0,4.0)
petal_width = st.slider("Petal width (cm)" , 0.1,2.5,1.2)

features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
if st.button("predict")
prediction = model.predict(features)
class_name = ['Sentos','Versicolor','virginica']
st.success("The predicted Iris species is: {class_name[prediction[0]]}")



