import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import pickle

# Create h1 header
st.title("This is my flower predictor.")
st.header("this is a great app")
st.subheader("Flowers are my fav.")

df = px.data.iris()

show_df = st.checkbox("Do you want to see the data?")

if show_df:
    df


# Get user input
s_l = st.number_input("Sepal Length (cm)", 0, 100)
s_w = st.number_input("Sepal Width (cm)", 0, 100)
p_l = st.number_input("Petal Lenght (cm)", 0, 100) 
p_w = st.number_input("Petal Width (cm)", 0, 100)

user_input = np.array([s_l, s_w, p_l, p_w]).reshape(1, -1)


# Import Model
with open("saved-iris-model.pkl", "rb") as flower_pickle:
    model = pickle.load(flower_pickle)


# Predict user input 
prediction = model.predict(user_input)
st.write(f"The predicted flower is {prediction[0]}")
