# -*- coding: utf-8 -*-
"""
Created on Mon Oct  17 21:01:15 2022
@author: Akash
"""

import pickle
import streamlit as st
from streamlit_option_menu import option_menu

# loading the saved models

calories_model = pickle.load(open('calories_model.sav', 'rb'))

# sidebar for navigation
with st.sidebar:
    selected = option_menu('Calories Burnt Prediction Model',

                           ['Calories Burnt Model'],
                           icons=['activity'],
                           default_index=0)

# Diabetes Prediction Page
if (selected == 'Calories Burnt Model'):

    # page title
    st.title('Calories Burnt Prediction Model')

    # getting the input data from the user
    col1, col2, col3 = st.columns(3)

    with col1:
        Gender = st.selectbox('Gender', ('Male', 'Female'))

        if (Gender == 'Male'):
            Gender = 0
        else:
            Gender = 1

    with col2:
        Heart_Rate = st.number_input('HeartRate (BPM)')

    with col3:
        Duration = st.number_input('Duration (Minutes)')

    # code for Prediction
    calories_predicted = ''

    # creating a button for Prediction

    if st.button('Predicted Calories Burnt'):
        calories_predicted = "You will burn around {} Calories.".format(
            round(calories_model.predict([[Gender, Duration, Heart_Rate]]), 2))

    st.success(calories_predicted)



