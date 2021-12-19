#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
from supervised.automl import AutoML
import joblib


# In[2]:


automl = joblib.load('car_prediction_model.pkl')

def predict(input_df):
    predictions_df = automl.predict(input_df)
    predictions = predictions_df['Label'][0]
    return predictions

def run():

    from PIL import Image
    image = Image.open('MLJR.jpg')
    #image_hospital = Image.open('hospital.jpg')

    st.image(image,use_column_width=False)

    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?",
    ("Online", "Batch"))

    st.sidebar.info('This app is created to predict prices of Hyundai car models')
    st.sidebar.success('https://mljar.com')
    
    #st.sidebar.image(image_hospital)

    st.title("HYUDAI CAR PRICE PREDICTION")

    if add_selectbox == 'Online':

        model = st.selectbox('model', ['Santa Fe', 'Ioniq','I800','Tucson','Kona','Kona','I30','I40','IX20','I20','IX35','I10','Veloster',
                                      'Terracan','Getz','Amica','Accent'])
        year = st.number_input('year', min_value=1, max_value=2080, value=25)
        transmission = st.selectbox('transmission', ['Semi-Auto', 'Automatic','Manual','Other'])
        mileage = st.number_input('mileage', min_value=1, max_value=200000, value=25)
        fuelType = st.selectbox('fuelType', ['Hybrid', 'Diesel','Petrol','Other'])
        tax = st.number_input('tax', min_value=1, max_value=2000, value=25)
        mpg = st.number_input('mpg', min_value=0.0, max_value=200.0, value=25.0)
        engineSize = st.number_input('engineSize', min_value=0.0, max_value=10.0, value=5.0)
        #bmi = st.number_input('BMI', min_value=10, max_value=50, value=10)
        #children = st.selectbox('Children', [0,1,2,3,4,5,6,7,8,9,10])
        #smoker = st.selectbox('smoker', ['yes', 'no'])
        
        #region = st.selectbox('Region', ['southwest', 'northwest', 'northeast', 'southeast'])

        output=""

        input_dict = {'model' : model, 'year' : year, 'transmission' : transmission, 'mileage' : mileage, 'fuelType' : fuelType, 'tax' : tax,
                     'mpg' : mpg,'engineSize' : engineSize}
        
        input_df = pd.DataFrame([input_dict])
        
        

        if st.button("Predict"):
            output = automl.predict(input_df)
            
        
                                
            
            #output = output
            #output = output = str(output)

        #st.success('The output is {}'.format(output))
        st.success(output)

    if add_selectbox == 'Batch':

        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])

        if file_upload is not None:
            data = pd.read_csv(file_upload)
            predictions = automl.predict(data)
            st.write(predictions)

if __name__ == '__main__':
    run()


# In[ ]:





# In[ ]:




