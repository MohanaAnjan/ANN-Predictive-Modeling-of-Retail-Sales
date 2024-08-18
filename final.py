
import pandas as pd 
import matplotlib.pyplot as plt
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import tensorflow as tf 
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import load_model
import numpy as np


df=pd.read_csv('D:/My Projects/fianl  project Doc/final_project_csv')
df=df.iloc[:,1:]



# -------------------------------This is the configuration page for our Streamlit Application---------------------------
st.set_page_config(
    page_title="Retail Sales Prediction ",
    layout="wide"
)

# -------------------------------This is the sidebar in a Streamlit application, helps in navigation--------------------
with st.sidebar:
    selected = option_menu("Main Menu", ["About Project", "Predictions"],
                           icons=["house", "gear"],
                           styles={"nav-link": {"font": "sans serif", "font-size": "20px", "text-align": "centre"},
                                   "nav-link-selected": {"font": "sans serif", "background-color": "#0072b1"},
                                   "icon": {"font-size": "20px"}
                                   }
                           )


# -----------------------------------------------About Project Section--------------------------------------------------
if selected == "About Project":
    st.markdown("# :blue[ANN Predictive Modeling of Retail Sales]")
    st.markdown('<div style="height: 50px;"></div>', unsafe_allow_html=True)
    st.markdown("### :blue[Technologies :] Python, Pandas, Numpy, Scikit-Learn,Tensorfllow, Streamlit, Python scripting, "
                "Deep Learning, Data Preprocessing, Visualization, EDA")
    st.markdown("### :blue[Overview :] Develop a predictive ANN model to forecast department-wide sales for each store over" 
                "the next year and analyze the impact of markdowns on sales during holiday weeks."
                "Provide actionable insights and recommendations to optimize markdown strategies and inventory management..")
    st.markdown("### :blue[Domain :] Marketing and Accountancy")
    st.markdown("### :blue[BY :] Mohana Anjan A V ")

if selected == "Predictions":
    st.markdown("# :blue[Predicting Results based on Trained Models]")
    with st.form("form1"):
        col1,col2 = st.columns([9,9]) 
    
        # -----New Data inputs from the user for predicting the resale price-----


    
        with col1:
            store= st.selectbox('Select Store Number',sorted(df.Store.unique()))
            holiday = st.selectbox ('Select IsHoliday ', ['Yes','No'])
            day= st.selectbox('Select Day',sorted(df.day.unique()))
            month= st.selectbox('Select Month',sorted(df.month.unique()))
            year=st.number_input('Enter the Year', min_value=1, max_value=2024)
            dept= st.selectbox('Select Department',sorted(df.Dept.unique()))
            type= st.selectbox('Select type',['A','B','C'])
            size= st.selectbox('Select size',sorted(df.Size.unique()))
            temperature = st.number_input('Enter the Temperature', min_value=1.00, max_value=100.00)
            
            
            
        with col2:

            fuel= st.number_input('Enter the Fuel Price', min_value=1.00, max_value=100.0)
            MarkDown1 = st.number_input('Enter the MarkDown1 price (First Discount)', min_value=0.00, max_value=1000000.0)
            MarkDown2 = st.number_input('Enter the MarkDown2 price (Second Discount)', min_value=0.00, max_value=1000000.0)
            MarkDown3 = st.number_input('Enter the MarkDown3 price (First Discount)', min_value=0.00, max_value=1000000.0)
            MarkDown4 = st.number_input('Enter the MarkDown4 price (Fourth Discount)', min_value=0.00, max_value=1000000.0)
            MarkDown5 = st.number_input('Enter the MarkDown5 price (Fifth Discount)', min_value=0.00, max_value=1000000.0)
            cpi = st.number_input('Enter the CPI', min_value=1.000000, max_value=1000.000000)
            unemp = st.number_input('Enter the Unmemployment', min_value=1.000, max_value=50.000)
            
            
            



        # -----Submit Button for PREDICT RESALE PRICE-----
            submit_button = st.form_submit_button(label="PREDICT RESALE PRICE")
            if submit_button is not None:

                holiday=1 if holiday=='Yes' else 0
                type= 1 if type == 'A' else 2 if type == 'B' else 3



                custom_objects = {'LeakyReLU': LeakyReLU}

            # Load the model with custom objects
                loaded_model = load_model('final_model.h5', custom_objects=custom_objects)

                with open(r'scaler.pkl', 'rb') as f:
                    scaler_loaded = pickle.load(f)
               
                
                
                    # -----Sending that data to the trained models for    price prediction----------------
                new_sample=np.array([[store,temperature,fuel,MarkDown1,MarkDown2,MarkDown3,MarkDown4,MarkDown5,
                                      cpi,unemp,holiday,dept,type,size,year,month,day
                     
                     ]])

                new_sample=scaler_loaded.transform(new_sample)
                

            # Reshape to 2D array
                new_sample5 = scaler_loaded.transform(new_sample)
                new_pred = loaded_model.predict(new_sample5)[0]
                st.write('## :green[Predicted weekly  price:] ', new_pred)

                    
