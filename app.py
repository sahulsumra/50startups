# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 11:11:21 2022

@author: admin
"""

import pickle
import numpy as np
import sklearn
import streamlit as st

final_model = pickle.load(open(r'G:\Startups\startup.pkl','rb'))


def main():
    
    st.title('Startup')
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Profit prediction App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    RD = st.text_input("R&D")
    
    Admin = st.text_input("Admin")
    
    MS = st.text_input("Marketing Spend")
    
    Florida = st.selectbox("State", ['New York', 'California', 'Florida'])
    if (Florida == 'Florida'):
        Florida = 1
        NewYork= 0
        
    elif(Florida == 'NewYork'):
        Florida = 0
        NewYork= 1
    
    else:
        Florida = 0
        NewYork= 0
        
    
    
  
    
   
    result=""
    if st.button("Predict"):
        result= final_model.predict([[RD, Admin, MS, Florida, NewYork]])
        output =round(result[0])
        st.success('The Profit is {}'.format(output))
   
   



if __name__=="__main__":
    main()