import streamlit as st  # Importing the streamlit library for creating web apps
import joblib  # Importing joblib for loading the pre-trained model
import pandas as pd  # Importing pandas for data manipulation
from sklearn import linear_model  # Importing linear_model from sklearn (though not used here)

# Loading the pre-trained fraud detection model from a file
loaded_model = joblib.load('fraud_detection_model.pkl')

# Setting the title of the Streamlit app
st.title('Fraud probability prediction')

# Creating a text area for user input with instructions on how to format the input
user_input = st.text_area('''
device_type_map = {'browser': 1, 'mobile': 2, "machine": 3} = device used for the transaction\n
connection_type_map = {'Wifi': 1, 'mobile_data': 2} - type of connection used in transaction\n
gps_status_map = {'ON': 1, 'OFF': 0} - gps is turned on or not\n
know_gps_map = {False: 0, True: 1} - gps is an habitual location for the account owner\n
know_recipient_map = {False: 0, True: 1} - money receiver is know for the account owner\n
value = 110.15 - value of the transaction \n  
input example: 1, 1, 1, 1, 1, 150  
ctrl+Enter to submit 
''', '')

# Checking if the user has provided input
if user_input:
    # Splitting the input string by commas, stripping whitespace, and converting to float
    user_list = [float(item.strip()) for item in user_input.split(',')]
    
    # Predicting the probability of fraud using the loaded model
    predict = loaded_model.predict(pd.DataFrame([user_list], columns=["device_type", "connection_type", "gps_status", "know_gps", "know_recipient","Value"]))
    
    # Displaying the prediction result in the Streamlit app
    st.write("Probability to be a fraud: {:.2f}%".format(predict[0]*100))
