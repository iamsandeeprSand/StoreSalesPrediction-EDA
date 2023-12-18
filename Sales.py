
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelBinarizer
import streamlit as st
import re
st.set_page_config(layout="wide")

st.write("""
<div style='text-align:center'>
    <h1 style='color:#009999;'>Sales Forecasting Application</h1>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["XGBoostRegressor","PowerBI Dashboard", "Tableau Dashboard", "Insights Recommended"])
with tab1:

#	Store	Dept	Temperature	Fuel_Price	MarkDown1	MarkDown2	MarkDown3	MarkDown4	MarkDown5	CPI	Unemployment	Size	month_date	day_date	year_date	Type_B	Type_C	IsHoliday_True
        # Define the possible values for the dropdown menus
        store = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45]
        department = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
        Type = ["A", "B", "C"]
        IsHoliday = [True, False]
        Day_Date = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
        Month_Date = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]



        # Define the widgets for user input
        with st.form("my_form"):
            col1,col2,col3=st.columns([5,2,5])
            with col1:
                st.write(' ')
                store = st.selectbox("store", store,key=1)
                department = st.selectbox("Department", department,key=2)
                Temperature = st.text_input("Enter Temperature in Celsius (Min : -2.0, Max : 46.5)")
                Fuel_Price = st.text_input("Enter Fuel Price (Min : 2.4, Max : 4.5)")
                MarkDown1 = st.text_input("Enter MarkDown1")
                MarkDown2 = st.text_input("Enter MarkDown2")
                MarkDown3 = st.text_input("Enter MarkDown3")
                MarkDown4 = st.text_input("Enter MarkDown4")
                MarkDown5 = st.text_input("Enter MarkDown5")
                Type = st.selectbox("Select the Type of Store", Type,key=3)
                IsHoliday = st.selectbox("Holiday", IsHoliday,key=4)

                # Convert the selected values to binary
                if IsHoliday == True:
                    IsHoliday = 1
                else:
                    IsHoliday = 0

                if Type == "A":
                    Type_B = 0
                    Type_C = 0
                elif Type == "B":
                    Type_B = 1
                    Type_C = 0
                else:
                    Type_B = 0
                    Type_C = 1

            with col3:
                st.write( f'<h5 style="color:rgb(0, 153, 153,0.4);">NOTE: Min & Max given for reference, you can enter any value</h5>', unsafe_allow_html=True )
                CPI = st.text_input("Enter Consumer Price Index (Min : 126, Max : 212)")
                Unemployment = st.text_input("Enter Unemployment percentage(Min : 3.8, Max : 14.4)")
                Size = st.text_input("Enter the size of the store (Min : 34875, Max : 219622)")
                Day_Date = st.selectbox("Day", Day_Date,key=5)
                Month_Date = st.selectbox("Month", Month_Date,key=6)
                year_date = st.text_input("Enter the year")
                submit_button = st.form_submit_button(label="PREDICT SELLING PRICE")
                st.markdown("""
                    <style>
                    div.stButton > button:first-child {
                        background-color: #009999;
                        color: white;
                        width: 100%;
                    }
                    </style>
                """, unsafe_allow_html=True)

            flag=0
            pattern = "^(?:\d+|\d*\.\d+)$"
            for i in [Fuel_Price,MarkDown1,MarkDown2,MarkDown3,MarkDown4,MarkDown5,CPI,Unemployment,Size,year_date,Temperature]:
                if re.match(pattern, i):
                    pass
                else:
                    flag=1
                    break

        if submit_button and flag==1:
            if len(i)==0:
                st.write("please enter a valid number space not allowed")
            else:
                st.write("You have entered an invalid value: ",i)

        if submit_button and flag==0:

            import pickle 
            with open(r"C:\Users\Administrator\Downloads\model.pkl", 'rb') as file:
                loaded_model = pickle.load(file)
            with open(r'C:\Users\Administrator\Downloads\scaler.pkl', 'rb') as f:
                scaler_loaded = pickle.load(f)
 
	#Store	Dept	Temperature	Fuel_Price	MarkDown1	MarkDown2	MarkDown3	MarkDown4	MarkDown5	CPI	Unemployment	Size	month_date	day_date	year_date	Type_B	Type_C	IsHoliday_True
            scale_columns = np.array([[np.log(float(Temperature)), np.log(float(Fuel_Price)), np.log(float(MarkDown1)), np.log(float(MarkDown2)), np.log(float(MarkDown3)), np.log(float(MarkDown4)), np.log(float(MarkDown5)), np.log(float(CPI)), np.log(float(Unemployment)), np.log(float(Size))]])
            new_sample = scaler_loaded.transform(scale_columns)
            f = np.array([[int(store), int(department)]])
            l = np.array([[int(Month_Date), int(Day_Date), int(year_date), int(Type_B), int(Type_C), int(IsHoliday)]])
            new_sample1= np.concatenate((f, new_sample, l), axis=1)
           
            new_pred = loaded_model.predict(new_sample1)[0]
            st.write('## :green[Predicted Weekly Sales:] ', new_pred)            



with tab2:

# Title
    st.title("My Tableau Dashboard")

# Image
    image_url = "https://private-user-images.githubusercontent.com/139530620/290078099-37cdf4db-4b35-426e-ae9a-ffba57af042e.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTEiLCJleHAiOjE3MDI0NDY3ODIsIm5iZiI6MTcwMjQ0NjQ4MiwicGF0aCI6Ii8xMzk1MzA2MjAvMjkwMDc4MDk5LTM3Y2RmNGRiLTRiMzUtNDI2ZS1hZTlhLWZmYmE1N2FmMDQyZS5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBSVdOSllBWDRDU1ZFSDUzQSUyRjIwMjMxMjEzJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDIzMTIxM1QwNTQ4MDJaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT01MzZlODRkNjBmYTVmYmI0OGM4ZDcwZTI1MWE5NWQ1ZGYwMTljNjU1ZTYxYjhmODdmODI1MTBhZWU5ZWNhNzdhJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.UXzXTe8HLhrmv-kSyAIL6JUxi8-tzB3IYON_b-WfME8"  # Replace with the URL of your image
    st.image(image_url, caption="Your Image Caption", use_column_width=True)

with tab3:


# Title
    st.title("My PowerBI Dashboard")

# Image
    image_url = "https://private-user-images.githubusercontent.com/139530620/290077788-b4f70d2c-401c-48d7-b40d-80a931f3961a.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTEiLCJleHAiOjE3MDI0NDY3ODIsIm5iZiI6MTcwMjQ0NjQ4MiwicGF0aCI6Ii8xMzk1MzA2MjAvMjkwMDc3Nzg4LWI0ZjcwZDJjLTQwMWMtNDhkNy1iNDBkLTgwYTkzMWYzOTYxYS5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBSVdOSllBWDRDU1ZFSDUzQSUyRjIwMjMxMjEzJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDIzMTIxM1QwNTQ4MDJaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT02NjhmNGQ3YjNiZTIwZTJmYjFiMGI2OTI5ODcxNzBjZDU4NjI4YjM2MWM2NzYxMzI3YTdlNTc3MmMyMmFjMjZmJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.9Jv03GFjUiu5Hj0WK4bpEMuTWywp5r2lJigqo8Na3Ek"  # Replace with the URL of your image
    st.image(image_url, caption="Your Image Caption", use_column_width=True)

#https://private-user-images.githubusercontent.com/139530620/290091992-fb216dd9-4eef-4744-93e3-18d16b264b27.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTEiLCJleHAiOjE3MDI0NDgxNDMsIm5iZiI6MTcwMjQ0Nzg0MywicGF0aCI6Ii8xMzk1MzA2MjAvMjkwMDkxOTkyLWZiMjE2ZGQ5LTRlZWYtNDc0NC05M2UzLTE4ZDE2YjI2NGIyNy5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBSVdOSllBWDRDU1ZFSDUzQSUyRjIwMjMxMjEzJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDIzMTIxM1QwNjEwNDNaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT0zNTA2NWUzYjMwMjgxMmJmNmY3Mjk3YjAxZjBlNjhhMzMxMzUyNzg4YjRhMjBlMzNlZDAwMmU1OTQ2MWM1NTI1JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.bZgFUMvwaotq8CD3qwDMCyBY7sG6fOttcS0BXStwQP8

with tab4:


# Title
    st.title("Insights")

# Image
    image_url = "https://private-user-images.githubusercontent.com/139530620/290091992-fb216dd9-4eef-4744-93e3-18d16b264b27.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTEiLCJleHAiOjE3MDI0NDgxNDMsIm5iZiI6MTcwMjQ0Nzg0MywicGF0aCI6Ii8xMzk1MzA2MjAvMjkwMDkxOTkyLWZiMjE2ZGQ5LTRlZWYtNDc0NC05M2UzLTE4ZDE2YjI2NGIyNy5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBSVdOSllBWDRDU1ZFSDUzQSUyRjIwMjMxMjEzJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDIzMTIxM1QwNjEwNDNaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT0zNTA2NWUzYjMwMjgxMmJmNmY3Mjk3YjAxZjBlNjhhMzMxMzUyNzg4YjRhMjBlMzNlZDAwMmU1OTQ2MWM1NTI1JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.bZgFUMvwaotq8CD3qwDMCyBY7sG6fOttcS0BXStwQP8"  # Replace with the URL of your image
    st.image(image_url, caption="Your Image Caption", use_column_width=True)
