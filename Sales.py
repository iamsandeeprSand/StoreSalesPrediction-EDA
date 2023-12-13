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

tab1, tab2, tab3 = st.tabs(["XGBoostRegressor","Dashboard EDA", "RandomForestRegressor"])
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
                Temperature = st.text_input("Enter Temperature in Celsius")
                Fuel_Price = st.text_input("Enter Fuel Price")
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
                CPI = st.text_input("Enter Consumer Price Index")
                Unemployment = st.text_input("Enter Unemployment percentage")
                Size = st.text_input("Enter the size of the store")
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
            #with open(r"/content/t.pkl", 'rb') as f:
            #   t_loaded = pickle.load(f)

            #with open(r"/content/s.pkl", 'rb') as f:
            #    s_loaded = pickle.load(f)
	#Store	Dept	Temperature	Fuel_Price	MarkDown1	MarkDown2	MarkDown3	MarkDown4	MarkDown5	CPI	Unemployment	Size	month_date	day_date	year_date	Type_B	Type_C	IsHoliday_True
            scale_columns = np.array([[np.log(float(Temperature)), np.log(float(Fuel_Price)), np.log(float(MarkDown1)), np.log(float(MarkDown2)), np.log(float(MarkDown3)), np.log(float(MarkDown4)), np.log(float(MarkDown5)), np.log(float(CPI)), np.log(float(Unemployment)), np.log(float(Size))]])
            new_sample = scaler_loaded.transform(scale_columns)
            f = np.array([[int(store), int(department)]])
            l = np.array([[int(Month_Date), int(Day_Date), int(year_date), int(Type_B), int(Type_C), int(IsHoliday)]])
            new_sample1= np.concatenate((f, new_sample, l), axis=1)
            #new_sample_ohe = t_loaded.transform(new_sample[:, [7]]).toarray()
           # new_sample_be = s_loaded.transform(new_sample[:, [8]]).toarray()
            #new_sample = np.concatenate((new_sample[:, [0,1,2, 3, 4, 5, 6,]], new_sample_ohe, new_sample_be), axis=1)

            new_pred = loaded_model.predict(new_sample1)[0]
            st.write('## :green[Predicted Weekly Sales:] ', new_pred)            



with tab2:

    def main():
        st.title("Sales Forecasting Tableau Dashboard")

        # Tableau dashboard link
        tableau_link = "https://public.tableau.com/app/profile/sandeep.r8535/viz/Sales_17017479255590/Dashboard2?publish=yes"

        # Embed the link using an iframe
        st.markdown(f'<iframe src="{tableau_link}" width="800" height="600"></iframe>', unsafe_allow_html=True)

    if __name__ == "__main__":
        main()
