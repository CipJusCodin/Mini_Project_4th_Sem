import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import StackingRegressor
import matplotlib.pyplot as plt

def load_data():
    df = pd.read_csv("GDP.csv", usecols=["Country Name", "Country Code", "2010", "2011", "2012", "2013", "2014",
                                         "2015", "2016", "2017", "2018", "2019", "2021", "2022"])
    return df

df = load_data()

numeric_cols = ["2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2021", "2022"]
imputer = SimpleImputer(strategy='mean')
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

df_long = pd.melt(df, id_vars=['Country Name', 'Country Code'], var_name='Year', value_name='GDP')
df_long['Year'] = df_long['Year'].astype(int)

df_long['Country Code'] = df_long['Country Code'].astype('category').cat.codes

X = df_long[['Country Code', 'Year']]
y = df_long['GDP']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

base_models = [
    ('linear_reg', LinearRegression()),
    ('decision_tree', DecisionTreeRegressor(random_state=42)),
    ('random_forest', RandomForestRegressor(n_estimators=100, random_state=42)),
    ('gradient_boost', GradientBoostingRegressor(n_estimators=100, random_state=42)),
    ('svr', SVR()),
    ('knn', KNeighborsRegressor())
]

stacked_model = StackingRegressor(
    estimators=base_models,
    final_estimator=LinearRegression()
)

stacked_model.fit(X_train, y_train)

country_code_mapping = dict(zip(df['Country Name'], df['Country Code'].astype('category').cat.codes))

def predict_gdp(country_name, year):
    if country_name not in country_code_mapping:
        return f"Country '{country_name}' not found in the data."
    
    country_code = country_code_mapping[country_name]
    input_data = np.array([[country_code, year]])
    gdp_prediction = stacked_model.predict(input_data)
    return gdp_prediction[0]

st.title('GDP Prediction App')
st.write('Enter the details below to predict GDP.')

country_name_input = st.text_input('Country Name')
year_input = st.number_input('Year', min_value=2000, max_value=2100, value=2025)

if st.button('Predict'):
    gdp_prediction = predict_gdp(country_name_input, year_input)
    if isinstance(gdp_prediction, str):
        st.error(gdp_prediction)
    else:
        st.write(f"Predicted GDP for {country_name_input} in {year_input}: ${gdp_prediction:,.2f}")
