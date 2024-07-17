import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)

df = pd.read_csv("GDP_deflator.csv", header=0, 
                 usecols=["Country Name", "Country Code", "2010", "2011", "2012", "2013", "2014", "2015", "2016",
                          "2017", "2018", "2019", "2021"])
df2 = pd.read_csv("NominalGDP.csv", header=2, 
                  usecols=["Country Name", "Country Code", "2010", "2011", "2012", "2013", "2014", "2015", "2016",
                           "2017", "2018", "2019", "2021"])

df = df.dropna()
df2 = df2.dropna()

df = df.reset_index(drop=True)
df2 = df2.reset_index(drop=True)


for i, c in df2["Country Code"].items():
    drop = True
    for c1 in df["Country Code"]:
        if c == c1:
            drop = False
            break
    if drop:
        df2.drop(index=i, inplace=True)

df2 = df2.reset_index(drop=True)

st.title('Nominal vs Real GDP')
st.write('The difference between Nominal GDP and Real GDP can give insights into inflation rates. If Nominal GDP grows much faster than Real GDP, it indicates rising prices (inflation).')
st.subheader('Select a Country')

country_list = df['Country Name'].tolist()
country_name = st.selectbox('Select a country', country_list)

ccode = df[df['Country Name'] == country_name].index[0]
index_in_NGDP = df2[df2['Country Code'] == df.at[ccode, 'Country Code']].index[0]

de = df.loc[ccode]
nom = df2.loc[index_in_NGDP]

de_copy = de.drop(labels=['Country Name', 'Country Code'])
nom_copy = nom.drop(labels=['Country Name', 'Country Code'])

real_GDP = [(nom_copy.loc[year] / de_copy.loc[year]) * 100 for year in de_copy.index]
realGDP = pd.Series(data=real_GDP, index=de_copy.index)

combined = pd.DataFrame({
    'Real GDP values': realGDP,
    'Nominal GDP values': nom_copy
})

st.write("Real GDP and Nominal GDP values together are:")
st.write(combined)

st.subheader(f"GDP Values for {country_name}")
fig, ax = plt.subplots()
combined.plot.bar(ax=ax)
plt.xlabel('Years')
plt.ylabel('Values ($)')
plt.title(country_name)
st.pyplot(fig)


st.subheader('Line Plot of GDP Values')
fig, ax = plt.subplots()
realGDP.plot(ax=ax, label='Real GDP', marker='o')
nom_copy.plot(ax=ax, label='Nominal GDP', marker='x')
plt.xlabel('Years')
plt.ylabel('Values ($)')
plt.title(f'{country_name} - Real vs Nominal GDP')
plt.legend()
st.pyplot(fig)

st.subheader('Percentage Change in Real GDP Year-over-Year')
realGDP_pct_change = realGDP.pct_change().dropna() * 100
fig, ax = plt.subplots()
realGDP_pct_change.plot.bar(ax=ax)
plt.xlabel('Years')
plt.ylabel('Percentage Change (%)')
plt.title(f'{country_name} - Year-over-Year Percentage Change in Real GDP')
st.pyplot(fig)

st.subheader('Scatter Plot of Real GDP vs Nominal GDP')
fig, ax = plt.subplots()
plt.scatter(realGDP, nom_copy)
plt.xlabel('Real GDP Values ($)')
plt.ylabel('Nominal GDP Values ($)')
plt.title(f'{country_name} - Real GDP vs Nominal GDP')
plt.grid(True)
st.pyplot(fig)
