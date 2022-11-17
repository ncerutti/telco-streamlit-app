import streamlit as st
import pandas as pd

st.write("Here we create data using a table:")
st.write(pd.DataFrame({
    'first column': [1, 2, 33, 4],
    'second column': [10, 20, 30, 40]
}))

data  = pd.read_csv('./data/training_data.csv', index_col=0)

st.write("Pasting a dataframe:")
st.write(data)

st.write("Churners")
target_bins = data.loc[:, 'Churn'].value_counts()
st.bar_chart(target_bins)

# Bar chart of InternetService by Contract

st.write("InternetService by Contract")
internet_service = data.loc[:, ['InternetService', 'Contract']]
internet_service = internet_service.groupby(['InternetService', 'Contract']).size().reset_index(name='counts')
internet_service = internet_service.pivot(index='InternetService', columns='Contract', values='counts')
st.bar_chart(internet_service)
