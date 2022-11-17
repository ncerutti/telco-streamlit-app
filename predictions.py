import streamlit as st

import pickle
import pandas as pd
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import xgboost as xgb


def load_pickles(model_pickle_path):
    model_pickle_opener = open(model_pickle_path, "rb")
    model = pickle.load(model_pickle_opener)
    return model


def pre_process_data(data, pipeline):
    data = pipeline.transform(data)
    return data


def predict(data, model):
    data = model.predict(data)
    return data


def generate_predictions(test_df):
    model_pickle_path = "pipe.pickle"
    model = load_pickles(model_pickle_path)
    prediction = predict(test_df, model)
    return prediction


if __name__ == "__main__":
    st.title("Churn Prediction")

    # Read the test data
    holdout = pd.read_csv("./data/holdout_data.csv")

    st.text("Enter customer data to predict churn.")
    holdout_nolab = holdout.drop("Churn", axis=1)
    chosen_customer = st.selectbox(
        "Select a customer.", holdout_nolab.loc[:, "customerID"]
    )

    # Visualize the data
    st.table(
        holdout_nolab.loc[holdout_nolab.loc[:, "customerID"] == chosen_customer, :]
    )

    # Generate predictions with a button
    if st.button("Predict!!1!!1!"):
        pred = generate_predictions(
            holdout_nolab.loc[holdout.loc[:, "customerID"] == chosen_customer, :]
        )
        if bool(pred):
            st.write("Customer will churn, my dude.")
        else:
            st.write("Customer will not churn, my dude.")
