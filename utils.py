import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


@st.cache(persist=True)
def load_data():
    data = pd.read_csv('data/mushrooms.csv')
    label = LabelEncoder()
    for col in data.columns:
        data[col] = label.fit_transform(data[col])
    return data


@st.cache(persist=True)
def split(df):
    y = df.type
    x = df.drop(columns=['type'])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    return x_train, x_test, y_train, y_test


def plot_metrics(metrics_list, model, x_test, y_test, class_names):
    if 'Confusion Matrix' in metrics_list:
        st.subheader("Confusion Matrix")
        disp = ConfusionMatrixDisplay.from_estimator(model, x_test, y_test, display_labels=class_names)
        st.pyplot(disp.figure_)

    if 'ROC Curve' in metrics_list:
        st.subheader("ROC Curve")
        disp = RocCurveDisplay.from_estimator(model, x_test, y_test)
        st.pyplot(disp.figure_)

    if 'Precision-Recall Curve' in metrics_list:
        st.subheader("Precision-Recall Curve")
        disp = PrecisionRecallDisplay.from_estimator(model, x_test, y_test)
        st.pyplot(disp.figure_)
