import streamlit as st
import numpy as np
import utils
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score

# Set the page configuration
st.set_page_config(
    page_title="Binary Classification Web App",
    page_icon=" ",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for radiant background, animated bubbles, and button hover effects
st.markdown("""
    <style>
    body {
        background: linear-gradient(45deg, #ff6b6b, #ffe66d);
        background-size: 400% 400%;
        animation: Gradient 15s ease infinite;
    }
    
    @keyframes Gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .main {
        background: rgba(255, 255, 255, 0.8);
        padding: 20px;
        border-radius: 10px;
    }
    
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        transition-duration: 0.4s;
        cursor: pointer;
        border-radius: 12px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
    
    .stButton>button:hover {
        background-color: white;
        color: black;
        box-shadow: 0 12px 24px rgba(0,0,0,0.2);
        transform: scale(1.1);
    }
    
    .stButton>button:active {
        transform: scale(0.95);
    }
    
    .main {
        position: relative;
        overflow: hidden;
    }
    
    .main::before {
        content: '';
        position: absolute;
        top: -100px;
        left: -100px;
        width: 200px;
        height: 200px;
        background: rgba(255, 255, 255, 0.3);
        border-radius: 50%;
        pointer-events: none;
        animation: bubble 20s linear infinite;
    }
    
    @keyframes bubble {
        0% {
            transform: translate(0, 0);
            opacity: 1;
        }
        100% {
            transform: translate(calc(100vw - 100px), calc(100vh - 100px));
            opacity: 0;
        }
    }
    
    </style>
    """, unsafe_allow_html=True)

# Main function
def main():
    # Header
    st.title("Binary Classification Web App")
    st.image("https://images.pexels.com/photos/2569641/pexels-photo-2569641.jpeg", width=700)  # Add a relevant image or logo
    
    
    # Introduction
    st.markdown("""
    Welcome to the Binary Classification Web App! 
    
    This application allows you to understand the paramters using machine learning models like Support Vector Machine (SVM), Logistic Regression, and Random Forest Classification.
    
    Use the sidebar to select the model and adjust the parameters to see how accurately the models perform.
    """)

    st.sidebar.title("Navigation")
    st.sidebar.markdown("Use the options below to navigate through the application.")

    df = utils.load_data()
    x_train, x_test, y_train, y_test = utils.split(df)
    class_names = ["edible", "poisonous"]

    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest Classification"))

    if classifier == 'Support Vector Machine (SVM)':
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C')
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key='kernel')
        gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key='gamma')

        metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))

        if st.sidebar.button("Classify", key="classify_svm"):
            st.subheader("Support Vector Machine (SVM) Results")
            model = SVC(C=C, kernel=kernel, gamma=gamma)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", round(accuracy, 2))
            st.write("Precision: ", round(precision_score(y_test, y_pred), 2))
            st.write("Recall: ", round(recall_score(y_test, y_pred), 2))
            utils.plot_metrics(metrics, model, x_test, y_test, class_names)

            if st.button("Show Predictions", key="show_predictions_svm"):
                st.subheader("Predictions")
                predictions = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
                st.write(predictions)

    elif classifier == 'Logistic Regression':
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='Lr')
        max_iter = st.sidebar.slider("Maximum no. of iterations", 100, 500, key='max_iter')

        metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))

        if st.sidebar.button("Classify", key="classify_lr"):
            st.subheader("Logistic Regression Results")
            model = LogisticRegression(C=C, max_iter=max_iter)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", round(accuracy, 2))
            st.write("Precision: ", round(precision_score(y_test, y_pred), 2))
            st.write("Recall: ", round(recall_score(y_test, y_pred), 2))
            utils.plot_metrics(metrics, model, x_test, y_test, class_names)

            if st.button("Show Predictions", key="show_predictions_lr"):
                st.subheader("Predictions")
                predictions = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
                st.write(predictions)

    elif classifier == 'Random Forest Classification':
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.number_input("Number of trees in the forest", 100, 5000, step=10, key='n_estimators')
        max_depth = st.sidebar.number_input("Maximum depth of the tree", 1, 100, step=2, key='max_depth')
        bootstrap = st.sidebar.radio("Bootstrap samples when building trees", [True, False], key='bootstrap')
        
        metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))

        if st.sidebar.button("Classify", key="classify_rf"):
            st.subheader("Random Forest Results")
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap, n_jobs=-1)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", round(accuracy, 2))
            st.write("Precision: ", round(precision_score(y_test, y_pred), 2))
            st.write("Recall: ", round(recall_score(y_test, y_pred), 2))
            utils.plot_metrics(metrics, model, x_test, y_test, class_names)

            if st.button("Show Predictions", key="show_predictions_rf"):
                st.subheader("Predictions")
                predictions = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
                st.write(predictions)

    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Mushroom Data Set (Classification)")
        st.write(df)
     # LinkedIn button at the bottom
    linkedin_url = "https://www.linkedin.com/in/arnav-angarkar/"
    if st.button("Connect on LinkedIn"):
        st.write(f"[Visit my LinkedIn Profile]({linkedin_url})")


if __name__ == '__main__':
    main()

