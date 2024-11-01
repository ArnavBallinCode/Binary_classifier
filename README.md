# Mushroom Classification Web App

This repository contains a Streamlit web application for classifying mushrooms as edible or poisonous. The app uses machine learning models such as Support Vector Machine (SVM), Logistic Regression, and Random Forest to make predictions based on user input.

## Features
- Load and preprocess mushroom dataset
- Train and evaluate models: SVM, Logistic Regression, Random Forest
- Display model performance metrics: Accuracy, Precision, Recall
- Visualize confusion matrix, ROC curve, and Precision-Recall curve
- User input form for making predictions on new data

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/arnavballincode/binary_classifier.git
    ```
2. Navigate to the project directory:
    ```bash
    cd binary_classifier
    ```
3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```
2. Open your web browser and go to `http://localhost:8501` to interact with the app.

## Dataset
The mushroom dataset used in this app is preprocessed and loaded from a CSV file. Ensure the `mushrooms.csv` file is in the project directory.

