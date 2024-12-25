import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt

# Title and Description
st.title("Machine Learning Prototype")
st.write("Upload a CSV file, select a model and feature selection, and view metrics and graphs.")

# File Upload
uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

# Dropdowns for Model and Feature Selection
model_choice = st.selectbox("Select Model", ["SVM", "Random Forest"])
feature_selection_choice = st.radio("Feature Selection", ["None", "Wrapper RFE"])

# Process Button
if st.button("Submit"):
    if uploaded_file is not None:
        # Read the uploaded CSV
        data = pd.read_csv(uploaded_file)
        st.write("Uploaded Dataset:", data.head())

        # Preprocessing: Handle non-numerical target column
        if "Level" in data.columns:
            # Map target levels to numeric values (1 = low, 2 = medium, 3 = high)
            label_encoder = LabelEncoder()
            data['Level'] = label_encoder.fit_transform(data['Level'])

        # Drop non-numerical columns except target
        data_processed = data.select_dtypes(include=["number"]).dropna()

        # Ensure target column is the last one
        X = data_processed.iloc[:, :-1]
        y = data_processed.iloc[:, -1]

        st.write("Processed Dataset:", data_processed.head())

        # Feature Scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Split Data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.7, random_state=42)

        # Feature Selection
        if feature_selection_choice == "Wrapper RFE":
            selector = RFE(estimator=RandomForestClassifier() if model_choice == "Random Forest" else SVC(), n_features_to_select=5)
            X_train = selector.fit_transform(X_train, y_train)
            X_test = selector.transform(X_test)

        # Model Training
        model = RandomForestClassifier() if model_choice == "Random Forest" else SVC(kernel="linear", random_state=42)
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Metrics Calculation
        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average='weighted'),
            "Recall": recall_score(y_test, y_pred, average='weighted'),
            "F1 Score": f1_score(y_test, y_pred, average='weighted'),
            "MCC": matthews_corrcoef(y_test, y_pred),
        }

        # Display Metrics
        st.write("### Metrics")
        st.write(metrics)

        # Visualize Metrics
        st.write("### Metrics Graph")
        fig, ax = plt.subplots()
        ax.bar(metrics.keys(), metrics.values())
        st.pyplot(fig)
    else:
        st.error("Please upload a valid CSV file.")
