import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score, matthews_corrcoef
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(page_title="Marketing Campaign Prediction", layout="wide")

# Title
st.title("Marketing Campaign Response Prediction")
st.markdown("### ML Assignment 2 - Vasu Devan S")

# Load models
@st.cache_resource
def load_models():
    models = {}
    model_names = ['logistic_regression', 'decision_tree', 'knn', 'naive_bayes', 'random_forest', 'xgboost']
    failed_models = []
    
    for name in model_names:
        try:
            with open(f'model/saved_models/{name}.pkl', 'rb') as f:
                models[name] = pickle.load(f)
        except Exception as e:
            failed_models.append(name)
            st.sidebar.warning(f"⚠️ Could not load {name.replace('_', ' ').title()}")
    
    if failed_models:
        st.sidebar.info(f"💡 Tip: Install missing dependencies with: pip install xgboost")
    
    return models

models = load_models()

if not models:
    st.error("❌ No models could be loaded. Please check the model files and dependencies.")
    st.stop()

# Sidebar for model selection
st.sidebar.header("Model Selection")

if models:
    st.sidebar.success(f"✅ {len(models)} models loaded successfully")
    model_choice = st.sidebar.selectbox(
        "Choose a model:",
        options=list(models.keys()),
        format_func=lambda x: x.replace('_', ' ').title()
    )
else:
    st.sidebar.error("❌ No models available")
    model_choice = None

# File upload
st.header("1. Upload Test Data")

# Add download button for sample test file
try:
    with open('model/test.csv', 'r') as f:
        sample_csv = f.read()
    st.download_button(
        label="📥 Download Sample Test CSV",
        data=sample_csv,
        file_name="sample_test.csv",
        mime="text/csv",
        help="Download a sample test file to see the expected format"
    )
except:
    pass

st.markdown("---")
uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)
    
    st.success(f"Data loaded successfully! Shape: {df.shape}")
    
    # Show data preview
    with st.expander("View Data Preview"):
        st.dataframe(df.head())
    
    # Preprocess data
    st.header("2. Data Preprocessing")
    
    # Check if Response column exists
    if 'Response' in df.columns:
        X = df.drop('Response', axis=1)
        y = df['Response']
        has_labels = True
    else:
        X = df
        has_labels = False
        st.warning("No 'Response' column found. Will only make predictions.")
    
    # Handle categorical variables
    categorical_cols = X.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        le = LabelEncoder()
        for col in categorical_cols:
            X[col] = le.fit_transform(X[col].astype(str))
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    st.success("Data preprocessing completed!")
    
    # Make predictions
    st.header("3. Model Predictions")
    
    selected_model = models[model_choice]
    predictions = selected_model.predict(X_scaled)
    
    # Get prediction probabilities if available
    try:
        pred_proba = selected_model.predict_proba(X_scaled)[:, 1]
    except:
        pred_proba = None
    
    # Display predictions
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Predictions", len(predictions))
        st.metric("Predicted Positive (1)", np.sum(predictions == 1))
    with col2:
        st.metric("Predicted Negative (0)", np.sum(predictions == 0))
        if pred_proba is not None:
            st.metric("Avg Prediction Probability", f"{np.mean(pred_proba):.3f}")
    
    # If we have true labels, calculate metrics
    if has_labels:
        st.header("4. Evaluation Metrics")
        
        # Calculate metrics
        accuracy = accuracy_score(y, predictions)
        precision = precision_score(y, predictions)
        recall = recall_score(y, predictions)
        f1 = f1_score(y, predictions)
        mcc = matthews_corrcoef(y, predictions)
        
        if pred_proba is not None:
            auc = roc_auc_score(y, pred_proba)
        else:
            auc = None
        
        # Display metrics in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Accuracy", f"{accuracy:.4f}")
            st.metric("Precision", f"{precision:.4f}")
        
        with col2:
            st.metric("Recall", f"{recall:.4f}")
            st.metric("F1 Score", f"{f1:.4f}")
        
        with col3:
            st.metric("MCC", f"{mcc:.4f}")
            if auc is not None:
                st.metric("AUC", f"{auc:.4f}")
        
        # Confusion Matrix
        st.header("5. Confusion Matrix")
        
        cm = confusion_matrix(y, predictions)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'Confusion Matrix - {model_choice.replace("_", " ").title()}')
        st.pyplot(fig)
        
        # Classification Report
        st.header("6. Classification Report")
        
        report = classification_report(y, predictions, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format("{:.3f}"))
        
        # Download predictions
        st.header("7. Download Results")
        
        results_df = df.copy()
        results_df['Predicted_Response'] = predictions
        if pred_proba is not None:
            results_df['Prediction_Probability'] = pred_proba
        
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="Download Predictions as CSV",
            data=csv,
            file_name=f"predictions_{model_choice}.csv",
            mime="text/csv"
        )
    
    else:
        # Just show predictions without evaluation
        st.header("4. Download Predictions")
        
        results_df = df.copy()
        results_df['Predicted_Response'] = predictions
        if pred_proba is not None:
            results_df['Prediction_Probability'] = pred_proba
        
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="Download Predictions as CSV",
            data=csv,
            file_name=f"predictions_{model_choice}.csv",
            mime="text/csv"
        )

else:
    st.info("Please upload a CSV file to begin.")
    
    # Show model information
    st.header("Available Models")
    st.markdown("""
    This application supports the following models:
    - **Logistic Regression**: Linear model for binary classification
    - **Decision Tree**: Non-linear tree-based classifier
    - **K-Nearest Neighbors (KNN)**: Instance-based learning algorithm
    - **Naive Bayes**: Probabilistic classifier based on Bayes' theorem
    - **Random Forest**: Ensemble method using multiple decision trees
    - **XGBoost**: Gradient boosting ensemble method
    
    ### Expected Data Format
    The uploaded CSV should contain the same features as the training data:
    - Customer demographics (Year_Birth, Education, Marital_Status, Income, etc.)
    - Purchase behavior (MntWines, MntFruits, MntMeatProducts, etc.)
    - Campaign responses (AcceptedCmp1-5)
    - Other relevant features
    
    Optionally include a 'Response' column for evaluation.
    """)

# Footer
st.markdown("---")
st.markdown("**Note**: For best results, ensure your test data has the same format as the training data.")
