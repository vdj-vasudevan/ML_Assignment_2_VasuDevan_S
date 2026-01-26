import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score, matthews_corrcoef
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(page_title="Marketing Campaign Prediction", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">🎯 Marketing Campaign Response Prediction</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">ML Assignment 2 - Vasu Devan S</p>', unsafe_allow_html=True)

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
st.sidebar.markdown("## 🤖 Model Selection")
st.sidebar.markdown("---")

if models:
    if len(models) != 6:
        st.sidebar.success(f"✅ {len(models)}/6 models loaded")
    
    model_display = {
        'logistic_regression': '📊 Logistic Regression',
        'decision_tree': '🌳 Decision Tree',
        'knn': '🎯 K-Nearest Neighbors',
        'naive_bayes': '📈 Naive Bayes',
        'random_forest': '🌲 Random Forest',
        'xgboost': '⚡ XGBoost'
    }
    
    model_choice = st.sidebar.selectbox(
        "Choose a model:",
        options=list(models.keys()),
        format_func=lambda x: model_display.get(x, x.replace('_', ' ').title())
    )
else:
    st.sidebar.error("❌ No models available")
    model_choice = None

# Sidebar file upload
st.sidebar.markdown("---")
st.sidebar.markdown("## 📁 Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=['csv'], help="Upload test data with customer features")
st.sidebar.info("👆 Please upload a CSV file to begin prediction.")
try:
    with open('model/test.csv', 'r') as f:
        sample_csv = f.read()
    st.sidebar.download_button(
        label="📥 Sample CSV",
        data=sample_csv,
        file_name="sample_test.csv",
        mime="text/csv",
        use_container_width=True
    )
except:
    pass

st.markdown("---")

# Main processing
if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)
    
    st.success(f"✅ Data loaded successfully! **{df.shape[0]}** rows × **{df.shape[1]}** columns")
    
    # Show data preview with stats
    col1, col2 = st.columns([2, 1])
    with col1:
        with st.expander("📋 View Data Preview", expanded=False):
            st.dataframe(df.head(10), use_container_width=True)
    with col2:
        with st.expander("📊 Quick Stats", expanded=False):
            st.write(f"**Total Records:** {df.shape[0]}")
            st.write(f"**Features:** {df.shape[1]}")
            st.write(f"**Missing Values:** {df.isnull().sum().sum()}")
    
    # Preprocess data
    st.markdown("## ⚙️ Step 2: Data Preprocessing")
    
    with st.spinner("Processing data..."):
        # Check if Response column exists
        if 'Response' in df.columns:
            X = df.drop('Response', axis=1)
            y = df['Response']
            has_labels = True
        else:
            X = df
            has_labels = False
            st.info("ℹ️ No 'Response' column found. Prediction mode only.")
        
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
    
    st.success("✅ Preprocessing completed: Encoded, Imputed, Scaled")
    
    # Make predictions
    st.markdown("## 🎯 Step 3: Model Predictions")
    
    selected_model = models[model_choice]
    predictions = selected_model.predict(X_scaled)
    
    # Get prediction probabilities if available
    try:
        pred_proba = selected_model.predict_proba(X_scaled)[:, 1]
    except:
        pred_proba = None
    
    # Display predictions with better visuals
    col1, col2, col3, col4 = st.columns(4)
    
    positive_count = np.sum(predictions == 1)
    negative_count = np.sum(predictions == 0)
    positive_pct = (positive_count / len(predictions)) * 100
    
    with col1:
        st.metric("📊 Total Predictions", f"{len(predictions):,}")
    with col2:
        st.metric("✅ Positive Response", f"{positive_count:,}", delta=f"{positive_pct:.1f}%")
    with col3:
        st.metric("❌ Negative Response", f"{negative_count:,}", delta=f"{100-positive_pct:.1f}%")
    with col4:
        if pred_proba is not None:
            st.metric("🎲 Avg Confidence", f"{np.mean(pred_proba):.1%}")
        else:
            st.metric("🎲 Avg Confidence", "N/A")
    
    # Prediction distribution chart
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ['#ff6b6b', '#51cf66']
    ax.bar(['Negative (0)', 'Positive (1)'], [negative_count, positive_count], color=colors, alpha=0.8)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Prediction Distribution', fontsize=14, fontweight='bold')
    for i, v in enumerate([negative_count, positive_count]):
        ax.text(i, v + 5, str(v), ha='center', fontweight='bold')
    st.pyplot(fig)
    plt.close()
    
    # If we have true labels, calculate metrics
    if has_labels:
        st.markdown("## 📈 Step 4: Evaluation Metrics")
        
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
        
        # Display metrics in columns with color coding
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.metric("🎯 Accuracy", f"{accuracy:.3f}")
        with col2:
            st.metric("🔍 Precision", f"{precision:.3f}")
        with col3:
            st.metric("📊 Recall", f"{recall:.3f}")
        with col4:
            st.metric("⚖️ F1 Score", f"{f1:.3f}")
        with col5:
            st.metric("🔗 MCC", f"{mcc:.3f}")
        with col6:
            if auc is not None:
                st.metric("📉 AUC", f"{auc:.3f}")
            else:
                st.metric("📉 AUC", "N/A")
        
        # Confusion Matrix
        st.markdown("## 🔲 Step 5: Confusion Matrix")
        
        cm = confusion_matrix(y, predictions)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn_r', ax=ax, cbar_kws={'label': 'Count'}, 
                    linewidths=2, linecolor='white', square=True, annot_kws={'size': 16, 'weight': 'bold'})
            ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
            ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
            ax.set_title(f'Confusion Matrix - {model_choice.replace("_", " ").title()}', fontsize=14, fontweight='bold', pad=20)
            ax.set_xticklabels(['Negative (0)', 'Positive (1)'])
            ax.set_yticklabels(['Negative (0)', 'Positive (1)'])
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.markdown("### 📊 Matrix Breakdown")
            tn, fp, fn, tp = cm.ravel()
            st.metric("✅ True Positives", tp)
            st.metric("✅ True Negatives", tn)
            st.metric("❌ False Positives", fp)
            st.metric("❌ False Negatives", fn)
        
        # Classification Report
        st.markdown("## 📋 Step 6: Classification Report")
        
        report = classification_report(y, predictions, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        
        # Style the dataframe
        styled_df = report_df.style.format("{:.3f}").background_gradient(cmap='RdYlGn', subset=['precision', 'recall', 'f1-score'], vmin=0, vmax=1)
        st.dataframe(styled_df, use_container_width=True)
        
        # Download predictions
        st.markdown("## 💾 Step 7: Download Results")
        
        results_df = df.copy()
        results_df['Predicted_Response'] = predictions
        if pred_proba is not None:
            results_df['Prediction_Probability'] = pred_proba
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Predictions as CSV",
                data=csv,
                file_name=f"predictions_{model_choice}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    else:
        # Just show predictions without evaluation
        st.markdown("## 💾 Step 4: Download Predictions")
        
        results_df = df.copy()
        results_df['Predicted_Response'] = predictions
        if pred_proba is not None:
            results_df['Prediction_Probability'] = pred_proba
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Predictions as CSV",
                data=csv,
                file_name=f"predictions_{model_choice}.csv",
                mime="text/csv",
                use_container_width=True
            )

else:
    
    # Show model information in tabs
    st.markdown("## 📚 About This Application")
    
    tab1, tab2, tab3 = st.tabs(["🤖 Models", "📊 Data Format", "🎯 Features"])
    
    with tab1:
        st.markdown("### Available Machine Learning Models")
        models_info = {
            '📊 Logistic Regression': 'Linear model for binary classification with high interpretability',
            '🌳 Decision Tree': 'Non-linear tree-based classifier with visual decision rules',
            '🎯 K-Nearest Neighbors': 'Instance-based learning algorithm using similarity metrics',
            '📈 Naive Bayes': 'Probabilistic classifier based on Bayes theorem',
            '🌲 Random Forest': 'Ensemble method using multiple decision trees for robust predictions',
            '⚡ XGBoost': 'Gradient boosting ensemble method with best performance'
        }
        for model, desc in models_info.items():
            st.markdown(f"**{model}**  \n{desc}")
            st.markdown("")
    
    with tab2:
        st.markdown("### Expected CSV Format")
        st.markdown("""
        Your CSV should contain customer features:
        - **Demographics**: Year_Birth, Education, Marital_Status, Income, Kidhome, Teenhome
        - **Purchase Behavior**: MntWines, MntFruits, MntMeatProducts, MntFishProducts, MntSweetProducts, MntGoldProds
        - **Purchase Channels**: NumWebPurchases, NumCatalogPurchases, NumStorePurchases, NumDealsPurchases
        - **Campaign History**: AcceptedCmp1, AcceptedCmp2, AcceptedCmp3, AcceptedCmp4, AcceptedCmp5
        - **Other**: Recency, NumWebVisitsMonth, Complain, Days_Enrolled
        
        **Optional**: Include 'Response' column (0/1) for model evaluation.
        """)
    
    with tab3:
        st.markdown("### Key Features (26 total)")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Customer Profile**
            - Year_Birth
            - Education (encoded)
            - Marital_Status (encoded)
            - Income
            - Kidhome, Teenhome
            - Days_Enrolled
            - Recency
            """)
        with col2:
            st.markdown("""
            **Behavioral Data**
            - Product spending (6 categories)
            - Purchase channels (4 types)
            - Campaign responses (5 campaigns)
            - Web visits & complaints
            """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>💡 Tip:</strong> For best results, ensure your test data matches the training data format.</p>
    <p>Built with ❤️ using Streamlit | ML Assignment 2 | Vasu Devan S</p>
</div>
""", unsafe_allow_html=True)
