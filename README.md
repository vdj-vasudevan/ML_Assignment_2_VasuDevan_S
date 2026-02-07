# ML_Assignment_2_VasuDevan_S

## Problem Statement

This project aims to predict customer response to marketing campaigns using machine learning classification models. The goal is to identify which customers are likely to respond positively to a marketing campaign, enabling businesses to optimize their marketing strategies and resource allocation.

## Dataset Description

**Dataset:** Marketing Campaign Dataset from Kaggle (rodsaldanha/arketing-campaign)

**Source:** Kaggle Public Repository

**Dataset Characteristics:**
- **Total Instances:** 2,240 customer records
- **Total Features:** 29 original features
- **Selected Features:** 26 features for modeling (excluding ID, Z_CostContact, Z_Revenue)
- **Target Variable:** Response (Binary: 0 = No Response, 1 = Positive Response)
- **Class Distribution:** Imbalanced binary classification problem
  - Class 0 (No Response): 1,906 instances (85.1%)
  - Class 1 (Positive Response): 334 instances (14.9%)
  - Imbalance Ratio: ~5.7:1

**Selected Features (26 features):**
1. Year_Birth - Customer's year of birth
2. Education - Customer's level of education (Label Encoded)
3. Marital_Status - Customer's marital status (Label Encoded)
4. Income - Customer yearly household income
5. Kidhome - Number of small children in household
6. Teenhome - Number of teenagers in household
7. Days_Enrolled - Days since customer enrollment (derived from Dt_Customer)
8. Recency - Number of days since last purchase
9. MntWines - Amount spent on wine products
10. MntFruits - Amount spent on fruits
11. MntMeatProducts - Amount spent on meat products
12. MntFishProducts - Amount spent on fish products
13. MntSweetProducts - Amount spent on sweets
14. MntGoldProds - Amount spent on gold products
15. NumDealsPurchases - Number of purchases made with discount
16. NumWebPurchases - Number of purchases made through website
17. NumCatalogPurchases - Number of purchases made using catalog
18. NumStorePurchases - Number of purchases made directly in stores
19. NumWebVisitsMonth - Number of visits to company's website in last month
20. AcceptedCmp3 - 1 if customer accepted offer in 3rd campaign, 0 otherwise
21. AcceptedCmp4 - 1 if customer accepted offer in 4th campaign, 0 otherwise
22. AcceptedCmp5 - 1 if customer accepted offer in 5th campaign, 0 otherwise
23. AcceptedCmp1 - 1 if customer accepted offer in 1st campaign, 0 otherwise
24. AcceptedCmp2 - 1 if customer accepted offer in 2nd campaign, 0 otherwise
25. Complain - 1 if customer complained in last 2 years, 0 otherwise
26. Response - Target variable

**Data Preprocessing:**
- Handled missing values using mean imputation for Income feature
- Excluded non-predictive features (ID, Z_CostContact, Z_Revenue)
- Encoded categorical variables (Education, Marital_Status) using Label Encoding
- Converted date feature (Dt_Customer) to numerical Days_Enrolled
- Applied StandardScaler for feature normalization
- Train-Test Split: 80-20 ratio with stratification

## Models Used

Six machine learning models were trained and evaluated to predict customer response:

1. **Logistic Regression** - Linear model for binary classification
2. **Decision Tree** - Non-linear tree-based classifier
3. **K-Nearest Neighbors (KNN)** - Instance-based learning algorithm
4. **Naive Bayes** - Probabilistic classifier based on Bayes' theorem
5. **Random Forest** - Ensemble method using multiple decision trees
6. **XGBoost** - Gradient boosting ensemble method

### Performance Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|--------------|----------|-----|-----------|--------|----|-----|
| Logistic Regression | 0.883929 | 0.910369 | 0.666667 | 0.447761 | 0.535714 | 0.484526 |
| Decision Tree | 0.832589 | 0.679457 | 0.442857 | 0.462687 | 0.452555 | 0.353915 |
| KNN | 0.863839 | 0.798292 | 0.607143 | 0.253731 | 0.357895 | 0.331290 |
| Naive Bayes | 0.803571 | 0.826067 | 0.400000 | 0.626866 | 0.488372 | 0.388544 |
| Random Forest (Ensemble) | 0.879464 | 0.904748 | 0.724138 | 0.313433 | 0.437500 | 0.423861 |
| XGBoost (Ensemble) | 0.890625 | 0.910409 | 0.695652 | 0.477612 | 0.566372 | 0.517982 |

## Live Web Application

🌐 **Access the live app here:** [Marketing Campaign Prediction App](https://vdj-vasudevan-ml-assignment-2-vasudevan-s-app-jx4vsl.streamlit.app/)

🎬 **Demo Available here: **docs/demo.mp4
<video width="800" controls>
  <source src="https://github.com/vdj-vasudevan/ML_Assignment_2_VasuDevan_S/blob/main/docs/demo.mp4" type="video/mp4">
</video>

### How to Use the App

1. **Open the App**: Click the link above to access the live application

2. **Expand Sidebar**: Click the arrow (>) on the top-left to open the sidebar

3. **Select Model**: Choose from 6 available ML models:
   - Logistic Regression
   - Decision Tree
   - K-Nearest Neighbors
   - Naive Bayes
   - Random Forest
   - XGBoost

4. **Upload Data**: 
   - Click "Browse files" to upload your CSV file
   - Or download the sample CSV file provided in the sidebar
   - Your CSV should contain the 26 customer features listed above

5. **View Results**: The app will automatically:
   - Preprocess your data (encoding, imputation, scaling)
   - Generate predictions
   - Display prediction metrics and visualizations
   - Show confusion matrix and classification report (if Response column exists)
   - Allow you to download results as CSV

6. **Explore Features**:
   - View data preview and quick statistics
   - See prediction distribution charts
   - Analyze model performance metrics
   - Download predictions with probabilities

### App Features

- ✅ Interactive model selection
- ✅ Real-time data preprocessing
- ✅ Comprehensive evaluation metrics (Accuracy, Precision, Recall, F1, MCC, AUC)
- ✅ Visual confusion matrix with breakdown
- ✅ Detailed classification report
- ✅ Prediction probability scores
- ✅ Downloadable results
- ✅ Responsive design with collapsible sidebar

## Author

Vasu Devan S

## License

This project is for educational purposes as part of ML Assignment 2.
