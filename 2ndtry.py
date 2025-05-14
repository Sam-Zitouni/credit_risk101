import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve, classification_report, confusion_matrix, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import RFE
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
import shap
from optbinning import OptimalBinning
from fairlearn.metrics import demographic_parity_difference
from sklearn.inspection import PartialDependenceDisplay
import joblib
import io
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(123)

# Streamlit App Title
st.title("Enhanced Credit Risk Management App")
st.markdown("""
This app predicts credit default risk using an optimized XGBoost model with advanced preprocessing, feature engineering, and ensemble techniques.
It includes Expected Loss (EL) calculations, fairness analysis, and comprehensive visualizations.
""")

# 1. Load and Preprocess the UCI Credit Card Default Dataset
@st.cache_data
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"
    df = pd.read_excel(url, header=1)
    df = df.rename(columns={'default payment next month': 'default'})
    df = df.drop(columns=['ID'])
    return df

df = load_data()
st.write("Dataset Loaded:", df.head())

# Outlier capping
def cap_outliers(df, column, lower_quantile=0.01, upper_quantile=0.99):
    lower = df[column].quantile(lower_quantile)
    upper = df[column].quantile(upper_quantile)
    df[column] = df[column].clip(lower, upper)
    return df

# Feature Engineering
bins = [20, 30, 40, 50, 60, 100]
labels = ['20-30', '30-40', '40-50', '50-60', '60+']
df['AGE_GROUP'] = pd.cut(df['AGE'], bins=bins, labels=labels, include_lowest=True)
df['AGE_PAY_0_INTERACTION'] = df['AGE'] * df['PAY_0']
for i in range(1, 7):
    df[f'PAY_TO_BILL_RATIO_{i}'] = df[f'PAY_AMT{i}'] / (df[f'BILL_AMT{i}'] + 1e-6)
df['BILL_AMT_TREND'] = df['BILL_AMT1'] - df['BILL_AMT6']
df['PAY_STATUS_TREND'] = df['PAY_0'] - df['PAY_6']

# Define feature types
numerical_cols = ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4',
                  'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3',
                  'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6', 'AGE_PAY_0_INTERACTION',
                  'PAY_TO_BILL_RATIO_1', 'PAY_TO_BILL_RATIO_2', 'PAY_TO_BILL_RATIO_3',
                  'PAY_TO_BILL_RATIO_4', 'PAY_TO_BILL_RATIO_5', 'PAY_TO_BILL_RATIO_6',
                  'BILL_AMT_TREND', 'PAY_STATUS_TREND']
categorical_cols = ['SEX', 'EDUCATION', 'MARRIAGE', 'PAY_0', 'PAY_2', 'PAY_3',
                    'PAY_4', 'PAY_5', 'PAY_6', 'AGE_GROUP']

# Handle outliers
for col in numerical_cols:
    df = cap_outliers(df, col)

# Handle missing values with KNN imputation
imputer = KNNImputer(n_neighbors=5)
df[numerical_cols] = imputer.fit_transform(df[numerical_cols])

# Impute only numerical categorical columns, excluding AGE_GROUP
categorical_cols_to_impute = [col for col in categorical_cols if col != 'AGE_GROUP']
if categorical_cols_to_impute:
    cat_imputer = KNNImputer(n_neighbors=5)
    df[categorical_cols_to_impute] = cat_imputer.fit_transform(df[categorical_cols_to_impute])

# Encode categorical variables
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Standardize numerical features
scaler = StandardScaler()
df_encoded[numerical_cols] = scaler.fit_transform(df_encoded[numerical_cols])

# 2. WoE and IV Calculation
@st.cache_data
def calculate_woe_iv(df, feature, target):
    optb = OptimalBinning(name=feature, dtype="numerical" if df[feature].dtype in ['int64', 'float64'] else "categorical", max_n_bins=10, min_bin_size=0.05)
    try:
        optb.fit(df[feature].values, df[target].values)
        binning_table = optb.binning_table.build()
        iv = binning_table['IV'].sum()
        woe = binning_table[['Bin', 'WoE']].set_index('Bin').to_dict()['WoE']
        return woe, iv, optb
    except:
        return {}, 0, None

woe_dict = {}
iv_dict = {}
optb_dict = {}
features = [col for col in df_encoded.columns if col != 'default']
for feature in features:
    woe, iv, optb = calculate_woe_iv(df_encoded, feature, 'default')
    woe_dict[feature] = woe
    iv_dict[feature] = iv
    optb_dict[feature] = optb

# Feature selection with IV and RFE
iv_threshold = 0.02
selected_features = [f for f, iv in iv_dict.items() if iv > iv_threshold]
if not selected_features:
    st.warning("No features passed IV threshold. Using all features.")
    selected_features = features

# Transform features to WoE values
df_woe = df_encoded.copy()
for feature in selected_features:
    if optb_dict[feature] is not None:
        df_woe[feature] = optb_dict[feature].transform(df_woe[feature], metric="woe")
    else:
        df_woe = df_woe.drop(columns=[feature])
        selected_features.remove(feature)

# Normalize WoE features
woe_scaler = StandardScaler()
df_woe[selectedwatermark] = False
df_woe[selected_features] = woe_scaler.fit_transform(df_woe[selected_features])

# 3. Train-Test Split
X = df_woe[selected_features]
y = df_woe['default']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

# 4. Model Training
# Handle class imbalance with SMOTE
smote = SMOTE(random_state=123)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# Optimized XGBoost
xgb_model = xgb.XGBClassifier(
    max_depth=5, learning_rate=0.1, n_estimators=200,
    subsample=0.8, colsample_bytree=0.8, random_state=123, eval_metric='logloss'
)
# Calibrate probabilities
calibrated_xgb = CalibratedClassifierCV(xgb_model, method='sigmoid', cv=5)
calibrated_xgb.fit(X_train_bal, y_train_bal)

# Ensemble with LightGBM and Random Forest
lgb_model = lgb.LGBMClassifier(random_state=123)
rf_model = RandomForestClassifier(random_state=123)
ensemble = VotingClassifier(
    estimators=[('xgb', calibrated_xgb), ('lgb', lgb_model), ('rf', rf_model)],
    voting='soft'
)
ensemble.fit(X_train_bal, y_train_bal)
y_pred_prob = ensemble.predict_proba(X_test)[:, 1]

# Save the trained model
model_filename = "credit_risk_model.joblib"
joblib.dump(ensemble, model_filename)

# Cache SHAP explainer
@st.cache_resource
def get_explainer(model, X):
    # Use KernelExplainer for ensemble compatibility
    def model_predict(X):
        return model.predict_proba(X)[:, 1]
    return shap.KernelExplainer(model_predict, X)
explainer = get_explainer(ensemble, X_train)

# Option to load a saved model
st.sidebar.header("Load Saved Model")
uploaded_model = st.sidebar.file_uploader("Upload a saved model (.joblib)", type=["joblib"])
if uploaded_model is not None:
    ensemble = joblib.load(uploaded_model)
    st.sidebar.success("Model loaded successfully!")
    y_pred_prob = ensemble.predict_proba(X_test)[:, 1]

# 5. Prediction Interface with Threshold and EL
st.header("Predict Default Risk")
st.markdown("Enter client details to predict default probability and calculate Expected Loss (EL).")

# Create input fields
input_data = {}
with st.form("prediction_form"):
    input_data['LIMIT_BAL'] = st.slider("Credit Limit (LIMIT_BAL)", 10000, 1000000, 50000)
    input_data['AGE'] = st.slider("Age", 20, 80, 30)
    input_data['PAY_0'] = st.selectbox("Payment Status (PAY_0)", [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8])
    input_data['BILL_AMT1'] = st.slider("Bill Amount (BILL_AMT1)", 0, 500000, 10000)
    input_data['PAY_AMT1'] = st.slider("Payment Amount (PAY_AMT1)", 0, 500000, 5000)
    lgd = st.slider("Loss Given Default (LGD)", 0.0, 1.0, 0.8, 0.05)
    ead = st.slider("Exposure at Default (EAD)", 10000, 1000000, input_data['LIMIT_BAL'])
    threshold = st.slider("Prediction Threshold", 0.0, 1.0, 0.5, 0.05)
    submitted = st.form_submit_button("Predict")

# Process input data
if submitted:
    input_df = pd.DataFrame([input_data])
    input_df['AGE_GROUP'] = pd.cut(input_df['AGE'], bins=bins, labels=labels, include_lowest=True)
    input_df['AGE_PAY_0_INTERACTION'] = input_df['AGE'] * input_df['PAY_0']
    input_df['PAY_TO_BILL_RATIO_1'] = input_df['PAY_AMT1'] / (input_df['BILL_AMT1'] + 1e-6)
    input_df = pd.get_dummies(input_df)
    for col in selected_features:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[selected_features]
    numerical_input_cols = [col for col in numerical_cols if col in input_df.columns]
    if numerical_input_cols:
        input_df[numerical_input_cols] = scaler.transform(input_df[numerical_input_cols])
    for feature in selected_features:
        if feature in optb_dict and optb_dict[feature] is not None:
            input_df[feature] = optb_dict[feature].transform(input_df[feature], metric="woe")
    input_df = woe_scaler.transform(input_df)
    prob = ensemble.predict_proba(input_df)[:, 1][0]
    pred = 1 if prob > threshold else 0
    st.write(f"**Predicted Default Probability (PD):** {prob:.2%}")
    st.write(f"**Predicted Default (Threshold {threshold:.2f}):** {'Yes' if pred else 'No'}")
    if pred:
        st.error("High risk of default!")
    else:
        st.success("Low risk of default.")
    el = prob * lgd * ead
    st.write(f"**Expected Loss (EL):** ${el:,.2f}")
    st.markdown(f"**EL Breakdown:** PD = {prob:.2%}, LGD = {lgd:.0%}, EAD = ${ead:,}")
    # SHAP explanation
    shap_values = explainer.shap_values(input_df)
    fig, ax = plt.subplots()
    shap.bar_plot(shap_values[0], feature_names=selected_features, max_display=5)
    st.pyplot(fig)
    # Save SHAP plot
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    st.download_button(
        label="Download SHAP Plot",
        data=buf,
        file_name="shap_plot.png",
        mime="image/png"
    )

# 6. Enhanced Model Performance and Visualizations
st.header("Model Performance and Insights")

# Optimal Threshold
thresholds = np.arange(0.1, 0.9, 0.05)
f1_scores = [f1_score(y_test, y_pred_prob > t) for t in thresholds]
optimal_threshold = thresholds[np.argmax(f1_scores)]
st.write(f"**Optimal Threshold (F1-Score):** {optimal_threshold:.2f}")
y_pred = (y_pred_prob > optimal_threshold).astype(int)

# Model Performance Metrics
st.subheader("Ensemble Model Performance")
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))
balanced_acc = balanced_accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
st.write(f"**Balanced Accuracy:** {balanced_acc:.3f}")
st.write(f"**F1-Score:** {f1:.3f}")
cv_scores = cross_val_score(ensemble, X, y, cv=5, scoring='balanced_accuracy')
st.write(f"**Cross-Validated Balanced Accuracy:** {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# Download model
st.download_button(
    label="Download Trained Model",
    data=open(model_filename, 'rb').read(),
    file_name=model_filename,
    mime='application/octet-stream'
)

# Visualizations
st.subheader("Visualizations")
visualizations = {
    "IV Bar Plot": st.checkbox("Show IV Bar Plot", value=True),
    "ROC Curve": st.checkbox("Show ROC Curve", value=True),
    "Precision-Recall Curve": st.checkbox("Show Precision-Recall Curve", value=True),
    "SHAP Feature Importance": st.checkbox("Show SHAP Feature Importance", value=True),
    "Confusion Matrix": st.checkbox("Show Confusion Matrix", value=True),
    "Age vs. Default Rate": st.checkbox("Show Age vs. Default Rate", value=True),
    "Calibration Plot": st.checkbox("Show Calibration Plot", value=True),
    "Partial Dependence Plot": st.checkbox("Show Partial Dependence Plot", value=True)
}

# Function to save plot
def save_plot(fig, filename):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    return buf

# IV Bar Plot
if visualizations["IV Bar Plot"]:
    fig, ax = plt.subplots()
    iv_df = pd.DataFrame({'Feature': iv_dict.keys(), 'IV': iv_dict.values()})
    iv_df = iv_df.sort_values('IV', ascending=False).head(10)
    sns.barplot(x='IV', y='Feature', data=iv_df, ax=ax)
    ax.set_title('Top 10 Features by Information Value (IV)')
    st.pyplot(fig)
    buf = save_plot(fig, "iv_bar_plot.png")
    st.download_button(
        label="Download IV Bar Plot",
        data=buf,
        file_name="iv_bar_plot.png",
        mime="image/png"
    )

# ROC Curve
if visualizations["ROC Curve"]:
    fig, ax = plt.subplots()
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve - Ensemble')
    ax.legend(loc='lower right')
    st.pyplot(fig)
    buf = save_plot(fig, "roc_curve.png")
    st.download_button(
        label="Download ROC Curve",
        data=buf,
        file_name="roc_curve.png",
        mime="image/png"
    )

# Precision-Recall Curve
if visualizations["Precision-Recall Curve"]:
    fig, ax = plt.subplots()
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
    ax.plot(recall, precision, label='Precision-Recall Curve')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve - Ensemble')
    optimal_idx = np.argmax(2 * precision * recall / (precision + recall + 1e-10))
    ax.plot(recall[optimal_idx], precision[optimal_idx], 'ro', label=f'Optimal Threshold = {thresholds[optimal_idx]:.2f}')
    ax.legend()
    st.pyplot(fig)
    buf = save_plot(fig, "precision_recall_curve.png")
    st.download_button(
        label="Download Precision-Recall Curve",
        data=buf,
        file_name="precision_recall_curve.png",
        mime="image/png"
    )

# SHAP Feature Importance
if visualizations["SHAP Feature Importance"]:
    fig, ax = plt.subplots()
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, plot_type="bar", max_display=10, show=False)
    plt.title('SHAP Feature Importance - Ensemble')
    st.pyplot(fig)
    buf = save_plot(fig, "shap_feature_importance.png")
    st.download_button(
        label="Download SHAP Feature Importance",
        data=buf,
        file_name="shap_feature_importance.png",
        mime="image/png"
    )

# Confusion Matrix
if visualizations["Confusion Matrix"]:
    fig, ax = plt.subplots()
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title('Confusion Matrix - Ensemble')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)
    buf = save_plot(fig, "confusion_matrix.png")
    st.download_button(
        label="Download Confusion Matrix",
        data=buf,
        file_name="confusion_matrix.png",
        mime="image/png"
    )

# Age vs. Default Rate
if visualizations["Age vs. Default Rate"]:
    fig, ax = plt.subplots()
    age_default_rate = df.groupby('AGE_GROUP')['default'].mean()
    sns.barplot(x='AGE_GROUP', y='default', data=age_default_rate.reset_index(), ax=ax)
    ax.set_title('Default Rate by Age Group')
    ax.set_xlabel('Age Group')
    ax.set_ylabel('Default Rate')
    st.pyplot(fig)
    buf = save_plot(fig, "age_default_rate.png")
    st.download_button(
        label="Download Age vs. Default Rate",
        data=buf,
        file_name="age_default_rate.png",
        mime="image/png"
    )

# Calibration Plot
if visualizations["Calibration Plot"]:
    fig, ax = plt.subplots()
    prob_true, prob_pred = calibration_curve(y_test, y_pred_prob, n_bins=10)
    ax.plot(prob_pred, prob_true, marker='o', label='Calibration Curve')
    ax.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title('Calibration Plot - Ensemble')
    ax.legend()
    st.pyplot(fig)
    buf = save_plot(fig, "calibration_plot.png")
    st.download_button(
        label="Download Calibration Plot",
        data=buf,
        file_name="calibration_plot.png",
        mime="image/png"
    )

# Partial Dependence Plot
if visualizations["Partial Dependence Plot"]:
    fig, ax = plt.subplots(figsize=(10, 6))
    top_features = iv_df['Feature'].head(2).tolist()
    # Get feature indices
    feature_indices = [selected_features.index(f) for f in top_features if f in selected_features]
    if feature_indices:
        display = PartialDependenceDisplay.from_estimator(
            ensemble, X_train, features=feature_indices, feature_names=selected_features, ax=ax
        )
        ax.set_title('Partial Dependence Plots')
        st.pyplot(fig)
        buf = save_plot(fig, "partial_dependence_plot.png")
        st.download_button(
            label="Download Partial Dependence Plot",
            data=buf,
            file_name="partial_dependence_plot.png",
            mime="image/png"
        )
    else:
        st.warning("Selected features for PDP not found in training data.")

# 7. Enhanced Analysis
st.subheader("Enhanced Analysis")

# Dataset Insights
default_rate = df['default'].mean()
st.write(f"**Dataset Default Rate:** {default_rate:.3f} ({default_rate*100:.1f}%)")
st.write(f"**Number of Features Selected (IV > {iv_threshold}):** {len(selected_features)}")

# Fairness Analysis
st.write("**Fairness Analysis:**")
dp_diff = demographic_parity_difference(y_test, y_pred, sensitive_features=df['AGE_GROUP'].iloc[X_test.index])
st.write(f"Demographic Parity Difference (Age Group): {dp_diff:.3f}")

# Feature Importance Tables
st.write("**Top 5 Features by IV:**")
st.write(iv_df.head(5)[['Feature', 'IV']])

st.write("**Top 5 Features by SHAP Importance:**")
shap_df = pd.DataFrame({
    'Feature': selected_features,
    'SHAP Importance': np.abs(shap_values).mean(axis=0)
}).sort_values('SHAP Importance', ascending=False).head(5)
st.write(shap_df)

# Confusion Matrix Insights
st.write("**Confusion Matrix Insights:**")
st.write(f"True Negatives: {cm[0,0]}")
st.write(f"False Positives: {cm[0,1]}")
st.write(f"False Negatives: {cm[1,0]}")
st.write(f"True Positives: {cm[1,1]}")
st.write(f"**Recall for Defaults:** {cm[1,1] / (cm[1,1] + cm[1,0]):.3f}")
st.write(f"**Precision for Defaults:** {cm[1,1] / (cm[1,1] + cm[0,1]):.3f}")