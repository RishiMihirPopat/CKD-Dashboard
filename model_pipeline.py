"""
CKD Biostatistical & Machine Learning Pipeline
Extracts, computes, and caches all stats, models, metrics, and SHAP values
from CKD.ipynb for the Merck Streamlit Dashboard.
"""

import os
import pickle
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    average_precision_score, precision_recall_fscore_support,
    roc_curve, precision_recall_curve
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import shap

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Chronic_Kidney_Dsease_data.csv")
CACHE_PATH = os.path.join(BASE_DIR, "ckd_pipeline_cache.pkl")

def cramers_v(table):
    chi2 = stats.chi2_contingency(table)[0]
    n = table.values.sum()
    r, c = table.shape
    if min(r-1, c-1) == 0:
        return 0.0
    return np.sqrt((chi2/n)/(min(r-1, c-1)))

def run_or_load_pipeline(force_recompute=False):
    if os.path.exists(CACHE_PATH) and not force_recompute:
        try:
            with open(CACHE_PATH, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading cache: {e}. Recomputing pipeline...")

    print("Running CKD pipeline analysis...")
    df = pd.read_csv(DATA_PATH)
    
    # 1. Cleaning & Feature Taxonomy
    df_clean = df.drop(columns=["PatientID", "DoctorInCharge"], errors="ignore")
    
    categorical_cols = [c for c in df_clean.columns if df_clean[c].nunique() <= 5]
    numerical_cols = [c for c in df_clean.columns if c not in categorical_cols]
    
    categorical_cols_no_target = [c for c in categorical_cols if c != "Diagnosis"]
    numerical_cols_no_target = [c for c in numerical_cols if c != "Diagnosis"]
    
    # 2. Shapiro-Wilk Normality Checks
    shapiro_results = []
    for col in numerical_cols_no_target:
        stat, p = stats.shapiro(df_clean[col].dropna())
        sk = stats.skew(df_clean[col].dropna())
        kt = stats.kurtosis(df_clean[col].dropna())
        shapiro_results.append({
            "Feature": col,
            "Statistic": round(stat, 4),
            "P_Value": p,
            "Skewness": round(sk, 4),
            "Kurtosis": round(kt, 4),
            "Normal": p >= 0.05
        })
    shapiro_df = pd.DataFrame(shapiro_results)
    
    # 3. Chi-Square Test for Categorical Variables
    chi2_results = []
    for col in categorical_cols_no_target:
        table = pd.crosstab(df_clean[col], df_clean["Diagnosis"])
        chi2, p, dof, ex = stats.chi2_contingency(table)
        cv = cramers_v(table)
        chi2_results.append({
            "Feature": col,
            "Chi2_Stat": round(chi2, 4),
            "P_Value": p,
            "DOF": dof,
            "Cramers_V": round(cv, 4),
            "Significant": p < 0.05
        })
    chi2_df = pd.DataFrame(chi2_results).sort_values("P_Value")
    
    # 4. Mann-Whitney U Test for Numerical Variables
    group0 = df_clean[df_clean["Diagnosis"] == 0]
    group1 = df_clean[df_clean["Diagnosis"] == 1]
    
    mw_results = []
    for col in numerical_cols_no_target:
        u_stat, p = stats.mannwhitneyu(group0[col], group1[col], alternative="two-sided")
        n0, n1 = len(group0), len(group1)
        r_rb = 1.0 - (2.0 * u_stat / (n0 * n1))
        mw_results.append({
            "Feature": col,
            "U_Statistic": round(u_stat, 2),
            "P_Value": p,
            "Rank_Biserial_R": round(r_rb, 4),
            "Abs_Effect_Size": abs(round(r_rb, 4)),
            "Significant": p < 0.05,
            "Mean_NoCKD": round(group0[col].mean(), 2),
            "Mean_CKD": round(group1[col].mean(), 2)
        })
    mw_df = pd.DataFrame(mw_results).sort_values("Abs_Effect_Size", ascending=False)
    
    # 5. Final Top 10 Statistically Significant Features
    final_features = [
        "SerumCreatinine", "GFR", "Itching", "FastingBloodSugar",
        "MuscleCramps", "BUNLevels", "ProteinInUrine",
        "SystolicBP", "HbA1c", "BMI"
    ]
    
    X = df_clean[final_features]
    y = df_clean["Diagnosis"]
    
    # 6. Stratified Train-Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # 7. Robust Scaling
    scaler = RobustScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=final_features, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=final_features, index=X_test.index)
    
    # 8. Model Evaluations (LR, RF, XGB Baseline, XGB Tuned)
    models = {}
    metrics = {}
    curves = {}
    threshold_sweeps = {}
    
    # --- Logistic Regression ---
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    lr_prob = lr.predict_proba(X_test_scaled)[:, 1]
    models["Logistic Regression"] = lr
    
    # --- Random Forest ---
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)
    rf_prob = rf.predict_proba(X_test_scaled)[:, 1]
    models["Random Forest"] = rf
    
    # --- XGBoost (Baseline) ---
    xgb_base = XGBClassifier(eval_metric="logloss", random_state=42)
    xgb_base.fit(X_train_scaled, y_train)
    xgb_base_prob = xgb_base.predict_proba(X_test_scaled)[:, 1]
    models["XGBoost (Baseline)"] = xgb_base
    
    # --- XGBoost (Tuned with SMOTE Pipeline) ---
    imb_pipeline = ImbPipeline([
        ("smote", SMOTE(random_state=42)),
        ("xgb", XGBClassifier(eval_metric="logloss", random_state=42))
    ])
    
    param_distributions = {
        "xgb__n_estimators": [100, 200, 300, 500],
        "xgb__max_depth": [3, 4, 5, 6, 8],
        "xgb__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "xgb__subsample": [0.6, 0.8, 1.0],
        "xgb__colsample_bytree": [0.6, 0.8, 1.0],
        "xgb__min_child_weight": [1, 3, 5],
        "xgb__gamma": [0, 0.1, 0.3],
        "xgb__scale_pos_weight": [1, 3, 5, 9]
    }
    
    def no_ckd_pr_auc(estimator, X_val, y_val):
        y_prob_class0 = estimator.predict_proba(X_val)[:, 0]
        y_true_class0 = (y_val == 0).astype(int)
        return average_precision_score(y_true_class0, y_prob_class0)
        
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    random_search = RandomizedSearchCV(
        imb_pipeline,
        param_distributions=param_distributions,
        n_iter=30,
        scoring=no_ckd_pr_auc,
        cv=cv,
        random_state=42,
        n_jobs=-1
    )
    random_search.fit(X_train_scaled, y_train)
    xgb_tuned = random_search.best_estimator_
    xgb_tuned_prob = xgb_tuned.predict_proba(X_test_scaled)[:, 1]
    models["XGBoost (Tuned + SMOTE)"] = xgb_tuned
    
    probs = {
        "Logistic Regression": lr_prob,
        "Random Forest": rf_prob,
        "XGBoost (Baseline)": xgb_base_prob,
        "XGBoost (Tuned + SMOTE)": xgb_tuned_prob
    }
    
    # Calculate metrics for each model
    summary_rows = []
    for name, y_prob in probs.items():
        y_pred = (y_prob >= 0.5).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision_ckd = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1_ckd = 2 * (precision_ckd * sensitivity) / (precision_ckd + sensitivity) if (precision_ckd + sensitivity) > 0 else 0
        
        precision_nockd = tn / (tn + fn) if (tn + fn) > 0 else 0
        recall_nockd = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1_nockd = 2 * (precision_nockd * recall_nockd) / (precision_nockd + recall_nockd) if (precision_nockd + recall_nockd) > 0 else 0
        
        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        
        summary_rows.append({
            "Model": name,
            "ROC_AUC": round(roc_auc, 4),
            "PR_AUC": round(pr_auc, 4),
            "Sensitivity (Recall CKD)": round(sensitivity, 4),
            "Specificity (Recall NoCKD)": round(specificity, 4),
            "Precision (CKD)": round(precision_ckd, 4),
            "F1_Score (CKD)": round(f1_ckd, 4),
            "Precision (NoCKD)": round(precision_nockd, 4),
            "F1_Score (NoCKD)": round(f1_nockd, 4),
            "Confusion_Matrix": cm
        })
        
        # Curves
        fpr, tpr, roc_thresh = roc_curve(y_test, y_prob)
        prec_curve, rec_curve, pr_thresh = precision_recall_curve(y_test, y_prob)
        curves[name] = {
            "fpr": fpr, "tpr": tpr,
            "precision": prec_curve, "recall": rec_curve
        }
        
        # Threshold sweeps
        thresholds = np.linspace(0.01, 0.99, 99)
        sweep_rows = []
        for t in thresholds:
            t_pred = (y_prob >= t).astype(int)
            p1, r1, f1_1, _ = precision_recall_fscore_support(y_test, t_pred, pos_label=1, average="binary", zero_division=0)
            p0, r0, f1_0, _ = precision_recall_fscore_support(y_test, t_pred, pos_label=0, average="binary", zero_division=0)
            sweep_rows.append({"Threshold": t, "Precision_CKD": p1, "Recall_CKD": r1, "F1_CKD": f1_1, "Precision_NoCKD": p0, "Recall_NoCKD": r0, "F1_NoCKD": f1_0})
        threshold_sweeps[name] = pd.DataFrame(sweep_rows)
        
    model_summary_df = pd.DataFrame(summary_rows)
    
    # 9. SHAP Explainability for Tuned XGBoost
    xgb_final_step = xgb_tuned.named_steps["xgb"]
    explainer = shap.TreeExplainer(xgb_final_step)
    shap_values = explainer.shap_values(X_test_scaled)
    
    feat_imp = pd.DataFrame({
        "Feature": final_features,
        "Gain_Importance": xgb_final_step.feature_importances_,
        "Mean_Abs_SHAP": np.abs(shap_values).mean(axis=0)
    }).sort_values("Gain_Importance", ascending=False)
    
    bundle = {
        "df_clean": df_clean,
        "categorical_cols": categorical_cols,
        "numerical_cols": numerical_cols,
        "final_features": final_features,
        "shapiro_df": shapiro_df,
        "chi2_df": chi2_df,
        "mw_df": mw_df,
        "X_train": X_train,
        "X_test": X_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "models": models,
        "probs": probs,
        "model_summary_df": model_summary_df,
        "curves": curves,
        "threshold_sweeps": threshold_sweeps,
        "explainer": explainer,
        "shap_values": shap_values,
        "feat_imp": feat_imp,
        "best_params": random_search.best_params_
    }
    
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(bundle, f)
        
    print("Pipeline completed and cached successfully!")
    return bundle

if __name__ == "__main__":
    run_or_load_pipeline(force_recompute=True)
