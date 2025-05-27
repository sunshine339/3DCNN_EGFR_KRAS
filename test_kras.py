import os
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, 
    recall_score, f1_score, confusion_matrix, classification_report
)

def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在")
    return joblib.load(model_path)

def load_features(features_csv, sample_size=None):
    df = pd.read_csv(features_csv)
    required_cols = ["Case ID"] + [c for c in df.columns if c.startswith("Feature_")]
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"特征文件中缺少列: {missing_cols}")

    if sample_size is not None and sample_size > 0:
        if sample_size > len(df):
            raise ValueError(f"采样数量 {sample_size} 超过数据总数 {len(df)}")
        df = df.sample(n=sample_size)  
    return df

def predict(model, X, ids=None):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]  
    results = pd.DataFrame({
        "Case ID": ids if ids is not None else np.arange(len(X)),
        "Predicted_Label": y_pred,
        "Probability_Mutant": y_prob,
        "Predicted_Class": ["Mutant" if p == 1 else "Wildtype" for p in y_pred]
    })
    return results

def evaluate(y_true, y_pred, y_prob):
    if y_true is None:
        return
    metrics = {
        "ROC-AUC": roc_auc_score(y_true, y_prob),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred)
    }
    print("\n[评估指标]")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

def main():
    model_path = r"\medicen\lgbm_best_model_kras.pkl"
    features_csv = r"\medicen\final_features_kras.csv"  
    output_csv = r"\medicen\predictions_kras_test.csv"
    model = load_model(model_path)
    df = load_features(features_csv, sample_size=20)
    feature_cols = [c for c in df.columns if c.startswith("Feature_")]
    X = df[feature_cols].values
    ids = df["Case ID"].values
    results = predict(model, X, ids)
    if "KRAS mutation status" in df.columns:
        df["label"] = df["KRAS mutation status"].map({"Wildtype": 0, "Mutant": 1})
        evaluate(df["label"].values, results["Predicted_Label"].values, results["Probability_Mutant"].values)
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    results.to_csv(output_csv, index=False)
    print(f"\n[完成] 预测结果已保存至: {output_csv}")

if __name__ == "__main__":
    main()