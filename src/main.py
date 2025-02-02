import pandas as pd
from eda import run_eda
from preprocessing import preprocess_data
from modeling import (
    train_logistic_regression,
    train_svm,
    train_decision_tree,
    train_random_forest,
    train_adaboost,
)
from evaluation import evaluate_model


def main():
    # 1. (Optional) Run EDA
    print("Running EDA...")
    df_eda = run_eda(data_path="../data/oasis_longitudinal.csv")

    # 2. Preprocess data
    print("Preprocessing data (with imputation=True)...")
    X_trainval, X_test, Y_trainval, Y_test, scaler = preprocess_data(
        df_eda, impute=True
    )

    # 3. Train each model
    print("Training Logistic Regression...")
    lr_model, lr_c, lr_cv_score, lr_test_score = train_logistic_regression(
        X_trainval, Y_trainval, X_test, Y_test
    )

    print("Training SVM...")
    svm_model, svm_params, svm_cv_score, svm_test_score = train_svm(
        X_trainval, Y_trainval, X_test, Y_test
    )

    print("Training Decision Tree...")
    dt_model, dt_depth, dt_cv_score, dt_test_score = train_decision_tree(
        X_trainval, Y_trainval, X_test, Y_test
    )

    print("Training Random Forest...")
    rf_model, rf_params, rf_cv_score, rf_test_score = train_random_forest(
        X_trainval, Y_trainval, X_test, Y_test
    )

    print("Training AdaBoost...")
    ab_model, ab_params, ab_cv_score, ab_test_score = train_adaboost(
        X_trainval, Y_trainval, X_test, Y_test
    )

    # 4. Evaluate each model
    print("Evaluating models...")
    lr_eval = evaluate_model(lr_model, X_test, Y_test, model_name="Logistic Regression")
    svm_eval = evaluate_model(svm_model, X_test, Y_test, model_name="SVM")
    dt_eval = evaluate_model(dt_model, X_test, Y_test, model_name="Decision Tree")
    rf_eval = evaluate_model(rf_model, X_test, Y_test, model_name="Random Forest")
    ab_eval = evaluate_model(ab_model, X_test, Y_test, model_name="AdaBoost")

    # 5. Print summary
    results = [lr_eval, svm_eval, dt_eval, rf_eval, ab_eval]
    print("\n=== MODEL COMPARISON ===")
    for r in results:
        print(
            f"Model: {r['model_name']}, Accuracy: {r['accuracy']:.3f}, Recall: {r['recall']:.3f}, AUC: {r['auc']:.3f}"
        )
    print("Done.")


if __name__ == "__main__":
    main()
