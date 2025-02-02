import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import cross_val_score


def train_logistic_regression(X_trainval, Y_trainval, X_test, Y_test):
    """
    Example: tune C parameter for LogisticRegression,
    then return the best model & its test score
    """
    best_score = 0
    best_c = None
    for c in [0.001, 0.1, 1, 10, 100]:
        model = LogisticRegression(C=c, max_iter=10000)
        scores = cross_val_score(
            model, X_trainval, Y_trainval, cv=5, scoring="accuracy"
        )
        score = np.mean(scores)
        if score > best_score:
            best_score = score
            best_c = c
    # Train final model
    final_model = LogisticRegression(C=best_c, max_iter=10000)
    final_model.fit(X_trainval, Y_trainval)
    test_score = final_model.score(X_test, Y_test)
    return final_model, best_c, best_score, test_score


def train_svm(X_trainval, Y_trainval, X_test, Y_test):
    best_score = 0
    best_params = (None, None, None)
    for c_param in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
        for gamma_param in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
            for kernel_param in ["rbf", "linear", "poly", "sigmoid"]:
                model = SVC(C=c_param, gamma=gamma_param, kernel=kernel_param)
                scores = cross_val_score(
                    model, X_trainval, Y_trainval, cv=5, scoring="accuracy"
                )
                score = np.mean(scores)
                if score > best_score:
                    best_score = score
                    best_params = (c_param, gamma_param, kernel_param)
    final_model = SVC(C=best_params[0], gamma=best_params[1], kernel=best_params[2])
    final_model.fit(X_trainval, Y_trainval)
    test_score = final_model.score(X_test, Y_test)
    return final_model, best_params, best_score, test_score


# Similarly for DecisionTree, RandomForest, AdaBoost
def train_decision_tree(X_trainval, Y_trainval, X_test, Y_test):
    best_score = 0
    best_depth = None
    for md in range(1, 9):
        model = DecisionTreeClassifier(max_depth=md, random_state=0)
        scores = cross_val_score(
            model, X_trainval, Y_trainval, cv=5, scoring="accuracy"
        )
        score = np.mean(scores)
        if score > best_score:
            best_score = score
            best_depth = md
    final_model = DecisionTreeClassifier(max_depth=best_depth, random_state=0)
    final_model.fit(X_trainval, Y_trainval)
    test_score = final_model.score(X_test, Y_test)
    return final_model, best_depth, best_score, test_score


def train_random_forest(X_trainval, Y_trainval, X_test, Y_test):
    best_score = 0
    best_params = (None, None, None)
    for M in range(2, 15, 2):
        for d in range(1, 9):
            for m in range(1, 9):
                model = RandomForestClassifier(
                    n_estimators=M, max_features=d, max_depth=m, random_state=0
                )
                scores = cross_val_score(
                    model, X_trainval, Y_trainval, cv=5, scoring="accuracy"
                )
                score = np.mean(scores)
                if score > best_score:
                    best_score = score
                    best_params = (M, d, m)
    final_model = RandomForestClassifier(
        n_estimators=best_params[0],
        max_features=best_params[1],
        max_depth=best_params[2],
        random_state=0,
    )
    final_model.fit(X_trainval, Y_trainval)
    test_score = final_model.score(X_test, Y_test)
    return final_model, best_params, best_score, test_score


def train_adaboost(X_trainval, Y_trainval, X_test, Y_test):
    best_score = 0
    best_params = (None, None)
    for M in range(2, 15, 2):
        for lr in [0.0001, 0.001, 0.01, 0.1, 1]:
            model = AdaBoostClassifier(n_estimators=M, learning_rate=lr, random_state=0)
            scores = cross_val_score(
                model, X_trainval, Y_trainval, cv=5, scoring="accuracy"
            )
            score = np.mean(scores)
            if score > best_score:
                best_score = score
                best_params = (M, lr)
    final_model = AdaBoostClassifier(
        n_estimators=best_params[0], learning_rate=best_params[1], random_state=0
    )
    final_model.fit(X_trainval, Y_trainval)
    test_score = final_model.score(X_test, Y_test)
    return final_model, best_params, best_score, test_score
