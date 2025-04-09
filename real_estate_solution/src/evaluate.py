from sklearn.metrics import r2_score, mean_squared_error

def evaluate_model(y_true, y_pred):
    return {
        "R2 Score": r2_score(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred)
    }
