from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from ipdb import set_trace
import pandas as pd
import numpy as np

# Import helper functions
from utils import (
    load_and_preprocess_data, normalize_features, check_stratification,
    train_and_evaluate, plot_bar_chart, plot_confusion_matrices,
    perform_grid_search, plot_grid_search_results, plot_validation_curve,
    validation_curve,load_and_preprocess_sorted_stratified
)

# --- Configuration ---
DATA_FILE = "ddos_dataset_first_version_clean.csv"
COLS_TO_EXCLUDE_SCALE = ['flow_id', 'source_ip', 'destination_ip', 'timestamp', 'simillarhttp']
COLS_TO_DROP_TRAIN = ['flow_id', 'source_ip', 'destination_ip', 'timestamp', 'simillarhttp']

# --- 1. Data Loading and Preprocessing ---
print("Loading and splitting data...")
X_train_raw, X_test_raw, y_train, y_test, full_df = load_and_preprocess_data(DATA_FILE)
X_train_raw_by_tm, X_test_raw_by_tm, y_train_by_tm, y_test_by_tm, full_df_by_tm\
    = load_and_preprocess_sorted_stratified(DATA_FILE,'timestamp')

# Normalize
print("Normalizing features...")
X_train_scaled, X_test_scaled = normalize_features(X_train_raw, X_test_raw, COLS_TO_EXCLUDE_SCALE)
X_train_scaled_by_tm, X_test_scaled_by_tm = normalize_features(X_train_raw_by_tm, X_test_raw_by_tm, COLS_TO_EXCLUDE_SCALE)

# Verify Stratification
check_stratification(y_train, y_test)
check_stratification(y_train_by_tm, y_test_by_tm)

# Drop Unused Columns for Training
X_train = X_train_scaled.drop(columns=COLS_TO_DROP_TRAIN)
X_test = X_test_scaled.drop(columns=COLS_TO_DROP_TRAIN)
X_train_by_tm = X_train_scaled_by_tm.drop(columns=COLS_TO_DROP_TRAIN)
X_test_by_tm = X_test_scaled_by_tm.drop(columns=COLS_TO_DROP_TRAIN)

# --- 2. Initial Model Screening ---
print("\nStarting Initial Model Screening...")
initial_models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_jobs=-1),
    'Decision Tree': DecisionTreeClassifier(),
    'Gaussian NB': GaussianNB()
}

initial_results = []
for name, model in initial_models.items():
    res = train_and_evaluate(model, X_train, y_train, X_test, y_test, name)
    initial_results.append(res)

# Visualization of Initial Results
scores_df = pd.DataFrame(initial_results)[['Model', 'Train_Acc', 'Test_Acc']]
scores_melted = scores_df.melt(id_vars="Model", var_name="Set", value_name="Accuracy")
plot_bar_chart(scores_melted, "Model", "Accuracy", "Initial Model Comparison", "Classifier", "Accuracy", hue="Set")

preds_list = [(res['Model'], res['Predictions']) for res in initial_results]
plot_confusion_matrices(preds_list, y_test)


# --- 3. Hyperparameter Tuning ---
print("\nStarting Hyperparameter Tuning...")

# A. Logistic Regression Tuning
print("\nTuning Logistic Regression...")
log_params = [
    {'C': [0.01, 0.1, 1, 10], 'penalty': ['l1', 'l2'], 'solver': ['saga']},
    {'C': [0.01, 0.1, 1, 10], 'penalty': ['l2'], 'solver': ['lbfgs']},
    {'penalty': [None], 'solver': ['lbfgs', 'saga']}
]
grid_log = perform_grid_search(LogisticRegression(max_iter=100), log_params, X_train, y_train)
plot_grid_search_results(grid_log) # Visualizing C parameter impact

# B. Random Forest Tuning
print("\nTuning Random Forest...")
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 10],
    'min_samples_leaf': [1, 4]
}
grid_rf = perform_grid_search(RandomForestClassifier(random_state=15), rf_params, X_train, y_train)
plot_grid_search_results(grid_rf)

# C. Decision Tree Tuning
print("\nTuning Decision Tree...")
dt_params = {
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 10],
    'min_samples_leaf': [1, 5],
    'criterion': ['gini', 'entropy'],
    'max_features': [None, 'sqrt']
}
grid_dt = perform_grid_search(DecisionTreeClassifier(random_state=15), dt_params, X_train, y_train)
plot_grid_search_results(grid_dt)

# D. Gaussian NB Tuning (Validation Curve)
print("\nTuning Gaussian NB...")
gnb_param_range = np.logspace(0, -15, 16)
train_scores, test_scores = validation_curve(
    GaussianNB(), X_train, y_train.values.ravel(),
    param_name='var_smoothing', param_range=gnb_param_range,
    cv=5, scoring='accuracy', n_jobs=-1
)
plot_validation_curve(train_scores, test_scores, gnb_param_range, 'var_smoothing', log_scale=True)
best_gnb_param = gnb_param_range[np.argmax(np.mean(test_scores, axis=1))]


# --- 4. Final Evaluation of Best Models ---
print("\nEvaluating Best Tuned Models...")
best_models = {
    'Best LogReg': LogisticRegression(max_iter=100, **grid_log.best_params_, n_jobs=-1),
    'Best RF': RandomForestClassifier(random_state=15, **grid_rf.best_params_, n_jobs=-1),
    'Best DTC': DecisionTreeClassifier(random_state=15, **grid_dt.best_params_),
    'Best GNB': GaussianNB(var_smoothing=best_gnb_param)
}

final_metrics = []
for name, model in best_models.items():
    res = train_and_evaluate(model, X_train, y_train, X_test, y_test, name)
    final_metrics.append(res)

# Visualization of Final Results
final_scores_df = pd.DataFrame(final_metrics)[['Model', 'Train_Acc', 'Test_Acc']]
final_melted = final_scores_df.melt(id_vars="Model", var_name="Set", value_name="Accuracy")
plot_bar_chart(final_melted, "Model", "Accuracy", "Final Tuned Model Comparison", "Classifier", "Accuracy", hue="Set")

# Compare Inference and Train Times
time_df = pd.DataFrame(final_metrics)[['Model', 'Infer_Time']]
plot_bar_chart(time_df, "Model", "Infer_Time", "Inference Time Comparison", "Classifier", "Time (s)")
time_df = pd.DataFrame(final_metrics)[['Model', 'Train_Time']]
plot_bar_chart(time_df, "Model", "Train_Time", "Inference Time Comparison", "Classifier", "Time (s)")

print("\nDone.")

# ==============================================================================
# --- 5. NEW: Evaluation on Sorted Stratified Split (Time-Based) ---
# ==============================================================================
print("\n" + "="*50)
print("   STARTING SORTED STRATIFIED SPLIT EVALUATION   ")
print("="*50)

# A. Evaluate BASE Models on Sorted Split
print("\n--- Base Models (Sorted/Time Split) ---")
# Re-instantiate base models to ensure they are fresh
models_sorted = {
    'LogReg (Sorted)': LogisticRegression(max_iter=1000, n_jobs=-1),
    'RF (Sorted)': RandomForestClassifier(random_state=15, n_jobs=-1),
    'DTC (Sorted)': DecisionTreeClassifier(random_state=15),
    'GNB (Sorted)': GaussianNB()
}

sorted_base_metrics = []

for name, model in models_sorted.items():
    # Note: We pass X_train_by_tm and y_train_by_tm here!
    res = train_and_evaluate(model, X_train_by_tm, y_train_by_tm, X_test_by_tm, y_test_by_tm, name)
    sorted_base_metrics.append(res)

# Visualize Base Sorted Results
sorted_base_df = pd.DataFrame(sorted_base_metrics)[['Model', 'Train_Acc', 'Test_Acc']]
sorted_base_melted = sorted_base_df.melt(id_vars="Model", var_name="Set", value_name="Accuracy")
plot_bar_chart(sorted_base_melted, "Model", "Accuracy", "Base Models (Sorted Stratified Split)", "Classifier", "Accuracy", hue="Set")


# B. Evaluate TUNED Models on Sorted Split
# We use the BEST params found in Step 3 (Random Split) but train on Step 5 (Sorted Split)
print("\n--- Tuned Models (Sorted/Time Split) ---")
print("Applying best hyperparameters found in Random Split to the Time-based Split...")

best_models_sorted = {
    'Best LogReg (Sorted)': LogisticRegression(max_iter=1000, **grid_log.best_params_, n_jobs=-1),
    'Best RF (Sorted)': RandomForestClassifier(random_state=15, **grid_rf.best_params_, n_jobs=-1),
    'Best DTC (Sorted)': DecisionTreeClassifier(random_state=15, **grid_dt.best_params_),
    'Best GNB (Sorted)': GaussianNB(var_smoothing=best_gnb_param)
}

sorted_tuned_metrics = []

for name, model in best_models_sorted.items():
    res = train_and_evaluate(model, X_train_by_tm, y_train_by_tm, X_test_by_tm, y_test_by_tm, name)
    sorted_tuned_metrics.append(res)

# Visualize Tuned Sorted Results
sorted_tuned_df = pd.DataFrame(sorted_tuned_metrics)[['Model', 'Train_Acc', 'Test_Acc']]
sorted_tuned_melted = sorted_tuned_df.melt(id_vars="Model", var_name="Set", value_name="Accuracy")
plot_bar_chart(sorted_tuned_melted, "Model", "Accuracy", "Tuned Models (Sorted Stratified Split)", "Classifier", "Accuracy", hue="Set")

print("\nProcessing Complete.")