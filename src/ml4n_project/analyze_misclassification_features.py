from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

# Import helper functions
from utils import *

def analyze_misclassification_features(model, X_test, y_test, true_class, pred_class):
    """
    Analyzes why 'true_class' is being misclassified as 'pred_class'.
    Compares the feature values of the misclassified samples vs the correct samples.
    """
    print(f"\n--- Analyzing Misclassification: True {true_class} vs. Pred {pred_class} ---")

    # 1. Generate Predictions internally
    y_pred = model.predict(X_test)

    # 2. Identify indices
    # Group A: The "Target" group we want to mimic (Correctly classified as Class Y)
    # logic: True Label is Y AND Predicted is Y
    correct_indices = (y_test.values.ravel() == pred_class) & (y_pred == pred_class)

    # Group B: The "Error" group (True Label is X, but Model thinks it is Y)
    error_indices = (y_test.values.ravel() == true_class) & (y_pred == pred_class)

    # 3. Extract Data
    if not any(correct_indices) or not any(error_indices):
        print("Not enough samples found for this specific error pattern.")
        return

    df_correct = X_test[correct_indices]
    df_error = X_test[error_indices]

    print(f"Comparing {len(df_error)} Misclassified Samples (True {true_class} -> Pred {pred_class})")
    print(f"Against {len(df_correct)} Correct Samples (True {pred_class} -> Pred {pred_class})")

    # 4. Compare Means
    # We look at the difference between the averages of the two groups.
    # Small difference = Model sees them as identical (Feature overlap)
    # Large difference = The feature is different, but the model ignored it (or it wasn't enough)
    diffs = (df_correct.mean() - df_error.mean()).abs().sort_values()

    print("\nTop 5 Features that are statistically IDENTICAL (Diff ~ 0):")
    # These are likely the culprits. If the features are identical, the model *cannot* distinguish them.
    print(diffs.head(5))
    print(diffs[diffs==0].index)

    print("\nTop 5 Features with LARGEST differences:")
    # These are features where the classes actually differ, but the model failed to use them correctly.
    print(diffs.tail(5))

# --- Load Configuration ---
with open('params.json', 'r') as f:
    config = json.load(f)

# Extract Global Settings
DATA_FILE = config['data_file']
COLS_TO_EXCLUDE_SCALE = config['cols_to_exclude_scale']
COLS_TO_DROP_TRAIN = config['cols_to_drop_train']
MAX_ITER = config['max_iter']
RANDOM_STATE = config['random_state']
CV_FOLDS = config['cv_folds']
IMPACT_TUNE = config['impact_tune']
SKIP_TUNING = config['skip_tuning']
models_params = config['models']

# Helper to map string names to Classes
MODEL_CLASSES = {
    'LogisticRegression': LogisticRegression,
    'RandomForestClassifier': RandomForestClassifier,
    'DecisionTreeClassifier': DecisionTreeClassifier,
    'GaussianNB': GaussianNB
}


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


#TODO maybe make it the same style as pca

# removing highly correlated features

X_train_scaled,c_t_d=cols_to_drop_by_corr(X_train_scaled, y_train, COLS_TO_EXCLUDE_SCALE,threshold=0.90)

X_test_scaled.drop(columns=c_t_d,inplace=True)
X_train_scaled_by_tm,c_t_d_2=cols_to_drop_by_corr(X_train_scaled_by_tm,
                                                  y_train_by_tm,
                                                  COLS_TO_EXCLUDE_SCALE,
                                                  threshold=0.90)
X_test_scaled_by_tm.drop(columns=c_t_d_2,inplace=True)

# remove features using PCA
X_train_scaled,X_test_scaled=cols_to_drop_by_PCA(X_train_scaled, X_test_scaled, COLS_TO_EXCLUDE_SCALE)

X_train_scaled_by_tm,X_test_scaled_by_tm=cols_to_drop_by_PCA(X_train_scaled_by_tm, X_test_scaled_by_tm, COLS_TO_EXCLUDE_SCALE)

# Drop Unused Columns for Training
X_train = X_train_scaled.drop(columns=COLS_TO_DROP_TRAIN)
X_test = X_test_scaled.drop(columns=COLS_TO_DROP_TRAIN)
X_train_by_tm = X_train_scaled_by_tm.drop(columns=COLS_TO_DROP_TRAIN)
X_test_by_tm = X_test_scaled_by_tm.drop(columns=COLS_TO_DROP_TRAIN)

tuned_params_stratified = load_params_from_file(IMPACT_TUNE, "stratified_split")

print("\nEvaluating Best Tuned Models...")
best_models = {
    'Best LogReg': LogisticRegression(max_iter=MAX_ITER, **tuned_params_stratified["LogisticRegression"], n_jobs=-1),
    'Best RF': RandomForestClassifier(random_state=15, **tuned_params_stratified["RandomForestClassifier"], n_jobs=-1),
    'Best DTC': DecisionTreeClassifier(random_state=15, **tuned_params_stratified["DecisionTreeClassifier"]),
    'Best GNB': GaussianNB(**tuned_params_stratified["GaussianNB"])
}

final_metrics = evaluate_and_plot_models(
    best_models, X_train, y_train, X_test, y_test,
    title="Tuned Model stratified_split"
)

# Analyze the specific Class 3 -> Class 2 confusion in Random Forest
print("\nInvoking Misclassification Analysis...")
analyze_misclassification_features(
    best_models['Best DTC'],
    X_test,
    y_test,
    true_class=3,
    pred_class=2
)
