from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

# Import helper functions
from utils import *
#TODO maybe clean up stuff later


#Load Configuration
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


# Data Loading and Preprocessing
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

# Initial Model Screening
print("\nStarting Initial Model Screening...")
initial_models = {}
for model_name, settings in config['models'].items():
    model_class = MODEL_CLASSES[model_name]
    params = settings.get('initial_params', {})
    initial_models[model_name] = model_class(**params)

initial_results = evaluate_and_plot_models(
    initial_models, X_train, y_train, X_test, y_test,
    title="Base Models (Stratified)"
)

# Hyperparameter Tuning

# in case of Impact tuning, exit after plotting

if SKIP_TUNING:
    tuned_params_stratified = load_params_from_file(IMPACT_TUNE, "stratified_split")
else:
    tuned_params_stratified = tune_hyperparameters(
        models_params, MODEL_CLASSES, X_train, y_train,
        IMPACT_TUNE,file_suffix="stratified_split"
    )



if not IMPACT_TUNE:
    # Final Evaluation of Best Models
    print("\nEvaluating Best Tuned Models...")
    best_models = {
        'Best LogReg': LogisticRegression(max_iter=MAX_ITER, **tuned_params_stratified["LogisticRegression"], n_jobs=-1),
        'Best RF': RandomForestClassifier(random_state=15,**tuned_params_stratified["RandomForestClassifier"] , n_jobs=-1),
        'Best DTC': DecisionTreeClassifier(random_state=15, **tuned_params_stratified["DecisionTreeClassifier"]),
        'Best GNB': GaussianNB(**tuned_params_stratified["GaussianNB"])
    }

    final_metrics = evaluate_and_plot_models(
        best_models, X_train, y_train, X_test, y_test,
        title="Tuned Model stratified_split"
    )

    # Compare Inference and Train Times
    time_df = pd.DataFrame(final_metrics)[['Model', 'Infer_Time']]
    plot_bar_chart(time_df, "Model", "Infer_Time", "Inference Time Comparison", "Classifier", "Time (s)")
    time_df = pd.DataFrame(final_metrics)[['Model', 'Train_Time']]
    plot_bar_chart(time_df, "Model", "Train_Time", "Train Time Comparison", "Classifier", "Time (s)")

    print("\nDone.")

# Evaluation on Sorted Stratified Split (Time-Based)
print("\n" + "="*50)
print("   STARTING SORTED STRATIFIED SPLIT EVALUATION   ")
print("="*50)

# Evaluate BASE Models on Sorted Split
print("\n--- Base Models (Sorted/Time Split) ---")
# Re-instantiate base models to ensure they are fresh
models_sorted = {
    'LogReg (Sorted)': LogisticRegression(max_iter=MAX_ITER, n_jobs=-1),
    'RF (Sorted)': RandomForestClassifier(random_state=15, n_jobs=-1),
    'DTC (Sorted)': DecisionTreeClassifier(random_state=15),
    'GNB (Sorted)': GaussianNB()
}

evaluate_and_plot_models(
    models_sorted, X_train_by_tm, y_train_by_tm, X_test_by_tm, y_test_by_tm,
    title="Base Models (Stratified Time)"
)

if SKIP_TUNING:
    tuned_params_sorted_time = load_params_from_file(IMPACT_TUNE, "sorted_time_split")
else:
    tuned_params_sorted_time = tune_hyperparameters(
        models_params, MODEL_CLASSES, X_train_by_tm, y_train_by_tm,
        IMPACT_TUNE, file_suffix="sorted_time_split"
    )

# Evaluate TUNED Models on Sorted Split
if not IMPACT_TUNE:
    print("\n--- Tuned Models (Sorted/Time Split) ---")
    print("Applying best hyperparameters found in Random Split to the Time-based Split...")

    best_models_sorted = {
        'Best LogReg (Sorted)': LogisticRegression(max_iter=MAX_ITER, **tuned_params_sorted_time["LogisticRegression"], n_jobs=-1),
        'Best RF (Sorted)': RandomForestClassifier(random_state=15,**tuned_params_sorted_time["RandomForestClassifier"] , n_jobs=-1),
        'Best DTC (Sorted)': DecisionTreeClassifier(random_state=15, **tuned_params_sorted_time["DecisionTreeClassifier"]),
        'Best GNB (Sorted)': GaussianNB(**tuned_params_sorted_time["GaussianNB"])
    }

    evaluate_and_plot_models(
        best_models_sorted, X_train_by_tm, y_train_by_tm, X_test_by_tm, y_test_by_tm,
        title="Tuned Model temporal stratified split"
    )



print("\nProcessing Complete.")