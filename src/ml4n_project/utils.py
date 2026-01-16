import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, validation_curve
from sklearn.decomposition import PCA
import time
import os
import re
import json

# Setup Output Directory
OUTPUT_DIR = "plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _save_and_close(title, plot_type):
    """
    Generates filename from title + plot_type and saves the figure.
    Example: Title "Initial Results", Type "bar" -> "initial_results_bar.png"
    """
    # 1. Clean the title to make it filename-safe
    clean_title = title.lower()
    clean_title = clean_title.replace(" ", "_")
    clean_title = re.sub(r'[^\w\-_]', '', clean_title)

    # 2. Construct Filename
    filename = f"{clean_title}_{plot_type}.png"
    path = os.path.join(OUTPUT_DIR, filename)

    # 3. Save and Close
    plt.savefig(path, bbox_inches='tight')
    print(f"Saved plot: {path}")
    plt.close()


# --- Visualization Functions ---

def plot_bar_chart(df, x_col, y_col, title, xlabel, ylabel, hue=None, rotation=45):
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(data=df, x=x_col, y=y_col, hue=hue, palette="viridis")

    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3, fontsize=10)

    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    if rotation:
        plt.xticks(rotation=rotation, ha='right', fontsize=10)
    plt.legend(title=hue, loc='upper left', bbox_to_anchor=(1, 1)) if hue else None
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    # AUTO-SAVE with suffix "_bar"
    _save_and_close(title, "bar")


def plot_confusion_matrices(predictions_list, y_true, title="Confusion Matrices"):
    """
    Plots confusion matrices side-by-side.
    predictions_list: List of tuples ('ModelName', y_pred)
    """
    count = len(predictions_list)
    fig, axes = plt.subplots(1, count, figsize=(6 * count, 5))
    if count == 1: axes = [axes]

    # --- ADD MAIN TITLE ---
    fig.suptitle(title, fontsize=18, y=0.98)

    for ax, (model_name, y_pred) in zip(axes, predictions_list):
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,annot_kws={"size": 7})
        ax.set_title(model_name, fontsize=14)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')

    # --- PREVENT OVERLAP ---
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # AUTO-SAVE with suffix "_cm" (Confusion Matrix)
    _save_and_close(title, "cm")

def plot_grid_search_results(grid, title="Grid Search Results"):
    """
    Visualizes the impact of hyperparameters using boxplots.
    Accepts the fitted GridSearchCV object.
    """
    results_df = pd.DataFrame(grid.cv_results_)

    # Identify parameter columns
    param_cols = [col for col in results_df.columns if col.startswith('param_')]
    score_col = 'mean_test_score'

    # Setup Figure
    fig, axes = plt.subplots(1, len(param_cols), figsize=(5 * len(param_cols), 5), sharey=True)
    if len(param_cols) == 1: axes = [axes]

    # --- ADD MAIN TITLE ---
    # y=0.97 moves the title slightly up so it's clearly distinct
    fig.suptitle(title, fontsize=18, y=0.97)

    for i, param in enumerate(param_cols):
        # Clean column name for display (remove 'param_')
        clean_name = param.replace("param_", "")

        # We fill NaNs with "None" string for better plotting categories
        plot_data = results_df.copy()
        plot_data[param] = plot_data[param].fillna("None")

        sns.boxplot(x=param, y=score_col, data=plot_data, ax=axes[i], palette="Set2")

        # Individual Subplot Title
        axes[i].set_title(f"Impact of {clean_name}")
        axes[i].set_xlabel(clean_name)
        axes[i].set_ylabel("Accuracy" if i == 0 else "")

    # --- PREVENT OVERLAP ---
    # rect=[left, bottom, right, top]
    # We leave 5% space at the top (top=0.95) for the suptitle
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # AUTO-SAVE with suffix "_impact"
    _save_and_close(title, "impact")

def plot_two_param_grid_scan(grid_or_results, param_grid, title="2D Grid Scan",
                              cmap="viridis", fmt=".4f"):
    """
    Plot a 2D heatmap of mean_test_score for exactly two varying parameters.
    - `param_grid` is the dictionary used for GridSearch (may contain >2 params).
    - Params with a single value are treated as fixed and included in the title.
    - Raises ValueError unless exactly two parameters have more than one value.
    Returns the pivoted DataFrame (index=param_y, columns=param_x).
    """
    # Validate param_grid
    if not isinstance(param_grid, dict):
        raise TypeError("`param_grid` must be a dict of parameter -> list_of_values")

    # Split into varying (>1) and fixed (==1) params
    varying = {}
    fixed = {}
    for k, v in param_grid.items():
        ln = len(v)
        if ln > 1:
            varying[k] = v
        elif ln == 1:
            fixed[k] = v[0]

    if len(varying) != 2:
        raise ValueError(f"Expected exactly 2 varying parameters; found {len(varying)}. Varying: {list(varying.keys())}")

    # Determine parameter names for x and y
    param_x, param_y = list(varying.keys())
    results_df = pd.DataFrame(grid_or_results.cv_results_)

    px = param_x if param_x.startswith("param_") else f"param_{param_x}"
    py = param_y if param_y.startswith("param_") else f"param_{param_y}"

    # Build title with fixed params listed
    if fixed:
        fixed_str = ", ".join(f"{k}={str(v)}" for k, v in fixed.items())
        title_full = f"{title} ({fixed_str})"
    else:
        title_full = title

    # Prepare plotting DataFrame (fill NaNs for categorical display)
    plot_df = results_df.copy()
    plot_df[px] = plot_df[px].fillna("None")
    plot_df[py] = plot_df[py].fillna("None")

    pivot = plot_df.pivot_table(index=py, columns=px, values="mean_test_score")

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(max(6, pivot.shape[1] * 1.2), max(5, pivot.shape[0] * 0.8)))
    sns.heatmap(pivot, annot=True, fmt=fmt, cmap=cmap, ax=ax, cbar_kws={"label": "Mean Test Score"},
                annot_kws={"size": 8})
    ax.set_xlabel(param_x.replace("param_", ""))
    ax.set_ylabel(param_y.replace("param_", ""))
    fig.suptitle(title_full, fontsize=16, y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Auto-save using existing helper (suffix "grid2d")
    _save_and_close(title_full, "grid2d")

    return pivot



def plot_validation_curve(train_scores, test_scores, param_range, param_name, title, log_scale=False):
    train_mean = np.mean(train_scores, axis=1)
    train_min = np.min(train_scores, axis=1)
    train_max = np.max(train_scores, axis=1)
    val_mean = np.mean(test_scores, axis=1)
    val_min = np.min(test_scores, axis=1)
    val_max = np.max(test_scores, axis=1)

    plt.figure()
    plt.plot(param_range, train_mean, label="train", color="tab:blue")
    plt.fill_between(param_range, train_min, train_max, alpha=0.5, color='tab:blue')
    plt.plot(param_range, val_mean, label="val", color="tab:red")
    plt.fill_between(param_range, val_min, val_max, alpha=0.5, color='tab:red')

    if log_scale:
        plt.xscale('log')

    plt.xlabel(param_name)
    plt.ylabel("Accuracy")
    plt_title = f"Validation Curve for {param_name}"
    plt.title(plt_title)
    plt.legend()

    file_title = f"{title}_{param_name}"
    # AUTO-SAVE with suffix "_val_curve"
    _save_and_close(file_title, "val_curve")


# --- Data Processing Functions ---

def cols_to_drop_by_corr(X_train, y_train, exclude_cols, threshold=0.90):
    """ remove highly correlated columns from X_train based on correlation threshold"""
    y = y_train.squeeze()

    Xn = X_train.drop(columns=exclude_cols)

    corr = Xn.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    # Build table
    pairs = upper.stack().reset_index()
    pairs.columns = ["A", "B", "corr_abs"]

    pairs = pairs[pairs["corr_abs"] > threshold]
    pairs = pairs.sort_values("corr_abs", ascending=False)
    pairs = pairs.reset_index(drop=True)

    target_score = {}

    for c in Xn.columns:
        s = Xn[c]
        v = s.corr(y)
        target_score[c] = float(abs(v))

    to_drop = []

    A_list = pairs["A"].tolist()
    B_list = pairs["B"].tolist()

    for i in range(len(pairs)):
        a = A_list[i]
        b = B_list[i]

        if (a not in to_drop) and (b not in to_drop):
            if target_score[a] < target_score[b]:
                drop = a
            else:
                drop = b

            to_drop.append(drop)

    X_train = X_train.drop(columns=to_drop)

    return X_train, to_drop


# Applies PCA to reduce dimensionality of X_train and X_test

def cols_to_drop_by_PCA(X_train, X_test, exclude_cols):
    print("Shape X_train before PCA:", X_train.shape)
    print("Shape X_test before PCA:", X_test.shape)

    train_to_re_add_saved = X_train[exclude_cols].copy()
    test_to_re_add_saved = X_test[exclude_cols].copy()

    X_train = X_train.drop(columns=(exclude_cols))
    X_test = X_test.drop(columns=(exclude_cols))

    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    pc_cols = [f"PC{i + 1}" for i in range(X_train_pca.shape[1])]
    X_train_pca_df = pd.DataFrame(X_train_pca, columns=pc_cols, index=X_train.index)
    X_test_pca_df = pd.DataFrame(X_test_pca, columns=pc_cols, index=X_test.index)

    X_train_final = pd.concat([X_train_pca_df, train_to_re_add_saved], axis=1)
    X_test_final = pd.concat([X_test_pca_df, test_to_re_add_saved], axis=1)

    print("Shape X_train final:", X_train_final.shape)
    print("Shape X_test final:", X_test_final.shape)

    return X_train_final, X_test_final

def load_and_preprocess_data(filepath, target_col='label', train_size=0.7, random_state=15):
    df = pd.read_csv(filepath, low_memory=False)
    X = df.drop(columns=[target_col])
    y = df[[target_col]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, train_size=train_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test, df  # Return df for EDA

def load_and_preprocess_sorted_stratified(filepath, sort_by, target_col='label', train_size=0.7):
    # 1. Load Data
    df = pd.read_csv(filepath, low_memory=False)

    # 2. Define a helper function to split a single group
    def split_group(group):
        # Sort this specific label group by the sort column (e.g., Timestamp)
        group_sorted = group.sort_values(by=sort_by)

        # Calculate split point for this specific label
        split_idx = int(len(group_sorted) * train_size)

        # Split into train and test
        train_part = group_sorted.iloc[:split_idx]
        test_part = group_sorted.iloc[split_idx:]

        return train_part, test_part

    # 3. Apply the split to each label group
    train_pieces = []
    test_pieces = []

    # Get unique labels
    unique_labels = df[target_col].unique()

    for label in unique_labels:
        # Get only the rows for this label
        group = df[df[target_col] == label]

        # Split this group
        tr, te = split_group(group)

        # Add to our lists
        train_pieces.append(tr)
        test_pieces.append(te)

    # 4. Concatenate all the pieces back together
    train_sorted = pd.concat(train_pieces)
    test_sorted = pd.concat(test_pieces)

    # 5. Separate Features (X) and Target (y)
    X_train, y_train = train_sorted.drop(columns=[target_col]), train_sorted[[target_col]]
    X_test, y_test = test_sorted.drop(columns=[target_col]), test_sorted[[target_col]]

    print(f"Total Train: {len(X_train)} | Total Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test, df



def normalize_features(X_train, X_test, cols_to_exclude):
    scaler = StandardScaler()
    all_features = X_train.columns.tolist()
    cols_to_scale = [col for col in all_features if col not in cols_to_exclude]

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test_scaled[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

    return X_train_scaled, X_test_scaled


def check_stratification(y_train, y_test, threshold=0.05):
    train_dist = y_train.value_counts(normalize=True).sort_index()
    test_dist = y_test.value_counts(normalize=True).sort_index()

    comparison = pd.DataFrame({'Train': train_dist, 'Test': test_dist})
    comparison['Diff'] = (comparison['Train'] - comparison['Test']).abs()

    print("\nStratification Check:")
    print(comparison.style.format("{:.2%}"))

    bad_splits = comparison[comparison['Diff'] > threshold]
    if not bad_splits.empty:
        print(f"\n⚠️ WARNING: Poor stratification (> {threshold:.0%} diff) in: {bad_splits.index.tolist()}")
    else:
        print("\n✅ Stratification looks good!")


# --- Model Evaluation Functions ---

def train_and_evaluate(model, X_train, y_train, X_test, y_test, model_name="Model"):
    start_train = time.time()
    model.fit(X_train, y_train.values.ravel())
    train_time = time.time() - start_train

    start_infer = time.time()
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    infer_time = time.time() - start_infer

    # Handle float predictions from regression-like models if labels are ints
    if y_test_pred.dtype == float:
        y_train_pred = y_train_pred.astype(int)
        y_test_pred = y_test_pred.astype(int)

    print(f"\n--- {model_name} ---")
    print("Training Report:")
    print(classification_report(y_train, y_train_pred))
    print("Test Report:")
    print(classification_report(y_test, y_test_pred))

    return {
        'Model': model_name,
        'Train_Acc': accuracy_score(y_train, y_train_pred),
        'Test_Acc': accuracy_score(y_test, y_test_pred),
        'Train_Time': train_time,
        'Infer_Time': infer_time,
        'Predictions': y_test_pred
    }


def perform_grid_search(estimator, param_grid, X_train, y_train, cv=2, verbose=0):
    grid = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        return_train_score=False,
        verbose=verbose,
        n_jobs=-1
    )
    grid.fit(X_train, y_train.values.ravel())

    print("-" * 30)
    print(f"Best Accuracy: {grid.best_score_:.4f}")
    print(f"Best Parameters: {grid.best_params_}")
    print("-" * 30)

    return grid


def evaluate_and_plot_models(models_dict, X_train, y_train, X_test, y_test, title):
    """
    Iterates through a dictionary of models, trains/evaluates them,
    and automatically plots the Accuracy Bar Chart and Confusion Matrices.
    Returns the list of results dictionaries for further processing (e.g. time plotting).
    """
    results = []

    # 1. Train and Evaluate Loop
    for name, model in models_dict.items():
        # Uses the existing train_and_evaluate function
        res = train_and_evaluate(model, X_train, y_train, X_test, y_test, name)
        results.append(res)

    # 2. Plot Bar Chart (Accuracy)
    scores_df = pd.DataFrame(results)[['Model', 'Train_Acc', 'Test_Acc']]
    scores_melted = scores_df.melt(id_vars="Model", var_name="Set", value_name="Accuracy")
    plot_bar_chart(scores_melted, "Model", "Accuracy", title, "Classifier", "Accuracy", hue="Set")

    # 3. Plot Confusion Matrices
    preds_list = [(res['Model'], res['Predictions']) for res in results]
    plot_confusion_matrices(preds_list, y_test, title=title)

    return results

def tune_hyperparameters(models_params, model_classes, X_train, y_train, impact_tune, file_suffix=""):
    """
    Performs hyperparameter tuning for all models defined in models_params.
    Handles both GridSearchCV (RF, LogReg, DT) and ValidationCurve (GNB).
    Saves the best parameters to a JSON file.
    """
    print("\nStarting Hyperparameter Tuning...")

    tuned_params = {}
    for model_name, settings in models_params.items():
        if "grid_params" not in settings or "grid_2d_params" not in settings:
            continue
        init_params = settings.get("initial_params", {})
        model_instance = model_classes[model_name](**init_params)
        if impact_tune:
            print(f"\nTuning {model_name}...")
            grid_params = settings["grid_params"]
            grid_search = perform_grid_search(model_instance, grid_params, X_train, y_train)
            plot_grid_search_results(grid_search, title=f'{model_name} Params Impact ({file_suffix})')
        else:
            print(f"\nTuning {model_name} (2D Grid Search)...")
            grid_2d_params = settings["grid_2d_params"]
            grid_search = perform_grid_search(model_instance, grid_2d_params, X_train, y_train)
            plot_two_param_grid_scan(grid_search, grid_2d_params, title=f'{model_name} 2D Grid Search ({file_suffix})')

        tuned_params[model_name] = grid_search.best_params_
        print(f"Best {model_name} Params: {grid_search.best_params_}")


    # Gaussian NB Tuning (Validation Curve)
    print("\nTuning Gaussian NB...")
    gs = models_params["GaussianNB"]["validation_curve_params"]
    gnb_param_range = (np.logspace if gs.get("log_scale") else np.linspace)(gs["param_range_start"], gs["param_range_end"], gs["param_range_num"])
    train_scores, test_scores = validation_curve(
        model_classes["GaussianNB"](), X_train, y_train.values.ravel(),
        param_name='var_smoothing', param_range=gnb_param_range,
        cv=5, scoring='accuracy', n_jobs=-1
    )
    plot_validation_curve(train_scores, test_scores, gnb_param_range, 'var_smoothing', file_suffix, log_scale=True)
    best_gnb_param = gnb_param_range[np.argmax(np.mean(test_scores, axis=1))]

    tuned_params["GaussianNB"] = {gs["param_name"]: best_gnb_param}


    params_file_name = "tuning_impact_params" if impact_tune else "tuning_2d_grid_params"
    with open(f"plots/{params_file_name}_{file_suffix}.json", "w") as file:
        json.dump(tuned_params, file, indent=4)

    return tuned_params

# --- Helper Function to Load Saved Params ---
def load_params_from_file(impact_tune, suffix):
    """
    Constructs the filename based on the tuning mode and suffix,
    then loads the JSON.
    """
    file_prefix = "tuning_impact_params" if impact_tune else "tuning_2d_grid_params"
    filename = f"plots/{file_prefix}_{suffix}.json"

    print(f"Skipping Training... Loading parameters from: {filename}")
    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"Could not find parameter file: {filename}. Please run with 'skip_tuning': false first.")

    with open(filename, 'r') as f:
        return json.load(f)

