import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, validation_curve
import time


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
    plt.show()


def plot_confusion_matrices(predictions_list, y_true):
    num_models = len(predictions_list)
    fig, axes = plt.subplots(1, num_models, figsize=(6 * num_models, 6))
    if num_models == 1: axes = [axes]  # Handle single model case

    for ax, (model_name, y_pred) in zip(axes, predictions_list):
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,annot_kws={"size": 6})
        ax.set_title(model_name, fontsize=14)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')

    plt.tight_layout()
    fig.tight_layout()
    plt.show()


def plot_grid_search_results(grid, param_name, score_col='mean_test_score', log_scale=False):
    results_df = pd.DataFrame(grid.cv_results_)

    # Fill NaNs for plotting
    param_cols = [col for col in results_df.columns if col.startswith('param_')]
    for col in param_cols:
        results_df[col] = results_df[col].fillna('None')

    # Determine plot type based on parameter count
    # Simple case: Plot boxplots for each parameter
    fig, axes = plt.subplots(1, len(param_cols), figsize=(5 * len(param_cols), 5), sharey=True)
    if len(param_cols) == 1: axes = [axes]

    for i, param in enumerate(param_cols):
        sns.boxplot(x=param, y=score_col, data=results_df, ax=axes[i])
        axes[i].set_title(f"Impact of {param}")
        axes[i].set_xlabel(param)
        if i > 0: axes[i].set_ylabel("")

    plt.tight_layout()
    plt.show()


def plot_validation_curve(train_scores, test_scores, param_range, param_name, log_scale=False):
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
    plt.title(f"Validation Curve for {param_name}")
    plt.legend()
    plt.show()


# --- Data Processing Functions ---

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


def perform_grid_search(estimator, param_grid, X_train, y_train, cv=2, verbose=3):
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