import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from tqdm import tqdm

from pprint import pprint

from sklearn.model_selection import ParameterGrid
from sklearn_extra.cluster import KMedoids 
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, rand_score, silhouette_score
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from typing import Literal, Protocol, Any, Dict, Tuple
from enum import Enum
from joblib import Parallel, delayed
from pathlib import Path
import os


project_root = Path(__file__).resolve().parents[2]
SAVE_PATH = project_root / "res" / "k_evaluation"
PARAMS_FILE = project_root / "res" / "clustering_params.json" # Define params file path


# Available algorithms and metrics kept as enum
class AlgorithmName(Enum):
    KMEANS = "kmeans"
    KMEDOIDS = "kmedoids"
    GMM = "gmm"
    DBSCAN = "dbscan"
    SPECTRAL = "spectral"

class MetricName(Enum):
    SILHOUETTE = "silhouette"
    INERTIA = "inertia"
    RI = "ri"
    ARI = "ari"
    ALL = "all"

# dummy preprocessing function
def preprocess(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | pd.DataFrame | None]:
    df = data.copy()
    y = None
    if "label" in df.columns:
        y = df["label"]
        df = df.drop(columns=["label"])
    non_numeric = df.select_dtypes(include=["object", "category"]).columns.tolist()
    df = df.drop(columns=non_numeric)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(axis=1, how="all") 
    df = df.fillna(df.median(numeric_only=True))

    return df, y


# Nice wrapper for typing
class ClusterEstimator(Protocol):
    def fit(self, X: np.ndarray, y: Any = None) -> Any:
        ...
    def predict(self, X: np.ndarray) -> np.ndarray | tuple:
        ...
    def fit_predict(self, X: np.ndarray, y: Any = None) -> np.ndarray:
        ...

# K Cluster Model 
class ClusteringModel:
    def __init__(self, algorithm: AlgorithmName, dataset: pd.DataFrame) -> None:
        self.algorithm = algorithm

        self.data: pd.DataFrame = dataset.copy()
        self.x, self.y = preprocess(dataset)
        self.x = self.x.to_numpy()
        if not self.y is None:
            self.y = self.y.to_numpy()
        self.model_params: dict[str, Any] = {}   
        print(self.x.shape)
        self.model_: ClusterEstimator | None = None

    ### HELPERS ###

    def _make_model(self, n_clusters: int | None = None, **kwargs: Any) -> ClusterEstimator:
        """Instantiates new clustering model."""
        params = {**self.model_params, **kwargs}

        match self.algorithm:
            case AlgorithmName.KMEANS:
                params.setdefault("n_clusters", n_clusters)
                return KMeans(**params)

            case AlgorithmName.KMEDOIDS:
                params.setdefault("n_clusters", n_clusters)
                return KMedoids(**params)

            case AlgorithmName.GMM:
                params.setdefault("n_components", n_clusters)
                return GaussianMixture(**params)
            case AlgorithmName.DBSCAN:
                return DBSCAN(**params)

            case AlgorithmName.SPECTRAL:
                params.setdefault("n_clusters", n_clusters)
                return SpectralClustering(**params)

            case _:
                raise ValueError(f"Unknown algorithm: {self.algorithm}\nPlease instead use one of : ['kmeans', 'kmedoids','gmm']")

    def _evaluate_single_k(
        self,
        k: int,
        selected_metrics: list[MetricName]
    ) -> tuple[int, Dict[MetricName, float]]:
        """Compute clustering metrics for one value of k."""

        print(f"[evaluate K] Started job for {k}")

        model = self._make_model(k)
        labels = model.fit_predict(self.x)

        scores_for_k = self._calculate_scores(model, labels, selected_metrics)
        return k, scores_for_k

    # BASIC FUNCTIONALITIES

    def fit(self, n_clusters: int = 3, **kwargs: Any) -> "ClusteringModel":
        """Fits the clustering model to the dataset"""
        self.model_ = self._make_model(n_clusters=n_clusters, **kwargs)
        self.model_.fit(self.x)
        return self

    def predict(self, X: pd.DataFrame| np.ndarray | None = None) -> np.ndarray:
        """Predicts cluster labels for new samples"""
        if self.model_ is None:
            raise RuntimeError("Model has not been fitted. Call `fit()` first.")
        data_to_use = X if X is not None else self.x

        # Check: Does the model have a 'predict' function? (e.g. KMeans, GMM) else DBSCAN
        if hasattr(self.model_, "predict"):
            return self.model_.predict(data_to_use)
        else:
            return self.model_.fit_predict(data_to_use)

    def evaluate_k(self, metrics: MetricName | list[MetricName] = MetricName.ALL, k_range: range = range(2, 11), n_jobs: int = -1, plot: bool = True, save: bool = True, save_dir = None ) -> Dict[MetricName, list[float]]:
        """Creates and saves plots used to find optimal value for k"""
        selected_metrics = self._resolve_metrics(metrics)

        print(f"[K evaluate] Starting Computation")
        parallel_results = Parallel(n_jobs=n_jobs)(
            delayed(self._evaluate_single_k)(k, selected_metrics)
            for k in k_range
        )

        scores: Dict[MetricName, list[float]] = {
            m: [] for m in selected_metrics
        }

        for _, score_dict in sorted(parallel_results, key=lambda x: x[0]):
            for metric in selected_metrics:
                scores[metric].append(score_dict[metric])

        print("Calculations finished, creating plots...")
        if plot:
            self.plot_k_evaluation(scores, k_range, save, save_dir)
        return scores

    def _resolve_metrics(self, metrics: MetricName | list[MetricName]) -> list[MetricName]:
        """Helper to standardize the input metrics into a list."""
        if metrics == MetricName.ALL:
            selected = [MetricName.SILHOUETTE, MetricName.INERTIA, MetricName.ARI, MetricName.RI]
        elif isinstance(metrics, MetricName):
            selected = [metrics]
        else:
            selected = metrics

        # Remove supervised metrics if we don't have labels (y)
        if self.y is None:
            selected = [m for m in selected if m not in (MetricName.RI, MetricName.ARI)]

        return selected

    def _calculate_scores(self, model: ClusterEstimator, labels: np.ndarray, selected_metrics: list[MetricName]) -> \
    Dict[MetricName, float]:
        """Helper to calculate scores for a fitted model."""
        scores: Dict[MetricName, float] = {}

        # Determine centers for inertia
        if hasattr(model, "cluster_centers_"):
            centers = model.cluster_centers_
        elif hasattr(model, "means_"):
            centers = model.means_
        else:
            centers = None

        for m in selected_metrics:
            if m == MetricName.SILHOUETTE:
                # Create mask for non-noise points, In case of DSCAN
                mask = labels != -1
                # Filter data and labels (Two-liner as requested)
                labels_core, data_core = labels[mask], self.x[mask]

                # Calculate score ONLY if we have valid clusters left
                scores[m] = silhouette_score(data_core, labels_core)

            elif m == MetricName.INERTIA:
                if hasattr(model, "inertia_"):
                    scores[m] = float(model.inertia_)
                elif centers is not None:
                    d = cdist(self.x, centers)
                    scores[m] = float(np.sum(np.min(d ** 2, axis=1)))
                else:
                    scores[m] = float("nan")

            elif m == MetricName.RI:
                scores[m] = rand_score(self.y, labels)

            elif m == MetricName.ARI:
                scores[m] = adjusted_rand_score(self.y, labels)
        return scores

    def evaluate_grid(self, param_grid: Dict[str, list], metrics: MetricName | list[MetricName] = MetricName.ALL,
                      plot: bool = True, title: str = "") -> pd.DataFrame:

        selected_metrics = self._resolve_metrics(metrics)
        results = []

        # Create the grid object
        grid_combos = ParameterGrid(param_grid)

        # Wrap the grid with tqdm for the progress bar
        # desc="..." adds a prefix like "[kmeans Grid]" to the bar
        print(f"[Grid Search] Starting computation for {len(ParameterGrid(param_grid))} combinations...")
        for params in tqdm(grid_combos, desc=f"[{self.algorithm.value} Grid]", unit="cfg"):
            model = self._make_model(**params)

            try:

                model.fit(self.x)
                labels = model.fit_predict(self.x)

                # Calculate Scores
                current_scores = self._calculate_scores(model, labels, selected_metrics)

                # Store results
                row = params.copy()
                for metric, value in current_scores.items():
                    row[metric.value] = value
                results.append(row)

            except Exception as e:
                print(f"Failed for params {params}: {e}")

        df_results = pd.DataFrame(results)

        # Identify which parameters actually have more than 1 option in the grid
        varying_params = [key for key, values in param_grid.items() if len(values) > 1]

        # Only plot if exactly 2 parameters are varying (to make a 2D heatmap)
        if plot and len(varying_params) == 2:
            print(f"Plotting 2D Heatmap for varying parameters: {varying_params}")
            self._plot_grid_results(df_results, varying_params, selected_metrics, title)
        elif plot:
            print(f"Skipping heatmap: Expected 2 varying parameters, found {len(varying_params)}.")

        return df_results

    def _plot_grid_results(self, df: pd.DataFrame, grid_keys: list[str], metrics: list[MetricName], title: str = "",
                           save: bool = True, save_dir = None) -> None:
        """
        Helper to plot 2D heatmaps for evaluate_grid.
        Expects exactly 2 parameters in grid_keys.
        """

        if save_dir is None:
            save_path = SAVE_PATH
        else :
            save_path = Path(save_dir)
        # The two parameters we are varying (e.g. 'n_components' & 'covariance_type')
        x_param = grid_keys[0]
        y_param = grid_keys[1]

        for m in metrics:
            metric_name = m.value

            # Skip if this metric wasn't calculated (e.g. ARI when no labels exist)
            if metric_name not in df.columns:
                continue

            plt.figure(figsize=(10, 6))

            # Smart Color Logic:
            # For Inertia (and Davies-Bouldin), Lower is Better -> Use reversed colormap (viridis_r)
            # For Silhouette/ARI, Higher is Better -> Use standard colormap (viridis)
            if m == MetricName.INERTIA:
                cmap = "viridis_r"
            else:
                cmap = "viridis"

            try:
                # Pivot table: Rows=Param2, Cols=Param1, Values=Score
                # This works automatically whether params are Strings or Ints
                pivot_table = df.pivot(index=y_param, columns=x_param, values=metric_name)

                sns.heatmap(pivot_table, annot=True, fmt=".3f", cmap=cmap)

                plt.title(f"{self.algorithm.value.upper()} - {metric_name.upper()}")
                plt.ylabel(y_param)
                plt.xlabel(x_param)
                plt.tight_layout()
                if save:
                    save_path.mkdir(parents=True, exist_ok=True)
                    if title:
                        filepath = os.path.join(save_path, f"grid_{title}_{self.algorithm.value}_{metric_name}.png")
                    else:
                        filepath = os.path.join(save_path, f"grid_{self.algorithm.value}_{metric_name}.png")
                    plt.savefig(filepath, dpi=150, bbox_inches="tight")
                    print(f"[saved] {filepath}")
                else:
                    plt.show()

            except ValueError as e:
                print(f"[Plot Error] Could not plot heatmap for {metric_name}: {e}")

    def plot_k_evaluation(self, scores: dict[MetricName, list[float]], k_range: range, save: bool = False, save_dir: str | None = None) -> None:

        k_values = list(k_range)

        if save_dir is None:
            save_path = SAVE_PATH
        else :
            save_path = Path(save_dir)
        
        for metric, values in scores.items():
            plt.figure(figsize=(8, 5))
            plt.plot(k_values, values, marker="o")
            plt.xlabel("Number of clusters (k)")
            plt.ylabel(metric.value.upper())
            plt.title(f"{self.algorithm.value.upper()} - {metric.value.upper()}")
            plt.grid(True)

            if save:
                save_path.mkdir(parents=True, exist_ok=True)
                filepath = os.path.join(save_path, f"{self.algorithm.value}_{metric.value}.png")
                plt.savefig(filepath, dpi=150, bbox_inches="tight")
                print(f"[saved] {filepath}")

            else:
                plt.show()


    def findOptimum(self):
        """Finds the best model for the data"""
        return 

    # VISUALISATION

    def plot(self, x: pd.DataFrame | np.ndarray | None = None, title: str = "", save: bool = True, save_dir = None) -> None:
        if save_dir is None:
            save_path = SAVE_PATH
        else :
            save_path = Path(save_dir)

        if self.model_ is None:
            raise RuntimeError("Call fit() before plotting")
        if x is None:
            x = self.x
        elif isinstance(x, pd.DataFrame):
            x = x.to_numpy()
        labels = self.predict(x)
        pca = PCA(2)
        x_pca = pca.fit_transform(x)
        # Plotting

        plt.figure(figsize = (16,12))
        scatter = plt.scatter(
            x_pca[:,0],
            x_pca[:,1],
            c = labels,
            cmap = "tab10",
            s = 10
        )
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")

        if title:
            plt.title(title)
        else:
            plt.title(f"PCA Clustering Visualization ({self.algorithm})")
        if save:
            save_path.mkdir(parents=True, exist_ok=True)
            filepath = os.path.join(save_path, f"{self.algorithm.value}_visualization.png")
            plt.savefig(filepath, dpi=150, bbox_inches="tight")
            print(f"[saved] {filepath}")

        else:
            plt.show()

    def plot_ecdf(self, save: bool = True, save_dir=None) -> None:
        if save_dir is None:
            save_path = SAVE_PATH
        else:
            save_path = Path(save_dir)

        if self.model_ is None:
            raise RuntimeError("Call fit() before plotting")

        labels = self.predict(self.x)

        # Convert to Series for easy counting
        # For DBSCAN, we typically exclude -1 (noise) from "Cluster Size" analysis
        unique_labels = np.unique(labels)
        if self.algorithm == AlgorithmName.DBSCAN and -1 in unique_labels:
            print(f"[{self.algorithm.value}] Excluding noise (-1) from ECDF plot.")
            valid_labels = labels[labels != -1]
        else:
            valid_labels = labels

        if len(valid_labels) == 0:
            print("No valid clusters found to plot ECDF.")
            return

        # Calculate cluster sizes (number of flows per cluster)
        _, cluster_counts = np.unique(valid_labels, return_counts=True)

        plt.figure(figsize=(8, 5))
        # Plot ECDF of the counts
        sns.ecdfplot(cluster_counts)
        plt.xlabel("Number of Flows (Cluster Size)")
        plt.ylabel("Proportion of Clusters")
        plt.title(f"ECDF of Flows per Cluster ({self.algorithm.value.upper()})")
        plt.grid(True)

        if save:
            save_path.mkdir(parents=True, exist_ok=True)
            filepath = os.path.join(save_path, f"{self.algorithm.value}_ecdf.png")
            plt.savefig(filepath, dpi=150, bbox_inches="tight")
            print(f"[saved] {filepath}")
        else:
            plt.show()

    def plot_gt(self, title: str = "", save: bool = True, save_dir=None) -> None:
        """Plots the PCA projection colored by Ground Truth labels."""
        if self.y is None:
            print("[Plot GT] No labels (y) found in dataset. Skipping GT plot.")
            return

        if save_dir is None:
            save_path = SAVE_PATH
        else:
            save_path = Path(save_dir)

        filename = "ground_truth_visualization.png"
        filepath = os.path.join(save_path, filename)

        # If saving is enabled and file exists, skip everything
        if save and os.path.exists(filepath):
            print(f"[Plot GT] File already exists at {filepath}. Skipping.")
            return

        # Compute PCA just like the clustering plot
        pca = PCA(n_components=2)
        x_pca = pca.fit_transform(self.x)

        plt.figure(figsize=(16, 12))

        # Plot using Seaborn (handles string labels + legend automatically)
        sns.scatterplot(
            x=x_pca[:, 0],
            y=x_pca[:, 1],
            hue=self.y,  # <--- Color by GROUND TRUTH (self.y)
            palette="tab10",
            s=15,  # Slightly larger points for visibility
            alpha=0.7,  # Slight transparency to see overlaps
            edgecolor=None
        )

        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")

        if title:
            plt.title(title)
        else:
            plt.title("PCA Visualization: Ground Truth Labels")

        if save:
            save_path.mkdir(parents=True, exist_ok=True)
            # Use a static filename since GT is the same regardless of the algorithm
            plt.savefig(filepath, dpi=150, bbox_inches="tight")
            print(f"[saved] {filepath}")
        else:
            plt.show()

def save_best_params(algorithm_name: str, params: dict):
    """Saves parameters, creating the file or updating the specific algorithm entry."""
    data = {}
    if os.path.exists(PARAMS_FILE):
        with open(PARAMS_FILE, "r") as f:
            data = json.load(f)

    def convert(o):
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return o

    # We rebuild the params dict using the converter logic
    clean_params = {k: convert(v) for k, v in params.items()}
    data[algorithm_name] = clean_params

    with open(PARAMS_FILE, "w") as f:
        json.dump(data, f, indent=4)
    print(f"[Saved] {algorithm_name} params to {PARAMS_FILE}")


def load_best_params(algorithm_name: str) -> dict | None:
    """Loads the best parameters from the JSON file."""
    if not PARAMS_FILE.exists():
        print(f"[Load Params] File not found: {PARAMS_FILE}")
        return None

    with open(PARAMS_FILE, "r") as f:
        data = json.load(f)
        return data.get(algorithm_name)



if __name__ == "__main__":
    # 1. Configuration
    CONFIG = {
        "data_file": "../../res/dataset_for_clustering.csv",
        "sample_size": None,  # Use subset for speed during tuning
        "k_range": range(2, 16),  # Ground truth is 12, so check around that
        "load_params": True,
        "algorithms": {
            ## Standard Euclidean KMeans
            #"KMEANS": {
            #    "run_k_eval": False,
            #    "grid_params": {
            #        "n_clusters": [8],  # Fixed k for tuning other params
            #        "init": ["k-means++", "random"],
            #        "algorithm": ["lloyd", "elkan"]
            #    }
            #},
            # KMedoids (Supports different metrics: Cosine vs Euclidean)
            #"KMEDOIDS": {
            #    "run_k_eval": True,
            #    "grid_params": {
            #        "n_clusters": [11],
            #        "metric": ["euclidean", "cosine"],  # <--- Similarity Metric Check
            #        "method": ["pam", "alternate"]
            #    }
            #},
            # Gaussian Mixture
            "GMM": {
                "run_k_eval": False,
                "grid_params": {
                    "n_components": [13],
                    "covariance_type": ["full", "tied", "diag", "spherical"],
                    "init_params": ["kmeans", "random"]
                }
            }
            # DBSCAN (Density based, no K)
            #"DBSCAN": {
            #    "run_k_eval": False,  # DBSCAN determines clusters automatically
            #    "grid_params": {
            #        "eps": [0.05,0.1 ,0.3, 0.5, 1.0, 1.5, 2.0, 3.0],
            #        "min_samples": [5, 10, 20],
            #        "metric": ["euclidean", "cosine"]  # <--- Similarity Metric Check
            #    }
            #}
        }
    }

    print("=" * 50)
    print("STARTING CLUSTERING ANALYSIS")
    print(f"Loading data from {CONFIG['data_file']}...")
    print("=" * 50)
    # Load Data
    # Note: Using simple pandas load as in original script, but respecting the path
    full_data = pd.read_csv(CONFIG['data_file'])
    # Use a subset for faster tuning if specified
    if CONFIG["sample_size"]:
        print(f"Subsampling to top {CONFIG['sample_size']} rows for performance...")
        data = full_data.head(CONFIG["sample_size"])
    else:
        data = full_data

    # 3. Iterate over Models (Except Spectral)
    for algo_name, settings in CONFIG["algorithms"].items():
        algo_enum = AlgorithmName(algo_name.lower())

        print(f"\n\n{'#' * 40}")
        print(f" Processing Algorithm: {algo_name} ")
        print(f"{'#' * 40}")

        model_wrapper = ClusteringModel(algo_enum, data)

        # Determine optimal k (if applicable)
        if settings["run_k_eval"]:
            print(f"\n--- 1. Determining Optimal Cluster Count (k) ---")
            # We use Silhouette and Inertia to judge 'k'
            # RI/ARI are included to compare against ground truth labels
            model_wrapper.evaluate_k(
                metrics=MetricName.ALL,
                k_range=CONFIG["k_range"],
                n_jobs=5,
                plot=True,
                save=True
            )
        else:
            print(f"\n--- Skipping k-evaluation (Algorithm is density-based) ---")

            best_params = None

            if CONFIG["load_params"]:
                print(f"Loading parameters from {PARAMS_FILE}...")
                best_params = load_best_params(algo_name)
                if best_params:
                    print(f"Loaded params: {best_params}")
                else:
                    print("Could not load params. Falling back to tuning/defaults.")
            else:

                # Hyperparameter Tuning & Similarity Metrics
                temp_results = []
                print(f"\n---  Tuning & Similarity Metrics Check ---")
                print("Running Grid Search to find best parameters and compare metrics (e.g. Euclidean vs Cosine)...")
                if "metric" in settings["grid_params"]:
                    for metric in settings["grid_params"]["metric"]:
                        print(f"\n-- Testing Similarity Metric: {metric} --")
                        # Update metric in grid params to single value for this run
                        single_metric_grid = settings["grid_params"].copy()
                        single_metric_grid["metric"] = [metric]

                        tmp = model_wrapper.evaluate_grid(
                            single_metric_grid,
                            metrics=MetricName.ALL,
                            plot=True,
                            title=metric
                        )
                        temp_results.append(tmp)
                    # Combine all results
                    df_results = pd.concat(temp_results, ignore_index=True)
                else:
                    df_results = model_wrapper.evaluate_grid(
                        settings["grid_params"],
                        metrics=MetricName.ALL,
                        plot=True
                    )

                # Simple logic to pick 'best' params: Maximize Silhouette Score
                if not df_results.empty and "silhouette" in df_results.columns:
                    best_row = df_results.sort_values(by="silhouette", ascending=False).iloc[0]
                    best_score = best_row["silhouette"]

                    # Reconstruct best params dict
                    best_params = {k: v for k, v in best_row.items() if k in settings["grid_params"]}

                    print(f"Best Parameters found (Silhouette={best_score:.4f}):")
                    pprint(best_params)
                    # Save best params
                    save_best_params(algo_name, best_params)
            #  Select Best Model and Visualize ---
            print(f"\n---  Final Model Evaluation & Visualization ---")

            print("Fitting final model with best parameters...")
            # For K-based, ensure n_clusters is passed if it was in the grid
            if "n_clusters" in best_params:
                k_best = int(best_params.pop("n_clusters"))  # extract k
                model_wrapper.fit(n_clusters=k_best, **best_params)
            else:
                # For DBSCAN
                model_wrapper.fit(**best_params)

            print(f"Exporting original data with {algo_name} cluster labels...")
            # Get labels from the fitted model
            labels = model_wrapper.predict()

            # Create a copy of the original (unprocessed) DataFrame
            clustered_df = model_wrapper.data.copy()
            clustered_df['cluster_assignment'] = labels
            output_csv_name = f"clustered_data_{algo_name.lower()}.csv"
            full_save_path = SAVE_PATH / output_csv_name
            clustered_df.to_csv(full_save_path, index=False)
            print(f"[Saved] Clustered results to: {full_save_path}")


            # Plot the final result
            model_wrapper.plot(title=f"Final Clustering: {algo_name} (Best Params)")

            model_wrapper.plot_ecdf()

            #TODO should only be run once
            model_wrapper.plot_gt()


    print("\n\nAnalysis Complete.")
