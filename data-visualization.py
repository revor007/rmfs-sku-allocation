from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter


BASE_DIR = Path(__file__).resolve().parent
PRODUCT_DATA_CANDIDATES = [
    BASE_DIR.parent / "Preprocessing" / "preprocessed_final.csv",
    BASE_DIR.parent / "fcgma" / "Preprocessing" / "preprocessed_final.csv",
]
CLUSTERING_DATASET_DIR = BASE_DIR / "Clustering" / "dataset"

ORIGINAL_PRICE_VISUALIZATION_PATH = BASE_DIR / "original_price_distribution.png"
SHELF_LIFE_VISUALIZATION_PATH = BASE_DIR / "shelf_life_distribution.png"
DISCOUNT_VISUALIZATION_PATH = BASE_DIR / "discount_distribution.png"
CAPACITY_VISUALIZATION_PATH = BASE_DIR / "clustering_capacity_distribution.png"
SKEW_SUMMARY_VISUALIZATION_PATH = (
    BASE_DIR / "clustering_dataset_skew_summary.png"
)

ORIGINAL_PRICE_CANDIDATES = ["original_price"]
SHELF_LIFE_CANDIDATES = ["shelf_life"]
ESTIMATED_DISCOUNT_CANDIDATES = ["estimation_discount"]
CAPACITY_CANDIDATES = ["capacity"]
CAPACITY_CATEGORY_CANDIDATES = ["capacity_category"]

CLUSTERING_SKEW_VARIABLES = {
    "capacity": {
        "label": "Capacity",
        "transform": "identity",
    },
    "original_price": {
        "label": "Original Price",
        "transform": "identity",
    },
    "estimation_discount": {
        "label": "Discount",
        "transform": "identity",
    },
    "shelf_life": {
        "label": "Shelf Life",
        "transform": "ceil",
    },
}

CAPACITY_CATEGORY_LABELS = {
    "volume": "Volume Capacity (ml)",
    "weight": "Weight Capacity (g)",
    "count": "Count Capacity",
    "length": "Length Capacity (cm)",
}


def find_product_data_path():
    for candidate in PRODUCT_DATA_CANDIDATES:
        if candidate.exists():
            return candidate

    checked_paths = ", ".join(str(path) for path in PRODUCT_DATA_CANDIDATES)
    raise FileNotFoundError(
        "Could not find preprocessed_final.csv. Checked: "
        f"{checked_paths}"
    )


def find_clustering_dataset_files():
    if not CLUSTERING_DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Could not find clustering dataset folder: {CLUSTERING_DATASET_DIR}"
        )

    dataset_files = sorted(
        path
        for path in CLUSTERING_DATASET_DIR.glob("*_dataset.csv")
        if ".0_dataset.csv" not in path.name
    )
    if not dataset_files:
        raise FileNotFoundError(
            "Could not find current clustering dataset CSV files in "
            f"{CLUSTERING_DATASET_DIR}"
        )
    return dataset_files


def normalize_column_name(name):
    return str(name).replace("\ufeff", "").strip().lower()


def find_column(columns, candidates, required=True):
    normalized = {normalize_column_name(col): col for col in columns}
    for candidate in candidates:
        actual_name = normalized.get(normalize_column_name(candidate))
        if actual_name is not None:
            return actual_name

    if required:
        raise KeyError(
            f"Could not find any of the expected columns {candidates}. "
            f"Available columns: {list(columns)}"
        )
    return None


def get_writable_output_path(path):
    if not path.exists():
        return path

    try:
        with open(path, "a", encoding="utf-8"):
            return path
    except PermissionError:
        return path.with_name(f"{path.stem}_latest{path.suffix}")


def build_discount_series(df, estimated_discount_col):
    discount_series = pd.to_numeric(
        df[estimated_discount_col],
        errors="coerce",
    )

    finite_values = discount_series.dropna()
    if not finite_values.empty and finite_values.abs().max() <= 1.0:
        discount_series = discount_series * 100.0

    return discount_series


def build_shelf_life_series(series):
    return np.ceil(pd.to_numeric(series, errors="coerce"))


def classify_skewness(skewness, threshold=0.35):
    if pd.isna(skewness):
        return "undetermined"
    if skewness > threshold:
        return "right_skewed"
    if skewness < -threshold:
        return "left_skewed"
    return "approximately_symmetric"


def load_product_data(path):
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig", decimal=",")

    columns = {
        "original_price": find_column(df.columns, ORIGINAL_PRICE_CANDIDATES),
        "shelf_life": find_column(df.columns, SHELF_LIFE_CANDIDATES),
        "discount_estimated": find_column(
            df.columns,
            ESTIMATED_DISCOUNT_CANDIDATES,
        ),
        "capacity": find_column(df.columns, CAPACITY_CANDIDATES, required=False),
        "capacity_category": find_column(
            df.columns,
            CAPACITY_CATEGORY_CANDIDATES,
            required=False,
        ),
    }

    df = df.copy()
    df[columns["original_price"]] = pd.to_numeric(
        df[columns["original_price"]],
        errors="coerce",
    )
    df[columns["shelf_life"]] = pd.to_numeric(
        df[columns["shelf_life"]],
        errors="coerce",
    )
    df["_discount_percent"] = build_discount_series(
        df,
        columns["discount_estimated"],
    )
    df["_shelf_life_ceiled"] = build_shelf_life_series(
        df[columns["shelf_life"]],
    )
    if columns["capacity"] is not None:
        df[columns["capacity"]] = pd.to_numeric(
            df[columns["capacity"]],
            errors="coerce",
        )
    if columns["capacity_category"] is not None:
        df[columns["capacity_category"]] = (
            df[columns["capacity_category"]]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
        )

    return df, columns


def transform_clustering_variable(series, transform_name):
    numeric = pd.to_numeric(series, errors="coerce")
    if transform_name == "ceil":
        return np.ceil(numeric)
    return numeric


def summarize_clustering_dataset_shapes(dataset_files):
    summary_rows = []

    for path in dataset_files:
        df = pd.read_csv(
            path,
            sep=";",
            encoding="utf-8-sig",
            decimal=",",
            usecols=list(CLUSTERING_SKEW_VARIABLES.keys()),
        )
        dataset_name = path.stem.replace("_dataset", "")

        for variable_name, config in CLUSTERING_SKEW_VARIABLES.items():
            values = transform_clustering_variable(
                df[variable_name],
                config["transform"],
            ).dropna()
            if values.empty:
                skewness = np.nan
                shape = "undetermined"
            else:
                skewness = float(values.skew())
                shape = classify_skewness(skewness)

            summary_rows.append(
                {
                    "dataset_name": dataset_name,
                    "variable": variable_name,
                    "variable_label": config["label"],
                    "skewness": skewness,
                    "shape": shape,
                    "observation_count": int(len(values)),
                }
            )

    return pd.DataFrame(summary_rows)


def save_figure(fig, path):
    output_path = get_writable_output_path(path)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def format_stat(value):
    if pd.isna(value):
        return "N/A"
    return f"{float(value):,.2f}"


def describe_distribution_shape(skewness, mean_value, median_value, std_value):
    if pd.isna(skewness):
        return "Undetermined"

    scale = std_value if std_value and std_value > 0 else max(abs(mean_value), 1.0)
    mean_median_gap = abs(mean_value - median_value)

    if abs(skewness) < 0.35 and mean_median_gap <= 0.1 * scale:
        return "Approximately symmetric"
    if skewness >= 0.35:
        return "Right-skewed"
    if skewness <= -0.35:
        return "Left-skewed"
    return "Mild skew"


def plot_numeric_distribution(
    series,
    output_path,
    title,
    x_label,
    color,
    percent_axis=False,
    note_lines=None,
):
    valid_values = pd.to_numeric(series, errors="coerce").dropna()
    excluded_count = int(len(series) - len(valid_values))
    if valid_values.empty:
        raise ValueError(f"No valid values were found for {title}.")

    bin_count = min(40, max(12, int(len(valid_values) ** 0.5)))
    fig, ax = plt.subplots(figsize=(10, 6))
    counts, bin_edges, _ = ax.hist(
        valid_values,
        bins=bin_count,
        color=color,
        edgecolor="black",
        linewidth=0.8,
        alpha=0.82,
        label="Observed distribution",
    )

    mean_value = valid_values.mean()
    median_value = valid_values.median()
    std_value = valid_values.std(ddof=0)
    skewness = valid_values.skew()
    shape_label = describe_distribution_shape(
        skewness=skewness,
        mean_value=mean_value,
        median_value=median_value,
        std_value=std_value,
    )

    ax.axvline(
        mean_value,
        color="#f58518",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_value:.2f}",
    )
    ax.axvline(
        median_value,
        color="#54a24b",
        linestyle="-.",
        linewidth=2,
        label=f"Median: {median_value:.2f}",
    )

    if std_value > 0 and np.isfinite(std_value):
        x_values = np.linspace(valid_values.min(), valid_values.max(), 400)
        bin_width = float(bin_edges[1] - bin_edges[0]) if len(bin_edges) > 1 else 1.0
        normal_curve = (
            (
                1.0
                / (std_value * np.sqrt(2.0 * np.pi))
            )
            * np.exp(-0.5 * ((x_values - mean_value) / std_value) ** 2)
            * len(valid_values)
            * bin_width
        )
        ax.plot(
            x_values,
            normal_curve,
            color="#e45756",
            linewidth=2,
            label="Normal reference",
        )

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Number of products")
    ax.grid(axis="y", alpha=0.3)
    if percent_axis:
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.legend()

    annotation_lines = [
        f"Products plotted: {len(valid_values):,}",
        f"Mean: {format_stat(mean_value)}",
        f"Median: {format_stat(median_value)}",
        f"Std dev: {format_stat(std_value)}",
        f"Skewness: {format_stat(skewness)}",
        f"Shape: {shape_label}",
    ]
    if note_lines:
        annotation_lines.extend(note_lines)

    ax.text(
        0.98,
        0.95,
        "\n".join(annotation_lines),
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "#cccccc"},
    )

    fig.tight_layout()
    return save_figure(fig, output_path)


def plot_capacity_histogram_on_axis(ax, series, category_label, color):
    valid_values = pd.to_numeric(series, errors="coerce").dropna()
    if valid_values.empty:
        return 0

    bin_count = min(40, max(12, int(len(valid_values) ** 0.5)))
    counts, bin_edges, _ = ax.hist(
        valid_values,
        bins=bin_count,
        color=color,
        edgecolor="black",
        linewidth=0.8,
        alpha=0.82,
        label="Observed distribution",
    )

    mean_value = valid_values.mean()
    median_value = valid_values.median()
    std_value = valid_values.std(ddof=0)
    skewness = valid_values.skew()
    shape_label = describe_distribution_shape(
        skewness=skewness,
        mean_value=mean_value,
        median_value=median_value,
        std_value=std_value,
    )

    ax.axvline(
        mean_value,
        color="#f58518",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_value:.2f}",
    )
    ax.axvline(
        median_value,
        color="#54a24b",
        linestyle="-.",
        linewidth=2,
        label=f"Median: {median_value:.2f}",
    )

    if std_value > 0 and np.isfinite(std_value):
        x_values = np.linspace(valid_values.min(), valid_values.max(), 400)
        bin_width = float(bin_edges[1] - bin_edges[0]) if len(bin_edges) > 1 else 1.0
        normal_curve = (
            (
                1.0
                / (std_value * np.sqrt(2.0 * np.pi))
            )
            * np.exp(-0.5 * ((x_values - mean_value) / std_value) ** 2)
            * len(valid_values)
            * bin_width
        )
        ax.plot(
            x_values,
            normal_curve,
            color="#e45756",
            linewidth=2,
            label="Normal reference",
        )

    ax.set_title(category_label)
    ax.set_xlabel(category_label)
    ax.set_ylabel("Number of products")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    ax.text(
        0.98,
        0.95,
        "\n".join(
            [
                f"Products plotted: {len(valid_values):,}",
                f"Mean: {format_stat(mean_value)}",
                f"Median: {format_stat(median_value)}",
                f"Std dev: {format_stat(std_value)}",
                f"Skewness: {format_stat(skewness)}",
                f"Shape: {shape_label}",
            ]
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "#cccccc"},
    )
    return len(valid_values)


def plot_global_capacity_distribution(df, columns):
    capacity_col = columns.get("capacity")
    if capacity_col is None:
        raise ValueError("The global product dataset does not include a capacity column.")

    category_col = columns.get("capacity_category")
    plot_df = df[[capacity_col]].copy()
    plot_df = plot_df.rename(columns={capacity_col: "capacity"})
    plot_df["capacity"] = pd.to_numeric(plot_df["capacity"], errors="coerce")

    if category_col is not None:
        plot_df["capacity_category"] = (
            df[category_col].fillna("").astype(str).str.strip().str.lower()
        )
    else:
        plot_df["capacity_category"] = "all"

    plot_df = plot_df.dropna(subset=["capacity"]).copy()
    plot_df = plot_df[plot_df["capacity_category"] != ""].copy()
    if plot_df.empty:
        raise ValueError("No valid capacity values were found in the global product dataset.")

    preferred_order = ["volume", "weight", "count", "length", "all"]
    ordered_categories = [
        category for category in preferred_order if category in plot_df["capacity_category"].unique()
    ]
    ordered_categories.extend(
        sorted(
            category
            for category in plot_df["capacity_category"].unique()
            if category not in ordered_categories
        )
    )

    category_frames = [
        (category, plot_df[plot_df["capacity_category"] == category].copy())
        for category in ordered_categories
    ]
    category_frames = [
        (category, category_df)
        for category, category_df in category_frames
        if not category_df.empty
    ]
    if not category_frames:
        raise ValueError("No capacity categories were available for the global capacity chart.")

    colors = ["#4c78a8", "#72b7b2", "#f58518", "#54a24b", "#b279a2"]
    fig, axes = plt.subplots(
        len(category_frames),
        1,
        figsize=(12, max(5, 4.1 * len(category_frames))),
        squeeze=False,
    )
    axes = axes.flatten()

    total_plotted = 0
    for index, (category, category_df) in enumerate(category_frames):
        axis = axes[index]
        category_label = CAPACITY_CATEGORY_LABELS.get(
            category,
            f"{category.title()} Capacity",
        )
        plotted_count = plot_capacity_histogram_on_axis(
            axis,
            category_df["capacity"],
            category_label,
            colors[index % len(colors)],
        )
        total_plotted += plotted_count

    fig.suptitle("Global Capacity Distribution", fontsize=16, y=1.01)
    fig.text(
        0.5,
        0.01,
        (
            f"Source: preprocessed_final.csv | "
            f"Products plotted across all panels: {total_plotted:,} | "
            "Panels are separated by capacity category so units are not mixed"
        ),
        ha="center",
        fontsize=10,
    )
    fig.tight_layout()
    return save_figure(fig, CAPACITY_VISUALIZATION_PATH)


def plot_clustering_skew_summary(summary_df):
    if summary_df.empty:
        raise ValueError("No clustering dataset skew summary data was available.")

    plot_df = (
        summary_df[summary_df["shape"] != "undetermined"]
        .groupby(["variable_label", "shape"])
        .size()
        .unstack(fill_value=0)
    )
    ordered_variables = [
        CLUSTERING_SKEW_VARIABLES[key]["label"]
        for key in CLUSTERING_SKEW_VARIABLES
        if CLUSTERING_SKEW_VARIABLES[key]["label"] in plot_df.index
    ]
    plot_df = plot_df.reindex(ordered_variables)

    shape_order = [
        "right_skewed",
        "approximately_symmetric",
        "left_skewed",
    ]
    plot_df = plot_df.reindex(columns=shape_order, fill_value=0)

    shape_labels = {
        "right_skewed": "Right-skewed",
        "approximately_symmetric": "Approximately symmetric",
        "left_skewed": "Left-skewed",
    }
    colors = {
        "right_skewed": "#e45756",
        "approximately_symmetric": "#72b7b2",
        "left_skewed": "#4c78a8",
    }

    fig, ax = plt.subplots(figsize=(11, 6.5))
    bottom = np.zeros(len(plot_df), dtype=float)
    total_counts = plot_df.sum(axis=1).to_numpy(dtype=float)

    for shape in shape_order:
        values = plot_df[shape].to_numpy(dtype=float)
        bars = ax.bar(
            plot_df.index,
            values,
            bottom=bottom,
            color=colors[shape],
            edgecolor="black",
            linewidth=0.8,
            label=shape_labels[shape],
        )
        for index, bar in enumerate(bars):
            count = int(values[index])
            if count <= 0:
                continue
            percentage = 100.0 * count / total_counts[index] if total_counts[index] else 0.0
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bottom[index] + values[index] / 2,
                f"{count}\n({percentage:.0f}%)",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if shape != "approximately_symmetric" else "black",
                weight="bold",
            )
        bottom += values

    ax.set_title("Distribution Shape Across Clustering Datasets")
    ax.set_xlabel("Variable")
    ax.set_ylabel("Number of dataset files")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    dataset_count = int(summary_df["dataset_name"].nunique())
    ax.text(
        0.98,
        0.95,
        "\n".join(
            [
                f"Datasets analyzed: {dataset_count:,}",
                "Current *_dataset.csv files only",
                "Shelf life is ceiled before skewness",
                "Shape rule: skewness > 0.35, < -0.35, otherwise symmetric",
            ]
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "#cccccc"},
    )

    fig.tight_layout()
    return save_figure(fig, SKEW_SUMMARY_VISUALIZATION_PATH)


def plot_original_price_distribution(series):
    return plot_numeric_distribution(
        series=series,
        output_path=ORIGINAL_PRICE_VISUALIZATION_PATH,
        title="Original Price Distribution",
        x_label="Original price",
        color="#4c78a8",
    )


def plot_shelf_life_distribution(series):
    return plot_numeric_distribution(
        series=series,
        output_path=SHELF_LIFE_VISUALIZATION_PATH,
        title="Shelf Life Distribution",
        x_label="Shelf life",
        color="#72b7b2",
    )


def plot_discount_distribution(series):
    return plot_numeric_distribution(
        series=series,
        output_path=DISCOUNT_VISUALIZATION_PATH,
        title="Discount Distribution",
        x_label="Discount (%)",
        color="#eeca3b",
        percent_axis=True,
    )


def main():
    source_path = find_product_data_path()
    df_product, columns = load_product_data(source_path)
    clustering_dataset_files = find_clustering_dataset_files()
    skew_summary_df = summarize_clustering_dataset_shapes(clustering_dataset_files)

    original_price_path = plot_original_price_distribution(
        df_product[columns["original_price"]]
    )
    shelf_life_path = plot_shelf_life_distribution(df_product["_shelf_life_ceiled"])
    discount_path = plot_discount_distribution(df_product["_discount_percent"])
    capacity_path = plot_global_capacity_distribution(df_product, columns)
    skew_summary_path = plot_clustering_skew_summary(skew_summary_df)

    print(f"Source file: {source_path}")
    print(f"Original price figure saved to: {original_price_path}")
    print(f"Shelf life figure saved to: {shelf_life_path}")
    print(f"Discount figure saved to: {discount_path}")
    print(f"Clustering capacity figure saved to: {capacity_path}")
    print(f"Clustering skew summary figure saved to: {skew_summary_path}")


if __name__ == "__main__":
    main()
