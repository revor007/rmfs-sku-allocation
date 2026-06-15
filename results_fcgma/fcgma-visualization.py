from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_HISTORY_PATH = BASE_DIR / "objective_history_trial_1.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize FCGMA objective-history CSV files."
    )
    parser.add_argument(
        "history_csv",
        nargs="?",
        default=str(DEFAULT_HISTORY_PATH),
        help="Path to an objective history CSV. Defaults to objective_history_trial_1.csv.",
    )
    return parser.parse_args()


def load_objective_history(path: Path) -> pd.DataFrame:
    history_df = pd.read_csv(path)
    required_columns = {"iteration", "best_objective"}
    missing_columns = required_columns.difference(history_df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns {sorted(missing_columns)} in {path}."
        )

    history_df = history_df.copy()
    history_df["iteration"] = pd.to_numeric(
        history_df["iteration"],
        errors="coerce",
    )
    history_df["best_objective"] = pd.to_numeric(
        history_df["best_objective"],
        errors="coerce",
    )
    history_df = history_df.dropna(subset=["iteration", "best_objective"]).copy()
    if history_df.empty:
        raise ValueError(f"No valid objective-history rows remain in {path}.")

    history_df["iteration"] = history_df["iteration"].astype(int)
    history_df = history_df.sort_values("iteration").reset_index(drop=True)
    return history_df


def summarize_history(history_df: pd.DataFrame) -> dict[str, float | int]:
    improvement_mask = history_df["best_objective"].gt(
        history_df["best_objective"].shift(fill_value=history_df["best_objective"].iloc[0] - 1)
    )
    improvement_points = history_df[improvement_mask].copy()
    return {
        "start_objective": float(history_df["best_objective"].iloc[0]),
        "final_objective": float(history_df["best_objective"].iloc[-1]),
        "best_objective": float(history_df["best_objective"].max()),
        "iterations_plotted": int(len(history_df)),
        "improvement_count": int(improvement_mask.sum()),
        "improvement_points": improvement_points,
    }


def plot_objective_history(history_df: pd.DataFrame, source_path: Path) -> Path:
    summary = summarize_history(history_df)
    improvement_points = summary["improvement_points"]
    output_path = source_path.with_suffix(".png")

    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.step(
        history_df["iteration"],
        history_df["best_objective"],
        where="post",
        color="#1f77b4",
        linewidth=2.2,
        label="Best objective",
    )
    ax.plot(
        history_df["iteration"],
        history_df["best_objective"],
        color="#1f77b4",
        alpha=0.18,
        linewidth=1.0,
    )

    if not improvement_points.empty:
        ax.scatter(
            improvement_points["iteration"],
            improvement_points["best_objective"],
            color="#d62728",
            s=22,
            zorder=3,
            label="Improvement point",
        )

    ax.set_title(f"Objective History: {source_path.stem}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best objective value")
    ax.grid(True, alpha=0.28)
    ax.legend(loc="lower right")

    annotation_lines = [
        f"Iterations plotted: {summary['iterations_plotted']:,}",
        f"Start objective: {summary['start_objective']:.6f}",
        f"Final objective: {summary['final_objective']:.6f}",
        f"Best objective: {summary['best_objective']:.6f}",
        f"Improvement points: {summary['improvement_count']:,}",
    ]
    ax.text(
        0.98,
        0.04,
        "\n".join(annotation_lines),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#cccccc"},
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    history_path = Path(args.history_csv).resolve()
    history_df = load_objective_history(history_path)
    output_path = plot_objective_history(history_df, history_path)

    print(f"Source CSV: {history_path}")
    print(f"Rows plotted: {len(history_df):,}")
    print(f"Visualization saved to: {output_path}")


if __name__ == "__main__":
    main()
