"""Chart generation service.

Renders a fixed set of exploratory charts for a stored dataset in
two formats:

- Matplotlib PNG for static consumption (Agg backend, no display).
- Plotly HTML for interactive exploration in a browser.

Files are written under ``<storage>/charts/<dataset_id>/`` and
exposed at ``<charts_static_url_prefix>/<dataset_id>/<file>``.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pathlib import Path
from uuid import UUID

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px

from app.core.exceptions import DatasetNotFoundError
from app.core.logging import get_logger
from app.models.charts import (
    ChartArtifact,
    ColumnChart,
    DatasetCharts,
    DatasetChartsResponse,
)
from app.repositories.dataset_repository import DatasetRepository
from app.utils.csv_inspector import detect_encoding, detect_separator

logger = get_logger(__name__)

_CATEGORICAL_DTYPES: tuple[str, ...] = ("object", "str", "string", "category", "bool")
_TOP_N_CATEGORICAL: int = 10


class ChartService:
    """Generate charts and return URLs pointing to the produced files."""

    def __init__(
        self,
        repository: DatasetRepository,
        charts_dir: Path,
        url_prefix: str,
    ) -> None:
        self._repository = repository
        self._charts_dir = charts_dir
        self._url_prefix = url_prefix.rstrip("/")

    def generate(self, dataset_id: UUID) -> DatasetChartsResponse:
        """Render all charts for ``dataset_id`` and return their URLs."""
        path = self._repository.find(dataset_id)
        if path is None:
            raise DatasetNotFoundError(f"Dataset '{dataset_id}' was not found.")

        frame = self._load_dataframe(path)
        target = self._charts_dir / str(dataset_id)
        target.mkdir(parents=True, exist_ok=True)

        numeric = frame.select_dtypes(include="number")
        categorical = frame.select_dtypes(include=list(_CATEGORICAL_DTYPES))

        charts = DatasetCharts(
            histograms=[
                self._histogram(target, dataset_id, col, numeric[col]) for col in numeric.columns
            ],
            boxplots=[
                self._boxplot(target, dataset_id, col, numeric[col]) for col in numeric.columns
            ],
            correlation_heatmap=self._correlation_heatmap(target, dataset_id, numeric),
            bar_charts=[
                self._bar_chart(target, dataset_id, col, categorical[col])
                for col in categorical.columns
            ],
            category_distributions=self._category_distribution(target, dataset_id, categorical),
        )

        logger.info(
            "Generated charts for %s (numeric=%d, categorical=%d)",
            dataset_id,
            len(numeric.columns),
            len(categorical.columns),
        )
        return DatasetChartsResponse(dataset_id=dataset_id, charts=charts)

    def _load_dataframe(self, path: Path) -> pd.DataFrame:
        raw = path.read_bytes()
        encoding = detect_encoding(raw)
        try:
            sample = raw[:8192].decode(encoding, errors="replace")
        except LookupError:
            encoding = "utf-8"
            sample = raw[:8192].decode(encoding, errors="replace")
        separator = detect_separator(sample)
        return pd.read_csv(path, sep=separator, encoding=encoding)

    def _url(self, dataset_id: UUID, filename: str) -> str:
        return f"{self._url_prefix}/{dataset_id}/{filename}"

    def _save_png(self, target: Path, filename: str, fig: plt.Figure) -> Path:
        path = target / filename
        fig.tight_layout()
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        return path

    @staticmethod
    def _safe_name(name: str) -> str:
        return "".join(c if c.isalnum() else "_" for c in str(name))[:60] or "col"

    def _histogram(
        self, target: Path, dataset_id: UUID, column: str, series: pd.Series
    ) -> ColumnChart:
        safe = self._safe_name(column)
        clean_series = series.dropna()

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(clean_series, bins=20, edgecolor="black")
        ax.set_title(f"Histogram: {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Frequency")
        self._save_png(target, f"histogram_{safe}.png", fig)

        html_path = target / f"histogram_{safe}.html"
        px.histogram(clean_series, x=column, nbins=20, title=f"Histogram: {column}").write_html(
            html_path, include_plotlyjs="cdn"
        )

        return ColumnChart(
            column=column,
            png_url=self._url(dataset_id, f"histogram_{safe}.png"),
            html_url=self._url(dataset_id, f"histogram_{safe}.html"),
        )

    def _boxplot(
        self, target: Path, dataset_id: UUID, column: str, series: pd.Series
    ) -> ColumnChart:
        safe = self._safe_name(column)
        clean_series = series.dropna()

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.boxplot(clean_series, vert=True)
        ax.set_title(f"Boxplot: {column}")
        ax.set_ylabel(column)
        self._save_png(target, f"boxplot_{safe}.png", fig)

        html_path = target / f"boxplot_{safe}.html"
        px.box(clean_series, y=column, title=f"Boxplot: {column}").write_html(
            html_path, include_plotlyjs="cdn"
        )

        return ColumnChart(
            column=column,
            png_url=self._url(dataset_id, f"boxplot_{safe}.png"),
            html_url=self._url(dataset_id, f"boxplot_{safe}.html"),
        )

    def _correlation_heatmap(
        self, target: Path, dataset_id: UUID, numeric: pd.DataFrame
    ) -> ChartArtifact | None:
        if numeric.shape[1] < 2:
            return None

        matrix = numeric.corr(numeric_only=True)

        fig, ax = plt.subplots(figsize=(max(5, matrix.shape[0]), max(4, matrix.shape[0])))
        image = ax.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(matrix.shape[0]))
        ax.set_yticks(range(matrix.shape[0]))
        ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
        ax.set_yticklabels(matrix.columns)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix.iat[i, j]:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(image, ax=ax)
        ax.set_title("Correlation heatmap")
        self._save_png(target, "correlation_heatmap.png", fig)

        html_path = target / "correlation_heatmap.html"
        px.imshow(
            matrix,
            title="Correlation heatmap",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            text_auto=".2f",
        ).write_html(html_path, include_plotlyjs="cdn")

        return ChartArtifact(
            png_url=self._url(dataset_id, "correlation_heatmap.png"),
            html_url=self._url(dataset_id, "correlation_heatmap.html"),
        )

    def _bar_chart(
        self, target: Path, dataset_id: UUID, column: str, series: pd.Series
    ) -> ColumnChart:
        safe = self._safe_name(column)
        counts = series.dropna().astype(str).value_counts().head(_TOP_N_CATEGORICAL)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(counts.index.astype(str), counts.values)
        ax.set_title(f"Top {_TOP_N_CATEGORICAL} values: {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=45)
        self._save_png(target, f"bar_{safe}.png", fig)

        html_path = target / f"bar_{safe}.html"
        px.bar(
            x=counts.index.astype(str),
            y=counts.values,
            labels={"x": column, "y": "Count"},
            title=f"Top {_TOP_N_CATEGORICAL} values: {column}",
        ).write_html(html_path, include_plotlyjs="cdn")

        return ColumnChart(
            column=column,
            png_url=self._url(dataset_id, f"bar_{safe}.png"),
            html_url=self._url(dataset_id, f"bar_{safe}.html"),
        )

    def _category_distribution(
        self, target: Path, dataset_id: UUID, categorical: pd.DataFrame
    ) -> ChartArtifact | None:
        if categorical.shape[1] == 0:
            return None

        unique_counts = pd.Series(
            {str(col): int(categorical[col].nunique(dropna=True)) for col in categorical.columns}
        ).sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(unique_counts.index, unique_counts.values)
        ax.set_title("Unique values per categorical column")
        ax.set_xlabel("Column")
        ax.set_ylabel("Unique values")
        ax.tick_params(axis="x", rotation=45)
        self._save_png(target, "category_distribution.png", fig)

        html_path = target / "category_distribution.html"
        px.bar(
            x=unique_counts.index,
            y=unique_counts.values,
            labels={"x": "Column", "y": "Unique values"},
            title="Unique values per categorical column",
        ).write_html(html_path, include_plotlyjs="cdn")

        return ChartArtifact(
            png_url=self._url(dataset_id, "category_distribution.png"),
            html_url=self._url(dataset_id, "category_distribution.html"),
        )
