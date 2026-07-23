"""Unit tests for ChartService."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID, uuid4

import pytest

from app.core.exceptions import DatasetNotFoundError
from app.repositories.dataset_repository import DatasetRepository
from app.services.chart_service import ChartService

_MIXED_CSV = (
    "age,salary,city\n30,3000,Lisbon\n25,2500,Porto\n40,5000,Lisbon\n35,4200,Braga\n28,2700,Porto\n"
)


def _store_csv(repository: DatasetRepository, content: str) -> UUID:
    dataset_id = uuid4()
    repository.save(dataset_id, "sample.csv", content.encode("utf-8"))
    return dataset_id


def test_generate_returns_expected_chart_groups(
    dataset_repository: DatasetRepository,
    chart_service: ChartService,
    temp_storage: Path,
) -> None:
    dataset_id = _store_csv(dataset_repository, _MIXED_CSV)

    response = chart_service.generate(dataset_id)

    assert response.dataset_id == dataset_id
    assert len(response.charts.histograms) == 2
    assert len(response.charts.boxplots) == 2
    assert response.charts.correlation_heatmap is not None
    assert len(response.charts.bar_charts) == 1
    assert response.charts.category_distributions is not None


def test_generate_writes_png_and_html_files(
    dataset_repository: DatasetRepository,
    chart_service: ChartService,
    temp_storage: Path,
) -> None:
    dataset_id = _store_csv(dataset_repository, _MIXED_CSV)

    chart_service.generate(dataset_id)

    charts_root = temp_storage / "charts" / str(dataset_id)
    files = {p.name for p in charts_root.iterdir()}

    assert "histogram_age.png" in files
    assert "histogram_age.html" in files
    assert "boxplot_salary.png" in files
    assert "correlation_heatmap.png" in files
    assert "correlation_heatmap.html" in files
    assert "bar_city.png" in files
    assert "category_distribution.png" in files


def test_generate_produces_expected_url_shape(
    dataset_repository: DatasetRepository,
    chart_service: ChartService,
) -> None:
    dataset_id = _store_csv(dataset_repository, _MIXED_CSV)

    response = chart_service.generate(dataset_id)

    prefix = f"/static/charts/{dataset_id}/"
    assert response.charts.histograms[0].png_url.startswith(prefix)
    assert response.charts.histograms[0].html_url.startswith(prefix)
    assert response.charts.correlation_heatmap.png_url.startswith(prefix)


def test_generate_omits_heatmap_when_less_than_two_numeric_columns(
    dataset_repository: DatasetRepository,
    chart_service: ChartService,
) -> None:
    dataset_id = _store_csv(dataset_repository, "age,city\n30,Lisbon\n25,Porto\n")

    response = chart_service.generate(dataset_id)

    assert response.charts.correlation_heatmap is None


def test_generate_omits_category_distribution_when_no_categorical_columns(
    dataset_repository: DatasetRepository,
    chart_service: ChartService,
) -> None:
    dataset_id = _store_csv(dataset_repository, "a,b\n1,2\n3,4\n5,6\n")

    response = chart_service.generate(dataset_id)

    assert response.charts.category_distributions is None
    assert response.charts.bar_charts == []


def test_generate_raises_for_unknown_dataset(chart_service: ChartService) -> None:
    with pytest.raises(DatasetNotFoundError):
        chart_service.generate(uuid4())
