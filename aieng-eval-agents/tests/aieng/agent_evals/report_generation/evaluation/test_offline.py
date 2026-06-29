"""Tests for the offline evaluation of the report generation agent."""

from pathlib import Path
from unittest.mock import ANY, Mock, patch

import pytest
from aieng.agent_evals.report_generation.evaluation.offline import (
    evaluate,
    final_result_evaluator,
    report_data_match_grader,
    report_schema_grader,
    trajectory_evaluator,
)


@patch("aieng.agent_evals.report_generation.evaluation.offline.AsyncClientManager.get_instance")
@patch("aieng.agent_evals.report_generation.evaluation.offline.DbManager.get_instance")
@pytest.mark.asyncio
async def test_evaluate(mock_db_manager_instance, mock_async_client_manager_instance):
    """Test the evaluate function."""
    test_dataset_name = "test_dataset"
    test_reports_output_path = Path("reports/")
    test_max_concurrency = 5

    mock_result = Mock()
    mock_dataset = Mock()
    mock_dataset.run_experiment.return_value = mock_result
    mock_langfuse_client = Mock()
    mock_langfuse_client.get_dataset.return_value = mock_dataset
    mock_async_client_manager_instance.return_value = Mock()
    mock_async_client_manager_instance.return_value.langfuse_client = mock_langfuse_client

    mock_db_manager_instance.return_value = Mock()

    await evaluate(
        dataset_name=test_dataset_name,
        reports_output_path=test_reports_output_path,
        max_concurrency=test_max_concurrency,
    )

    mock_dataset.run_experiment.assert_called_once_with(
        name="Evaluate Report Generation Agent",
        description="Evaluate the Report Generation Agent with data from Langfuse",
        task=ANY,
        evaluators=[
            final_result_evaluator,
            trajectory_evaluator,
            report_data_match_grader,
            report_schema_grader,
        ],
        max_concurrency=test_max_concurrency,
    )

    task = mock_dataset.run_experiment.call_args_list[0][1]["task"]
    assert task.__name__ == "run"
    assert task.__self__.__class__.__name__ == "ReportGenerationTask"
    assert task.__self__.reports_output_path == test_reports_output_path

    mock_db_manager_instance.return_value.close.assert_called_once()
    mock_async_client_manager_instance.return_value.close.assert_called_once()


def _evaluations_by_name(evaluations):
    """Map a list of Evaluation objects by their name for easy assertions."""
    return {evaluation.name: evaluation for evaluation in evaluations}


def test_report_data_match_grader_exact_match():
    """An identical report (different row order) should be an exact match."""
    expected_output = {
        "final_report": {
            "report_data": [["2011-01", 100.0], ["2011-02", 200.0]],
            "report_columns": ["Month", "Sales"],
            "filename": "report.xlsx",
        }
    }
    output = {
        "final_report": {
            "report_data": [["2011-02", 200.004], ["2011-01", 99.999]],
            "report_columns": ["Month", "Sales"],
            "filename": "report.xlsx",
        }
    }

    results = _evaluations_by_name(
        report_data_match_grader(input="q", output=output, expected_output=expected_output)
    )

    assert results["Data Exact Match"].value is True
    assert results["Row Count Match"].value is True
    assert results["Row Overlap"].value == 1.0


def test_report_data_match_grader_treats_empty_cells_as_zero():
    """None and empty-string cells should be treated as 0."""
    expected_output = {
        "final_report": {
            "report_data": [["2010-12", 748957.02, 0, 0]],
        }
    }
    output = {
        "final_report": {
            "report_data": [["2010-12", 748957.02, None, ""]],
        }
    }

    results = _evaluations_by_name(
        report_data_match_grader(input="q", output=output, expected_output=expected_output)
    )

    assert results["Data Exact Match"].value is True


def test_report_data_match_grader_partial_overlap():
    """A partial match should report overlap but not an exact match."""
    expected_output = {
        "final_report": {
            "report_data": [["a", 1.0], ["b", 2.0]],
        }
    }
    output = {
        "final_report": {
            "report_data": [["a", 1.0], ["b", 9.0]],
        }
    }

    results = _evaluations_by_name(
        report_data_match_grader(input="q", output=output, expected_output=expected_output)
    )

    assert results["Data Exact Match"].value is False
    assert results["Row Count Match"].value is True
    assert results["Row Overlap"].value == 0.5


def test_report_data_match_grader_missing_report():
    """A missing report (no write_xlsx) should score zero."""
    expected_output = {
        "final_report": {
            "report_data": [["a", 1.0]],
        }
    }
    output = {"final_report": None}

    results = _evaluations_by_name(
        report_data_match_grader(input="q", output=output, expected_output=expected_output)
    )

    assert results["Data Exact Match"].value is False
    assert results["Row Count Match"].value is False
    assert results["Row Overlap"].value == 0.0


def test_report_schema_grader_valid():
    """Matching column count and a .xlsx filename should pass both checks."""
    expected_output = {
        "final_report": {
            "report_columns": ["Month", "Sales"],
        }
    }
    output = {
        "final_report": {
            "report_columns": ["SalesMonth", "TotalSales"],
            "filename": "monthly_report.xlsx",
        }
    }

    results = _evaluations_by_name(
        report_schema_grader(input="q", output=output, expected_output=expected_output)
    )

    assert results["Column Count Match"].value is True
    assert results["Valid Filename"].value is True


def test_report_schema_grader_invalid():
    """A mismatched column count and non-xlsx filename should fail both checks."""
    expected_output = {
        "final_report": {
            "report_columns": ["Month", "Sales"],
        }
    }
    output = {
        "final_report": {
            "report_columns": ["Month"],
            "filename": "report.csv",
        }
    }

    results = _evaluations_by_name(
        report_schema_grader(input="q", output=output, expected_output=expected_output)
    )

    assert results["Column Count Match"].value is False
    assert results["Valid Filename"].value is False


def test_report_schema_grader_missing_report():
    """A missing report should fail both checks."""
    expected_output = {
        "final_report": {
            "report_columns": ["Month", "Sales"],
        }
    }
    output = {"final_report": None}

    results = _evaluations_by_name(
        report_schema_grader(input="q", output=output, expected_output=expected_output)
    )

    assert results["Column Count Match"].value is False
    assert results["Valid Filename"].value is False
