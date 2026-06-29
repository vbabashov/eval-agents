"""
Evaluate the report generation agent against a Langfuse dataset.

Example
-------
>>> from aieng.agent_evals.report_generation.evaluation.offline import evaluate
>>> evaluate(
>>>     dataset_name="OnlineRetailReportEval",
>>>     reports_output_path=Path("reports/"),
>>> )
"""

import logging
from collections import Counter
from pathlib import Path
from typing import Any

from aieng.agent_evals.async_client_manager import AsyncClientManager
from aieng.agent_evals.db_manager import DbManager
from aieng.agent_evals.report_generation.agent import EventParser, EventType, get_report_generation_agent
from aieng.agent_evals.report_generation.prompts import (
    MAIN_AGENT_INSTRUCTIONS,
    RESULT_EVALUATOR_INSTRUCTIONS,
    RESULT_EVALUATOR_TEMPLATE,
    TRAJECTORY_EVALUATOR_INSTRUCTIONS,
    TRAJECTORY_EVALUATOR_TEMPLATE,
)
from google.adk.agents import Agent
from google.adk.events.event import Event
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import Client
from google.genai.types import Content, GenerateContentConfig, Part
from langfuse._client.datasets import DatasetItemClient
from langfuse.experiment import Evaluation, LocalExperimentItem
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Will have the structure:
# {
#     "final_report": str | None,
#     "trajectory": {
#         "actions": list[str],
#         "parameters": list[str],
#     },
# }
EvaluationOutput = dict[str, None | Any]


class EvaluatorResponse(BaseModel):
    """Typed response from the evaluator."""

    explanation: str
    is_answer_correct: bool


async def evaluate(
    dataset_name: str,
    reports_output_path: Path,
    max_concurrency: int = 5,
) -> None:
    """Evaluate the report generation agent against a Langfuse dataset.

    The database connection to the report generation database is obtained
    from the environment variable `REPORT_GENERATION_DB__DATABASE`.

    Parameters
    ----------
    dataset_name : str
        Name of the Langfuse dataset to evaluate against.
    reports_output_path : Path
        The path to the reports output directory.
    langfuse_project_name : str
        The name of the Langfuse project to use for tracing.
    max_concurrency : int, optional
        The maximum concurrency to use for the evaluation, by default 5.
    """
    # Get the client manager singleton instance and langfuse client
    client_manager = AsyncClientManager.get_instance()
    langfuse_client = client_manager.langfuse_client

    # Find the dataset in Langfuse
    dataset = langfuse_client.get_dataset(dataset_name)

    # Initialize the task for the report generation agent evaluation
    # We need this task so we can pass parameters to the agent, since
    # the agent has to be instantiated inside the task function
    report_generation_task = ReportGenerationTask(reports_output_path=reports_output_path)

    # Run the experiment with the agent task and evaluator
    # against the dataset items
    result = dataset.run_experiment(
        name="Evaluate Report Generation Agent",
        description="Evaluate the Report Generation Agent with data from Langfuse",
        task=report_generation_task.run,
        evaluators=[
            final_result_evaluator,
            trajectory_evaluator,
            report_data_match_grader,
            report_schema_grader,
        ],
        max_concurrency=max_concurrency,
    )

    # Log the evaluation result
    logger.info(result.format().replace("\\n", "\n"))

    try:
        # Gracefully close the services
        DbManager.get_instance().close()
        await client_manager.close()
    except Exception as e:
        logger.warning(f"Client manager services not closed successfully: {e}")


class ReportGenerationTask:
    """Define a task for the the report generation agent."""

    def __init__(self, reports_output_path: Path):
        """Initialize the task for an report generation agent evaluation.

        Parameters
        ----------
        reports_output_path : Path
            The path to the reports output directory.
        """
        self.reports_output_path = reports_output_path

    async def run(self, *, item: LocalExperimentItem | DatasetItemClient, **kwargs: dict[str, Any]) -> EvaluationOutput:
        """Run the report generation agent against an item from a Langfuse dataset.

        Parameters
        ----------
        item : LocalExperimentItem | DatasetItemClient
            The item from the Langfuse dataset to evaluate against.

        Returns
        -------
        EvaluationOutput
            The output of the report generation agent with the values it should
            be evaluated against.
        """
        # Run the report generation agent
        report_generation_agent = get_report_generation_agent(
            instructions=MAIN_AGENT_INSTRUCTIONS,
            reports_output_path=self.reports_output_path,
        )
        # Handle both TypedDict and class access patterns
        item_input = item["input"] if isinstance(item, dict) else item.input
        events = await run_agent_with_retry(report_generation_agent, item_input)

        # Extract the report data and trajectory from the agent's response
        actions: list[str] = []
        parameters: list[Any | None] = []
        final_report: str | None = None

        # The trajectory will be the list of actions and the
        # parameters passed to each one of them
        for event in events:
            parsed_events = EventParser.parse(event)

            for parsed_event in parsed_events:
                if parsed_event.type == EventType.FINAL_RESPONSE:
                    # Picking up the final message displayed to the user
                    actions.append("final_response")
                    parameters.append(parsed_event.text)

                if parsed_event.type == EventType.TOOL_CALL:
                    # Picking up tool calls and their arguments
                    actions.append(parsed_event.text)
                    parameters.append(parsed_event.arguments)

                    # The final report will be the arguments sent by the
                    # write_xlsx tool call
                    # If there is more than one call to the write_xlsx tool call,
                    # the last one will be used because the previous
                    # calls are likely failed calls
                    if parsed_event.text == "write_xlsx":
                        final_report = parsed_event.arguments

                # Not tracking EventType.THOUGHT or EventType.TOOL_RESPONSE

        if final_report is None:
            logger.warning("No call to `write_xlsx` function found in the agent's response")

        return {
            "final_report": final_report,
            "trajectory": {
                "actions": actions,
                "parameters": parameters,
            },
        }


async def final_result_evaluator(
    *,
    input: str,
    output: EvaluationOutput,
    expected_output: EvaluationOutput,
    **kwargs,
) -> Evaluation:
    # ruff: noqa: A002
    """Evaluate the proposed final answer against the ground truth.

    Uses LLM-as-a-judge and returns the reasoning behind the answer.

    Parameters
    ----------
    input : str
        The input to the report generation agent.
    output : EvaluationOutput
        The output of the report generation agent with the values it should be
        evaluated against.
    expected_output : EvaluationOutput
        The evaluation output the report generation agent should have.
    kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    Evaluation
        The evaluation result, including the reasoning behind the answer.
    """
    # Define the evaluator agent
    client_manager = AsyncClientManager.get_instance()

    # Format the input for the evaluator agent
    evaluator_input = RESULT_EVALUATOR_TEMPLATE.format(
        question=input,
        ground_truth=expected_output["final_report"],
        proposed_response=output["final_report"],
    )

    # Get the additional evaluation instructions if it
    # exists for this specific sample
    additional_instructions = _get_additional_instructions(expected_output, "final_report")

    client = Client()
    response = client.models.generate_content(
        model=client_manager.configs.default_evaluator_model,
        contents=evaluator_input,
        config=GenerateContentConfig(
            system_instruction=RESULT_EVALUATOR_INSTRUCTIONS + additional_instructions,
            response_mime_type="application/json",
            response_schema=EvaluatorResponse.model_json_schema(),
        ),
    )

    # Parsing and returning the evaluation result
    assert isinstance(response.parsed, dict), f"response.parsed must be a dictionary: {response.parsed}"
    evaluator_response = EvaluatorResponse(**response.parsed)

    return Evaluation(
        name="Final Result",
        value=evaluator_response.is_answer_correct,
        comment=evaluator_response.explanation,
    )


async def trajectory_evaluator(
    *,
    input: str,
    output: EvaluationOutput,
    expected_output: EvaluationOutput,
    **kwargs,
) -> Evaluation:
    # ruff: noqa: A002
    """Evaluate the agent's trajectory against the ground truth.

    Uses LLM-as-a-judge and returns the reasoning behind the answer.

    Parameters
    ----------
    input : str
        The input to the report generation agent.
    output : EvaluationOutput
        The output of the report generation agent with the values it should be
        evaluated against.
    expected_output : EvaluationOutput
        The evaluation output the report generation agent should have.
    kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    Evaluation
        The evaluation result, including the reasoning behind the answer.
    """
    # Define the evaluator agent
    client_manager = AsyncClientManager.get_instance()

    assert isinstance(expected_output["trajectory"], dict), "Expected trajectory must be a dictionary"
    assert isinstance(output["trajectory"], dict), "Actual trajectory must be a dictionary"

    # Format the input for the evaluator agent
    evaluator_input = TRAJECTORY_EVALUATOR_TEMPLATE.format(
        question=input,
        expected_actions=expected_output["trajectory"]["actions"],
        expected_descriptions=expected_output["trajectory"]["description"],
        actual_actions=output["trajectory"]["actions"],
        actual_parameters=output["trajectory"]["parameters"],
    )

    # Get the additional evaluation instructions if it
    # exists for this specific sample
    additional_instructions = _get_additional_instructions(expected_output, "trajectory")

    client = Client()
    response = client.models.generate_content(
        model=client_manager.configs.default_evaluator_model,
        contents=evaluator_input,
        config=GenerateContentConfig(
            system_instruction=TRAJECTORY_EVALUATOR_INSTRUCTIONS + additional_instructions,
            response_mime_type="application/json",
            response_schema=EvaluatorResponse.model_json_schema(),
        ),
    )

    # Parsing and returning the evaluation result
    assert isinstance(response.parsed, dict), f"response.parsed must be a dictionary: {response.parsed}"
    evaluator_response = EvaluatorResponse(**response.parsed)

    return Evaluation(
        name="Trajectory",
        value=evaluator_response.is_answer_correct,
        comment=evaluator_response.explanation,
    )


def _get_additional_instructions(expected_output: EvaluationOutput, key: str) -> str:
    additional_instructions_dict = expected_output.get("additional_instructions", {})
    if additional_instructions_dict:
        return additional_instructions_dict.get(key, "")

    return ""


def _normalize_cell(value: Any) -> Any:
    """Normalize a single report cell for deterministic comparison.

    Empty values (``None`` or ``""``) are coerced to ``0`` so that valid
    placeholder values (e.g. the first row of a month-over-month change report)
    do not cause spurious mismatches. Numeric values are rounded to two decimal
    places to match the ±0.01 tolerance used by the LLM-as-a-judge evaluator.

    Parameters
    ----------
    value : Any
        The raw cell value from the report data.

    Returns
    -------
    Any
        The normalized cell value.
    """
    if value is None or value == "":
        return 0
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return round(float(value), 2)
    return value


def _normalize_rows(report_data: Any) -> list[tuple[Any, ...]] | None:
    """Normalize report rows into a comparable, order-independent form.

    Parameters
    ----------
    report_data : Any
        The ``report_data`` field from a report (a 2D array of values), or
        anything else if the report is malformed.

    Returns
    -------
    list[tuple[Any, ...]] | None
        A sorted list of normalized row tuples, or ``None`` if ``report_data``
        is not a list of rows.
    """
    if not isinstance(report_data, list):
        return None

    normalized: list[tuple[Any, ...]] = []
    for row in report_data:
        if not isinstance(row, (list, tuple)):
            return None
        normalized.append(tuple(_normalize_cell(cell) for cell in row))

    # Sort so that row order does not affect the comparison. Use the string
    # representation as the key to tolerate mixed types within a column.
    return sorted(normalized, key=lambda r: tuple(str(cell) for cell in r))


def report_data_match_grader(
    *,
    input: str,  # noqa: A002
    output: EvaluationOutput,
    expected_output: EvaluationOutput,
    **kwargs: Any,
) -> list[Evaluation]:
    """Deterministically compare report data against the ground truth.

    This is a pure-code counterpart to :func:`final_result_evaluator`. It checks
    that the agent's report data matches the expected data, ignoring row order
    and applying a ±0.01 numeric tolerance. Empty placeholder cells (``None`` or
    ``""``) are treated as ``0``.

    Parameters
    ----------
    input : str
        The input to the report generation agent (unused, kept for the
        evaluator interface).
    output : EvaluationOutput
        The output of the report generation agent.
    expected_output : EvaluationOutput
        The ground-truth evaluation output.
    kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    list[Evaluation]
        ``Data Exact Match`` (boolean), ``Row Count Match`` (boolean) and
        ``Row Overlap`` (numeric fraction of expected rows that were matched).
    """
    actual_report = output.get("final_report")
    expected_report = expected_output.get("final_report")

    actual_rows = _normalize_rows(actual_report.get("report_data")) if isinstance(actual_report, dict) else None
    expected_rows = _normalize_rows(expected_report.get("report_data")) if isinstance(expected_report, dict) else None

    if expected_rows is None:
        comment = "Expected report data is missing or malformed."
        return [
            Evaluation(name="Data Exact Match", value=False, comment=comment),
            Evaluation(name="Row Count Match", value=False, comment=comment),
            Evaluation(name="Row Overlap", value=0.0, comment=comment),
        ]

    if actual_rows is None:
        comment = "Agent produced no valid report data (no `write_xlsx` call or malformed data)."
        return [
            Evaluation(name="Data Exact Match", value=False, comment=comment),
            Evaluation(name="Row Count Match", value=False, comment=comment),
            Evaluation(name="Row Overlap", value=0.0, comment=comment),
        ]

    row_count_match = len(actual_rows) == len(expected_rows)

    # Compare as multisets so duplicate rows are accounted for correctly.
    actual_remaining = Counter(actual_rows)
    matched = 0
    for row in expected_rows:
        if actual_remaining.get(row, 0) > 0:
            actual_remaining[row] -= 1
            matched += 1

    exact_match = matched == len(expected_rows) and row_count_match
    row_overlap = matched / len(expected_rows) if expected_rows else 0.0

    return [
        Evaluation(
            name="Data Exact Match",
            value=exact_match,
            comment=f"Matched {matched}/{len(expected_rows)} expected rows (order-insensitive, ±0.01 tolerance).",
        ),
        Evaluation(
            name="Row Count Match",
            value=row_count_match,
            comment=f"Got {len(actual_rows)} rows, expected {len(expected_rows)}.",
        ),
        Evaluation(
            name="Row Overlap",
            value=row_overlap,
            comment=f"{matched}/{len(expected_rows)} expected rows present in the agent's report.",
        ),
    ]


def report_schema_grader(
    *,
    input: str,  # noqa: A002
    output: EvaluationOutput,
    expected_output: EvaluationOutput,
    **kwargs: Any,
) -> list[Evaluation]:
    """Deterministically check the report's structural properties.

    Verifies that the number of report columns matches the ground truth and that
    the output filename is a valid ``.xlsx`` file. This is a cheap, pure-code
    complement to the LLM-as-a-judge evaluators.

    Parameters
    ----------
    input : str
        The input to the report generation agent (unused, kept for the
        evaluator interface).
    output : EvaluationOutput
        The output of the report generation agent.
    expected_output : EvaluationOutput
        The ground-truth evaluation output.
    kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    list[Evaluation]
        ``Column Count Match`` (boolean) and ``Valid Filename`` (boolean).
    """
    actual_report = output.get("final_report")
    expected_report = expected_output.get("final_report")

    if not isinstance(actual_report, dict):
        comment = "Agent produced no report (no `write_xlsx` call)."
        return [
            Evaluation(name="Column Count Match", value=False, comment=comment),
            Evaluation(name="Valid Filename", value=False, comment=comment),
        ]

    actual_columns = actual_report.get("report_columns")
    expected_columns = expected_report.get("report_columns") if isinstance(expected_report, dict) else None

    if isinstance(actual_columns, list) and isinstance(expected_columns, list):
        column_count_match = len(actual_columns) == len(expected_columns)
        column_comment = f"Got {len(actual_columns)} columns, expected {len(expected_columns)}."
    else:
        column_count_match = False
        column_comment = "Report columns missing or malformed in the agent's output or the ground truth."

    filename = actual_report.get("filename")
    valid_filename = isinstance(filename, str) and filename.strip().lower().endswith(".xlsx")
    filename_comment = f"Filename is '{filename}'." if valid_filename else f"Filename '{filename}' is not a valid .xlsx file."

    return [
        Evaluation(name="Column Count Match", value=column_count_match, comment=column_comment),
        Evaluation(name="Valid Filename", value=valid_filename, comment=filename_comment),
    ]


@retry(stop=stop_after_attempt(5), wait=wait_exponential())
async def run_agent_with_retry(agent: Agent, agent_input: str) -> list[Event]:
    """Run an agent with Tenacity's retry mechanism.

    Parameters
    ----------
    agent : agents.Agent
        The agent to run.
    agent_input : str
        The input to the agent.

    Returns
    -------
    list[Event]
        The events from the agent run.
    """
    try:
        logger.info(f"Running agent {agent.name} with input '{agent_input[:100]}...'")

        # Create session and runner
        session_service = InMemorySessionService()
        runner = Runner(app_name=agent.name, agent=agent, session_service=session_service)
        current_session = await session_service.create_session(
            app_name=agent.name,
            user_id="user",
            state={},
        )

        # create the user message and run the agent
        content = Content(role="user", parts=[Part(text=agent_input)])
        events = []
        async for event in runner.run_async(
            user_id="user",
            session_id=current_session.id,
            new_message=content,
        ):
            events.append(event)

        return events
    except Exception as e:
        logger.error(f"Error running agent {agent.name} with input '{agent_input[:100]}...': {e}")
        raise e  # raising the exception so the retry mechanism can try again
