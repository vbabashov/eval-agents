# Plan: Add two code-based evals to report_generation

## Goal
Add two deterministic (pure-code, no LLM) evaluators to the report_generation
offline evaluation, complementing the existing two LLM-judge evaluators
(`final_result_evaluator`, `trajectory_evaluator`). They reuse the existing
ground-truth dataset `OnlineRetailReportEval.json` with no dataset changes.

## Key facts
- Evaluators live in:
  aieng-eval-agents/aieng/agent_evals/report_generation/evaluation/offline.py
- Item-level evaluator signature (Langfuse EvaluatorFunction):
  `async def fn(*, input: str, output: EvaluationOutput, expected_output: EvaluationOutput, **kwargs) -> Evaluation | list[Evaluation]`
- output["final_report"] = args passed to write_xlsx tool = dict with
  {report_data: list[list], report_columns: list[str], filename: str}; can be None if write_xlsx never called.
- expected_output["final_report"] same shape. Also expected_output may have
  "additional_instructions" (sample id 4 special tolerance for first-row MoM).
- Rubric tolerance used by LLM judge: numeric ±0.01; row order irrelevant.
- Evaluators registered in dataset.run_experiment(evaluators=[...]) in offline.py.
- Test asserting the evaluators list: tests/.../report_generation/evaluation/test_offline.py
  (test_evaluate asserts evaluators=[final_result_evaluator, trajectory_evaluator]).

## The two new evals (deterministic, item-level)
1. `report_data_match_grader` — order-insensitive, ±0.01 tolerance comparison of
   output report_data vs expected report_data. Normalize rows: round floats to 2
   decimals, coerce None/"" to 0 for numeric-ish cells, compare as multisets.
   Emit: Evaluation "Data Exact Match" (bool), Evaluation "Row Count Match" (bool).
   Handle output["final_report"] is None -> value False.
2. `report_schema_grader` — checks report_columns count matches expected and
   filename ends with ".xlsx". Emit: Evaluation "Column Count Match" (bool),
   Evaluation "Valid Filename" (bool). Handle None -> False.

## Files to modify
- offline.py: add the two grader functions (or a sibling code_graders.py imported in);
  add them to evaluators=[...] list in evaluate().
- test_offline.py: update evaluators list assertion + add unit tests for new graders.
- Optionally notebook 03_Running_Offline_Evaluations.ipynb: mention new metrics.

## Verification
- Run pytest on tests/.../report_generation/evaluation/test_offline.py
- get_errors on edited files
- (optional) full offline run via `python -m implementations.report_generation.evaluate`

## STATUS: IMPLEMENTED + TESTED (8 passed)
- Added report_data_match_grader (Data Exact Match, Row Count Match, Row Overlap)
  and report_schema_grader (Column Count Match, Valid Filename) in offline.py.
- Helpers _normalize_cell / _normalize_rows; Counter import added.
- Registered both in evaluators=[...]; updated test_offline.py assertion + 7 new tests.

## Open decisions
- Placement: inline in offline.py vs new module. Recommend inline for minimal footprint.
- None/"" coercion for sample-4 special first-row case.
