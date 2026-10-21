# Clarification and scope: MVP checks

Reuse existing agent tests; no additional evaluator, judge or dataset is needed.

## Covered offline

| Scenario | Existing test location |
| --- | --- |
| Missing comparison details pause before retrieval; reply resumes with context | `tests/test_finance_autonomous_agent.py::test_parent_clarification_interrupt_resumes_inside_agent` |
| Out-of-scope result can finish without search | `tests/test_finance_autonomous_agent.py::test_model_can_report_out_of_scope_without_searching` |
| “Submit a new question” guidance does not create another clarification interrupt | `tests/test_finance_autonomous_agent.py::test_new_question_guidance_does_not_open_another_clarification` |
| A general clarification followed by a changed unsupported task finishes without search | `tests/test_finance_autonomous_agent.py::test_changed_task_after_general_clarification_stops_without_search` |
| Missing/invalid calculator inputs, repair, three-reply limit, unsupported deposits, persistent resume | `tests/test_calculation_agent.py` |
| Changed task after calculation clarification yields out_of_scope | `tests/test_calculation_agent.py::test_changed_task_after_calculation_clarification_is_out_of_scope` |
| Simple-interest clarification limit and unsupported APY | `tests/test_simple_interest_agent.py` |

```bash
pytest -q tests/test_finance_autonomous_agent.py tests/test_calculation_agent.py \
  tests/test_simple_interest_agent.py
```

## Limits and future evaluation

Models are scripted: these checks validate real runtime execution, checkpointed
clarification and handling of scope decisions. They do not establish that the
actual LLM notices every missing detail or classifies every unsupported request
correctly. Unsupported calculation features are validated by the tool, while
asking the user whether to omit them is model-controlled.

A future on-demand live evaluation can check whether the model asks only necessary
questions, preserves the original task, avoids invented inputs, and recognizes
changed/out-of-scope requests. Reuse representative user questions and capture
status, clarification text, tool calls and inputs in the eventual finance report.
No live evaluation, new production routing policy or reporting framework is added
by this step.
