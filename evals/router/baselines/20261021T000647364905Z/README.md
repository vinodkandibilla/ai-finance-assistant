# Initial live router baseline

Run date: October 20, 2026 (America/Phoenix); directory timestamp is UTC.
Model: `gpt-4o-mini`. One run per mode, 63 labeled cases, current prompt unchanged.
Offline verification: 197 tests passed.

| Mode | Passed | Failed | Errors | Skipped | Accuracy | Critical accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| Classifier | 54 | 1 | 0 | 8 | 98.18% | 100% (37/37) |
| Routes | 62 | 1 | 0 | 0 | 98.41% | 100% (43/43) |

Both modes misclassified `clarify_how_much` ("How much should I invest?") as
`goals`, expected `clarify`. Route mode selected `goals_node`, expected
`clarify_node`. Review this ambiguous label and prediction before any prompt
change; the dataset has not been relabeled to improve the score.

Reports include full predictions and configuration hashes. `prompt.txt` and
`cases.jsonl` preserve inputs. `source.json` records the commit and dirty working
tree; `source-snapshot.json` preserves relevant uncommitted evaluation/router
code. Progress JSON files preserve intermediate results.

The earlier sibling run `20261021T000618591330Z` stopped after a sandbox network
failure on its first call. It is an incomplete attempt, not an accuracy baseline.
The completed run used network access outside the sandbox and had no evaluation
errors. Provider integration emitted Pydantic serialization warnings; validated
classification results were returned successfully.

This single-run baseline is a comparison reference, not an absolute quality
threshold or proof of stable accuracy. Repeat both variants when assessing small
prompt improvements. No router prompt changes were made.
