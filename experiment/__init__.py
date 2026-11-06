"""Scaled generated-code reliability study of AutoML-Agent.

A thin, logging-only harness around the UNMODIFIED AutoML-Agent system. It runs
the agent at its full as-designed configuration across a grid of LLM backends,
tabular datasets, and repetitions, archiving every generated script and recording
every execution attempt, then computes failure-rate / taxonomy / semantic metrics.

Note: this package is `experiment/` (singular) and is deliberately separate from
the pre-existing `experiments/` (plural) package in this repo.
"""
