"""
Policies (B0-B3).

CONTRACT
--------
A policy is a callable: (Observation) -> (action, delay_hours).

It may read ONLY the Observation it is handed. Nothing in this package may
import HiddenState, read a `hidden` attribute, or touch a record's ground
truth -- doing so voids every number the benchmark produces.

tests/test_no_leakage.py enforces this by AST inspection of every module in
this package, so the rule is checked, not merely documented.
"""
