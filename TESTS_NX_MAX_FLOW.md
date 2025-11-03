# Nx-Max-Flow Test Coverage and De-duplication

This document summarizes how tests for the `nx-max-flow` backend of `maximin-support` are covered across the suite, and which redundant tests were removed.

1) What it test: 
        - Correctness
        - Resolute/irresolute
        - Diff committee sizes
        - Disjoint approval blocs
    Where:
        - `test_abcrules.py::test_abcrules_correct`
        - `test_abcrules.py::test_abcrules_correct_with_max_num_of_committees`

2) What it test: 
        - Single votes
        - Tiny profiles
    Where:
        - `test_abcrules.py::test_abcrules_correct_simple`
        - `test_abcrules.py::test_abcrules_correct_simple2`

3) What it test: 
        - Weighted cases
    Where:
        - `test_abcrules.py::test_abcrules_weightsconsidered`
        - `test_abcrules.py::test_converted_to_weighted_abc_yaml_instances`

4) What it test:
        - Empty approval ballots
        - All-empty ballots behavior (irresolute committees still valid size)
    Where:
        - `test_abcrules.py::test_abcrules_handling_empty_ballots`

5) What it test: 
        - `Profile(0)` and negative candidate counts raise `ValueError`
        - `Profile(num_cand, cand_names)` length mismatch raises.
    Where:
        - `test_preferences.py::test_invalidprofiles`

6) What it test:
        - Voter weight <= 0 raises `ValueError`
   Where:
        - `test_preferences.py::test_invalidweights`

7) What it test:
        - committee size <= 0 raises `ValueError`
   Where:
        - `test_abcrules.py::test_maximin_support_nx_invalid_committeesize`

8) What it test:
        - `resolute=True` is incompatible with `max_num_of_committees > 1`
   Where:
        - `test_abcrules.py::test_resolute_and_max_num_of_committees`

9) What it test:
        - MMS winners satisfy the following properties (with YAML instances):
            - Justified Representation (JR)
            - Propotional Justified Representation (PJR)
            - Priceability
   Where:
        - `test_properties.py::test_properties_with_rules`

10) What it test:
        - Solver consistency (brute-force vs gurobi for properties)
        - MMS output agreement with YAML/other backends
   Where:
        - `test_properties.py::test_matching_output_different_approaches_and_implications`
        - `test_abcrules.py::test_selection_of_abc_yaml_instances`



## Explicit nx-max-flow (marked `@pytest.mark.networkx` and thus included when networkx tests are not disabled):

- Random large profiles for `maximin-support` using `nx-max-flow`:
  - `test_abcrules.py::test_maximin_support_nx_random_profiles`

- Error handling and parameter validation for `nx-max-flow`:
  - `test_abcrules.py::test_maximin_support_nx_invalid_committeesize`
  - `test_abcrules.py::test_maximin_support_nx_invalid_committeesize_exceeds_candidates`


## Generic tests that also exercise nx-max-flow (not excluded)

- Includes `nx-max-flow` via `MARKS`:
  - `test_abcrules.py::test_abcrules_toofewcandidates`
  - `test_abcrules.py::test_abcrules_noapprovedcandidates`
  - `test_abcrules.py::test_abcrules_weightsconsidered`
  - `test_abcrules.py::test_abcrules_correct_simple`
  - `test_abcrules.py::test_abcrules_correct_simple2`
  - `test_abcrules.py::test_abcrules_return_lists_of_sets`
  - `test_abcrules.py::test_abcrules_handling_empty_ballots`
  - `test_abcrules.test_output`
  - `test_abcrules.test_resolute_and_max_num_of_committees`
  - `test_abcrules.test_max_num_of_committees`

- Includes `nx-max-flow` via ABC YAML selections/ Weighted Conversions:
  - `test_abcrules.py::test_selection_of_abc_yaml_instances`
  - `test_abcrules.py::test_converted_to_weighted_abc_yaml_instances`

Exclude `nx-max-flow`:
  - `test_abcrules.test_natural_tiebreaking_order_resolute`
  - `test_abcrules.test_natural_tiebreaking_order_max_num_of_committees`
  - `test_abcrules.test_lexicographic_tiebreaking_with_toofewcandidates`

Executing Pytests:
    - Exclude mip, ortools, gmpy2 and slow marks. (as instructed in README.md)
        `pytest -m 'not mip and not ortools and not gmpy2 and not slow' `
    - Exclude mip, ortools, gmpy2, slow and networkx marks.
        `pytest -m 'not mip and not ortools and not gmpy2 and not slow and not networkx' `