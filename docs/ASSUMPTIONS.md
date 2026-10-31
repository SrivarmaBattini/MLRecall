# Assumptions register (version 1)

Every number below is a design choice, not a measured fact about human memory.
Each one lives in `config.py`, is tested by a script, and should be listed in your report.
"Test" says how to check it on your own machine.

| # | Assumption | Where | Test |
|---|---|---|---|
| A1 | A quiz has 5 multiple-choice questions | `MemoryRules.n_questions` | `python -m rl.validate_rules` (part 1: how noisy 5 questions are) |
| A2 | The first quiz sets the starting level: 0-1 right = level 0, 2-3 = 1, 4 = 2, 5 = 3 | `first_quiz_min_correct` | `validate_rules` part 2 |
| A3 | A review quiz is a success at 4 or more right out of 5; success = level +1, failure = level -1 | `review_success_min_correct` | `validate_rules` parts 1 and 3. Change the threshold and rerun to see sensitivity |
| A4 | Level multipliers are fixed at 1, 2, 4, 8 (never estimated) | `level_multipliers` | `validate_rules --world power` and `validate_estimator --world power` (mismatched curve) |
| A5 | Topic base half-life by difficulty: easy 8 days, medium 4, hard 2 (set by the author in each lesson's front matter) | `base_half_life`, `difficulty:` in `content/topics/*.md` | Change the values, regenerate the dataset, rerun the validators |
| A6 | Learner speed `s_j` has a log-normal prior (sigma 0.5), clamped to 0.25-4, estimated by MAP from quiz counts | `speed_prior_sigma`, `speed_bounds` | `python -m rl.validate_estimator` (bias and error as reviews accumulate) |
| A7 | A learner who knows an item answers wrongly 10% of the time; a learner who does not know it guesses right 25% of the time (4 options) | `quiz_slip`, `quiz_guess` | `validate_rules` part 3 (expected vs observed) |
| A8 | Forgetting is exponential: recall = 2^(-days / half-life) | `rl/memory_model.py` | `--world power` runs use a different (power-law) curve |
| A9 | The quiz taken right after reading measures comprehension, not long-term recall; it only sets a starting level | `apply_quiz` | Design assumption. First reviews correct it quickly |
| A10 | A same-day repeat quiz is stored as practice and changes neither level nor speed estimate | `apply_quiz` | `tests/test_memory_model.py` |
| A11 | Any completed quiz counts as a review, whether or not the agent recommended it | `backend/service.py` | `tests/test_api.py` |
| A12 | The agent maximises the discounted sum of estimated recall over all studied topics, each topic weighted equally | `rl/simulator.py` | Design choice |
| A13 | Half-life is discretised into 12 log-spaced buckets (0.5-40 days); days since quiz is capped at 40 | `AgentConfig` | `rl.verify_agent` (index accuracy per bucket) |
| A14 | The agent is trained only in simulation. QWI aims at the Whittle index; how close it gets is measured, not guaranteed | `rl/qwi.py` | `python -m rl.verify_agent` |
| A15 | The Whittle policy computed with known dynamics is a reference, not a proven global optimum. Indexability is measured per bucket, not assumed | `rl/whittle_reference.py` | `rl.verify_agent` section 1 |

## Known limits to state in the report

- **The simulated learners and the demo account are simulated.** Results from them show how the method behaves under the model, not that real learners retain more.
- **Early estimates are weak.** With only a few review quizzes the speed estimate stays close to its prior; check the bias and error table from `validate_estimator`.
- **The model can be over-pessimistic about topics it believes are nearly forgotten.** Check part 3 of `validate_rules` for the gap between expected and observed.
- **A mismatched forgetting curve increases miscalibration.** Compare the `exp` and `power` runs.
- **A 5-question quiz is a noisy measurement.** Part 1 of `validate_rules` shows how often a learner with a given true recall passes.
- **The agent only sees quizzes.** Reading or watching a lesson never counts as evidence of learning.
