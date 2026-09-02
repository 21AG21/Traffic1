# Six Ladders: the whole apprenticeship at six levels

_September 2026. Supersedes `22_LEARNING_PATH_ML.md` and `23_LESSONS_SIX_LEVELS.md`, which it combines; companion to `21_PLAN_M1_TRAFFIC_MODEL.md`._

**The whole apprenticeship, told six times: every phase, every topic, every lesson and every book, from age five to a PhD at Apple.**

Nine phases carry you from an empty terminal to designing the next traffic model. Each phase lists what you need to learn, teaches it through lessons and topic ladders written at six levels, explains every book, course and paper at the same six levels, and ends with things to build, a checkpoint that runs, and the usual pitfalls. Pick a level with the switcher and read the whole course at that height, or turn on every level and watch one idea gain precision rung by rung.

The six levels, in order: Age 5, Middle school, High school, Stanford undergrad, Stanford grad, PhD at Apple. Every lesson, topic and resource below carries all six.

## Contents

- Phase 0 · Setup and habits (≈8 h)
  - Topic: The terminal, git and a Python environment
  - Lesson 1: The question: what are we predicting?
- Phase 1 · Python that does work (≈40 h)
  - Topic: Python: from variables to functions to files
  - Topic: Tests, debugging and time zones
  - Lesson 2: The data: detectors, five-minute rows, incidents
- Phase 2 · The math the model uses (≈60 h)
  - Topic: Linear algebra: vectors, dot products, matrices
  - Topic: Calculus: derivatives, the chain rule, gradients
  - Topic: Probability and statistics: distributions, medians, likelihood
  - Lesson 7: Logistic regression from scratch: the math the model runs on
- Phase 3 · Data handling at scale (≈50 h)
  - Lesson 3: Cleaning: from a giant text file to a tidy table
- Phase 4 · Classical machine learning (≈70 h)
  - Lesson 4: Labels: what does 'slow' mean?
  - Lesson 5: Features: turning the past into columns
  - Lesson 6: Baselines and honest evaluation
  - Lesson 8: Gradient boosted trees
- Phase 5 · Time series and forecasting (≈45 h)
  - Lesson 10: Jams move: sequences, neighbors and the road graph
  - Lesson 11: From scores to alerts: calibration, thresholds, lead time
- Phase 6 · Neural networks (≈80 h)
  - Lesson 9: Neural networks and backpropagation, on the M1
- Phase 7 · ML engineering (≈50 h)
  - Lesson 12: Shipping it on one Mac mini: memory, scheduling, monitoring
- Phase 8 · Depth (open-ended)
  - Topic: Uncertainty: quantiles, Bayesian thinking, conformal prediction
  - Topic: Theory: why learning works at all
  - Topic: Reading papers and reproducing benchmarks

## Six rules

- **The project is the syllabus.** Every topic is here because a stage needs it, and you learn it the week you build that stage. Nothing is studied 'just in case' until phase 8.
- **Build it twice.** First with the library, so it works. Then the core from scratch in numpy, the way `03_train_from_scratch.py` does, so you know what the library hides. Then back to the library, faster.
- **Evaluate before you model.** Baselines, metrics and leakage come before any algorithm past logistic regression. A number without a baseline next to it is not a result.
- **Math just in time.** Linear algebra, calculus and probability in the amounts the stage needs, derived once by hand. Depth comes in phase 8, when you know what it is for.
- **Write it down.** A lab notebook every week, a README per stage, a write-up per phase. If you cannot explain it in plain words you do not have it yet.
- **Checkpoints are things that run.** Each phase ends with code that produces a number or a file, not with a quiz.

### What really good looks like

- **Framing.** Turn a fuzzy question ('will traffic be bad?') into a unit, a target, a horizon and a metric, and say what the answer cannot mean.
- **Data.** Build the pipeline from raw files, know its failure modes (time zones, imputation, duplicates, gaps), and measure its memory and time.
- **Models.** Know a dozen models well enough to say which wins on which data and why, and to implement the core of any of them from a paper.
- **Evaluation.** Design a split that cannot leak, pick metrics the imbalance cannot fool, put a baseline next to every number, and calibrate what you ship.
- **Engineering.** Ship one implementation used by training and serving, schedule it, monitor it, retrain it, and roll it back.
- **Reading.** Read a new paper in an evening and say what it would change in stage 9 and what it would not fix.
- **Explaining.** Describe all of the above to a smart person outside the field without hiding behind jargon.

## Phase 0: Setup and habits

_weeks 0–1 · ≈8 h · builds stage 2 runs_

**Goal.** A working machine and a working routine.

### What you need to learn

- The terminal: `cd`, `ls`, `cat`, `head`, pipes, environment variables.
- Git: clone, branch, commit, push, diff, `.gitignore`; what a repository and a README are.
- Native arm64 Python on Apple silicon (miniforge or python.org), virtual environments, `pip`, VS Code, Jupyter.
- The shape of this repo: the numbered scripts, `data/`, `models/`, `.env`, and why secrets never get committed.

### Topic: The terminal, git and a Python environment

**Goal.** Talk to the computer by typing, keep every version of your work, and give this project its own toolbox.

**Age 5.**

The computer has a secret room called the terminal where you talk to it by typing instead of clicking.

Git is a magic notebook that remembers every version of your work, so if you mess something up you can go back to yesterday's page.

A Python environment is a toybox where you keep the tools for one project, so they do not get mixed up with another project's tools.

_Try this._ Ask someone to open the terminal and type `ls`. It means 'list', and it shows what is in the folder you are standing in.

**Middle school.**

The terminal runs commands: `cd` moves between folders, `ls` lists files, `cat` shows one, and `python script.py` runs a program.

Git saves snapshots called commits, each with a message saying what changed. `git log` shows the history, and GitHub is where a copy lives online.

A virtual environment (`python -m venv .venv`) is a private folder of installed packages for this project, so upgrading something for one project cannot break another.

_Try this._ Make a folder, put a text file in it, run `git init`, `git add .`, `git commit -m 'first'`. Change the file and run `git diff`. Read what it shows you.

**High school.**

Shell essentials: absolute and relative paths, pipes with `|`, redirects with `>`, environment variables (`export PEMS_USERNAME=...`), `.env` files that scripts load, `which python`.

Git's model: working tree, staging area, repository. Branches are pointers to commits. `git status` before every commit; `.gitignore` keeps `data/` and `.env` out of history forever.

Python on Apple silicon: install a native arm64 Python (miniforge or python.org), check with `python -c 'import platform; print(platform.machine())'` which must print `arm64`, then `pip install -r requirements.txt` inside the venv. VS Code needs to be pointed at that interpreter.

_Try this._ Run the collector once with `python 02_collect_events.py`, then `git status`. The `data/` folder should not appear as untracked. If it does, fix `.gitignore` before anything else.

**Stanford undergrad.**

These tools exist for reproducibility. A venv plus pinned requirements turns 'works on my machine' into 'works on any machine'; git makes every change attributable and reversible; the shell makes procedures scriptable, and a Makefile target is a shell script with declared dependencies.

Know the difference between `pip` and `conda`, between a shell variable and an exported one, and how `PATH` resolves the word `python`. Rosetta versus native is visible through `arch` and `platform.machine()`; native wheels exist for numpy, LightGBM and PyTorch.

Git workflow even when alone: feature branches, small commits, and a pull request you review yourself. Reading your own diff before merging catches most mistakes.

_Try this._ Write a Makefile with `setup`, `collect` and `test` targets, where `test` depends on `setup`. Run `make test` from a fresh clone.

**Stanford grad.**

Environments are dependency graphs: solver conflicts, lock files (pip-tools, uv, conda-lock), and the reason a lock file belongs in the experiment record. Numerical results can change with a BLAS backend or a compiler flag.

Git internals: content-addressed objects (blob, tree, commit), refs, the reflog as a safety net, and `bisect` for finding the commit that broke a metric. Shell as glue: `set -euo pipefail`, and knowing when a task has outgrown the shell and belongs in Python.

On macOS: launchd rather than cron, `caffeinate` for long jobs, and the permission system (Full Disk Access) that silently blocks scheduled jobs from reading files.

_Try this._ Break a metric deliberately in one commit among twenty, then find it with `git bisect` driven by a script. Afterward, delete a branch and recover it from the reflog.

**PhD at Apple.**

At scale, environments become containers and hermetic builds (Nix, Bazel), because ML reproducibility depends on driver versions, BLAS backends and compiler flags that a requirements file does not capture. Git's model does not fit large data and model artifacts, which is why data versioning and model registries grew separate lineage systems.

Security hygiene is part of the craft: secrets in a credential manager, least-privilege tokens, and a scanner that fails the build if a key appears in a diff.

The reproducibility problem in research is mostly this stage done badly. A lock file, a data hash and a hardware note are the minimum for a claim to be checkable a year later.

_Try this._ Write the 'reproduce this result' section of a paper for this repository: exact commands, content hashes for data, versions, hardware, and expected runtime. Then have someone else follow it.

**Check yourself.**

<details>
<summary>Why must `.env` never be committed?</summary>

It holds credentials. Git history is permanent, and public repositories are scanned for keys within minutes of a push.

</details>

<details>
<summary>What does a virtual environment isolate?</summary>

Installed packages and the interpreter. Not your data, not the operating system's libraries, and not your git history.

</details>

### Lesson 1: The question: what are we predicting?

_Lives in the plan's 'What is being predicted'; stages 6, 8 and 10._

**Goal.** Turn 'will traffic be bad?' into a unit, a target, a horizon and a metric, and see why a probability is the right kind of answer.

**Age 5.**

Roads get slow sometimes, like when everyone leaves the park at the same time and the path gets crowded.

We are building a helper that looks at the road right now and guesses whether it will be slow in a little while, so people can pick a different road before they get stuck.

It is like looking at a dark sky and guessing it will rain soon. Sometimes the guess is wrong, and that is okay. A good helper is right more often than a coin.

_Try this._ Look out the window and guess whether it will rain in one hour. Check later. Were you right? What clue did you use?

**Middle school.**

Every half mile or so on the freeway there is a sensor in the pavement that counts cars and measures their speed, and it writes a row of numbers every five minutes.

Our program reads the last hour of those rows and answers, for every sensor, one question three times: will this spot be slow 15, 30 and 60 minutes from now?

The answer is not yes or no. It is a number between 0 and 1. A 0.8 means 'very likely slow'; a 0.1 means 'probably fine'. Later we decide how big the number must be before we warn anyone.

To find out whether the program is any good, we wait and see what actually happened. At 4:00 pm sensor 12 shows 65 mph and the program says 0.7 for 4:30. At 4:30 the sensor shows 22 mph. That guess was right.

_Try this._ Write down what 'slow' should mean. Under 30 mph? Slower than it usually is at this time? Both are reasonable, and they are different questions. Keep your answer; lesson 4 comes back to it.

**High school.**

A prediction problem has four parts. The unit: one station at one 5-minute timestamp. The inputs: everything known at that moment, and nothing after it. The target: a yes-or-no fact about the future, here 'the station is inside a slow episode h minutes later', with h in {15, 30, 60}. The metric: how guesses are scored.

A slow episode is three consecutive 5-minute steps in which speed is below 60 % of what is typical for that station at that time of day. Three steps, so a single noisy reading does not count.

The model outputs a probability rather than a verdict because a 0.55 and a 0.95 carry different information, and because the decision of when to warn depends on how costly a false alarm is compared with a missed jam. That threshold is chosen afterwards, not baked into the model.

One model per horizon, so the 15-minute question and the 60-minute question get their own answers.

_Try this._ Pick a station in your head. Write one row on paper: a few input columns (speed now, speed 5 minutes ago, speed at the next station downstream) and one target column (slow in 30 minutes?). Notice that at the moment you would use this row, the target column is blank. That blank is the whole point.

**Stanford undergrad.**

Formally this is supervised binary classification on a panel indexed by (station s, time t). The feature vector x_{s,t} is a function of the information available at t, written F_t; the target is y^{(h)}_{s,t} = 1[label(s, t+h) = 1]. We fit f_h(x) ≈ P(y^{(h)} = 1 | x) for each horizon, a direct multi-horizon strategy rather than a recursive one.

The distinction between nowcasting (h = 0) and forecasting (h > 0) is not cosmetic. With `speed_lag_1` in the features and a three-step label, h = 0 is almost solved by copying the last value, which is exactly what the reference plan would have measured.

Calibrated probabilities matter because the alert is a decision. With costs C_FA for a false alarm and C_miss for a missed episode, the Bayes decision rule warns when p > C_FA / (C_FA + C_miss). That is only valid if p is an honest probability.

The metric follows from prevalence: positives are a few percent of rows, so the natural summary is average precision, the area under the precision-recall curve, because the operational question is 'how good is the ranking near the top'.

_Try this._ Derive the threshold rule: write expected cost as a function of the decision for a row with probability p and show that warning is cheaper exactly when p exceeds C_FA / (C_FA + C_miss). Then compute it for a false alarm costing 1 and a miss costing 4.

**Stanford grad.**

The label is a function of a future path, three consecutive slow steps ending at t + h, not of a single future point, so the targets across horizons and across adjacent timestamps are strongly dependent. The panel is neither independent across t (autocorrelation) nor across s (propagation), so the effective sample size is orders of magnitude below the ten million rows, and every confidence interval must respect that.

The target is defined relative to a profile estimated from training data. Label noise therefore inherits the profile estimator's error, and the label's meaning shifts with the split. Two competing framings avoid part of this: regress the speed ratio with a quantile loss and threshold afterwards, keeping ordinal information; or model episode onset as a hazard and predict time-to-onset.

Direct multi-horizon classifiers ignore coherence: nothing forces the 60-minute probability to relate sensibly to the 30-minute one. Joint sequence outputs or monotone post-processing are the usual fixes.

Prevalence drifts across seasons and roadwork, so a single test window is one draw from a moving distribution. Report rolling-origin folds, not one number.

_Try this._ Show that if the profile were computed on the test period too, the positive rate on test would be pinned near a fixed quantile by construction, and explain why that hides real change.

**PhD at Apple.**

At the scale of a phone maps product the unit is a road segment with probe data from devices, not a fixed detector, and the problem becomes travel-time and speed prediction over a graph of millions of segments under privacy constraints: on-device aggregation, differential privacy on aggregate speeds, and sparse coverage that varies by segment and hour. The binary 'slow episode' is a product simplification that trades information for interpretability; production systems keep a continuous target and threshold at the edge.

The literature that matters here is graph-based ETA prediction on supersegments (Derrow-Pinion et al., 2021) and the spatio-temporal forecasting benchmarks on PeMS data (DCRNN, STGCN, Graph WaveNet). Open questions worth a thesis chapter: multi-horizon coherence with monotone or joint outputs; non-stationarity handled online versus by periodic retraining; and performative prediction, since drivers who act on an alert change the very label the alert predicted.

The 16 GB Mac mini is a useful proxy for on-device budgets: model size, latency and energy are the constraints that decide what ships, and a model that only runs in a data center is a different product.

_Try this._ Define the label on probe data with heterogeneous sampling rates. State the estimator bias it introduces for low-coverage segments and propose a correction.

**Guided walkthrough.**

1. Load one month of cleaned rows (or fake them) and pick a station.

   ```python
   import pandas as pd, numpy as np
   df = pd.read_parquet("data/pems_processed/2026-03.parquet")
   s = df[df.station_id == df.station_id.iloc[0]].sort_values("timestamp").reset_index(drop=True)
   ```

2. Build the input side of one row: speed now and 5, 10, 15 minutes ago.

   ```python
   for k in (0, 1, 2, 3):
       s[f"speed_lag_{k}"] = s["speed"].shift(k)
   ```

3. Build the target side: the label 30 minutes later is the label six rows down.

   ```python
   s["y_30"] = s["label"].shift(-6)      # negative shift looks forward
   print(s.loc[100, ["timestamp", "speed_lag_0", "y_30"]])
   print(s.loc[106, ["timestamp", "label"]])   # same value as y_30 at row 100
   ```

4. Confirm there is nothing in the inputs from after the row's timestamp. Every lag has k ≥ 0 and the only forward shift is on the target.
5. Count the positives. That number is the prevalence and the baseline average precision.

   ```python
   print(s["y_30"].mean())
   ```

**Check yourself.**

<details>
<summary>Why is the target column blank at the moment you would use the row?</summary>

Because the target is a fact about t + h, and at t it has not happened yet. Training rows are built from history where t + h is already known; serving rows never have it.

</details>

<details>
<summary>Why three separate models rather than one model with a 'horizon' column?</summary>

Simplicity and honest early stopping: each horizon stops training where its own validation score peaks, and the three thresholds are chosen independently. A single model is possible and slightly more coherent across horizons, at the cost of one more feature to get wrong.

</details>

<details>
<summary>Why not output yes or no?</summary>

The cost of a false alarm and of a miss are not equal and are not known when the model is trained. A probability defers that decision to a threshold chosen later, and can be evaluated for honesty (calibration).

</details>

**Misconceptions.**

- 'The model predicts crashes.' It predicts slow episodes; most are propagation of a jam that already exists somewhere downstream, or the daily build-up.
- 'Accuracy is the score.' With 5 % positives, always saying 'not slow' is 95 % accurate. Average precision and recall at a threshold are the scores.
- 'More horizons are free.' Each horizon is a separate, harder problem; skill at 60 minutes is modest and must be shown against persistence.

### Read and watch, explained at every level

#### Software Carpentry: The Unix Shell, Version Control with Git

_free, half a day each, exactly the commands you will use_

**Age 5.** Two short lessons on the internet that teach you to talk to the computer by typing, and to use the magic notebook that remembers your work. You follow along and type every command yourself, and the lesson tells you what should happen after each one.

**Middle school.** Free workshops written for scientists who have never programmed. Each takes about half a day. The Shell lesson teaches moving between folders, listing files, and joining small commands together with pipes. The Git lesson teaches saving snapshots of your work and pushing them to GitHub so a copy lives online. Do them in order with a terminal open, and do the exercises; watching is not the same as typing.

**High school.** The Carpentries lessons are hands-on with checkpoints every few minutes. The shell lesson covers navigation, wildcards, pipes and filters, loops and simple shell scripts. The git lesson covers `init`, `add`, `commit`, `diff`, `log`, branches, remotes, and collaborating through GitHub, including how to resolve a conflict. That is exactly the set of commands this repository's workflow needs, and nothing you will not use. Budget one day for both and take notes on the commands you find yourself reaching for.

**Stanford undergrad.** These lessons are the minimum viable toolkit for reproducible work: scripting a procedure rather than clicking through it, and versioning every change so any result can be traced to the code that produced it. Their pedagogy, live coding with frequent small exercises, is worth copying when you later teach someone else. After them, the man pages and `git help <command>` become readable, and Pro Git's chapters 1 to 3 will feel like a review with more depth.

**Stanford grad.** Compact enough to finish in a day, they leave deliberate gaps you will fill later: shell robustness (`set -euo pipefail`, quoting, exit codes), git internals and history rewriting, and the difference between a merge and a rebase. Treat them as the vocabulary lesson before the grammar book. Their assessment data (they have been taught to tens of thousands of researchers and iterated on feedback) is a rare example of evidence-based technical teaching.

**PhD at Apple.** Written and maintained by The Carpentries community, these are the lessons to hand to a new hire from a domain science before they touch the pipeline. For a research lead the value is twofold: a shared baseline of tooling across a team, and a template for writing your own onboarding lesson for this repository in the same style, with exercises that run against real PeMS files.

#### Pro Git, chapters 1–3

_free online; enough Git for a year_

**Age 5.** A book about a tool called git. Git is like a magic notebook that remembers every version of your drawing, so if you make a mess you can flip back to yesterday's page. The first three chapters teach you how to save a page, look back at old pages, and share the notebook with a friend.

**Middle school.** Pro Git is the free official book for git, the tool programmers use to save snapshots of their work and share them on GitHub. Chapter 1 explains what git is and installs it, chapter 2 covers the everyday commands (making commits, seeing what changed, undoing mistakes, working with a remote copy), and chapter 3 covers branches, which are parallel copies you can experiment on and merge back. Read it with a terminal open and run each command on a practice folder.

**High school.** The reference for git, free online, by Scott Chacon and Ben Straub. Chapter 1 explains the snapshot model: git stores whole snapshots, not differences, and a commit is a pointer to one. Chapter 2 is the daily toolkit: `init`, `clone`, `add`, `commit`, `status`, `diff`, `log`, `remote`, `push`, `pull`, plus tagging and aliases. Chapter 3 is branching and merging, including what a fast-forward is and how to handle a conflict. That is enough for a year of solo and small-team work; the later chapters (rebase, hooks, internals) can wait until you need them.

**Stanford undergrad.** The value of the first three chapters is the mental model rather than the commands: the three areas (working tree, index, repository), commits as immutable objects in a graph, and branches as movable pointers to commits. Once that picture is in your head, every confusing git situation becomes a diagram you can draw, and `git log --graph --oneline --all` becomes your debugger. Read chapter 3 twice; branching is where most people's understanding is vague.

**Stanford grad.** Chapter 10 on internals is the next read: the content-addressed object store (blobs, trees, commits named by hash) is the same idea as the manifest hashes in this pipeline and as data-versioning tools such as DVC and lakeFS. Git's model of history is what makes reproducible experiments possible: a commit hash plus a data hash names a result forever. Chapter 7's tools (stash, bisect, reflog, rewriting history) are what you reach for when an experiment goes wrong.

**PhD at Apple.** At scale git's limits show: monorepos, large binary artifacts (git LFS), and the mismatch between versioning code and versioning data or models, which is why model registries and data lakes grew their own lineage systems. Still, the discipline the book teaches, small atomic commits with messages that explain why rather than what, is what makes a research codebase reviewable and a production incident debuggable at 3 am. For a team, the book is also the reference for a written branching policy.

#### The repo's own `00_CONCEPTS.md` and `05_PLAN_COMBINED_DATASETS.md`

_read them once now, again in phase 4_

**Age 5.** The project's own notes, written by the people who built it. They explain what the helper is for and how the pieces fit together, like the instruction sheet that comes in a toy box.

**Middle school.** Two documents inside this repository. The concepts file explains the ideas behind the first model, including the math of the logistic regression, in plain language. The plan file describes how the PeMS sensor data and the 511 incident feed will be combined into one training table. Read them once now, even if half of it is unclear, and again after phase 4, when all of it will make sense.

**High school.** Read them like a map of the project. `00_CONCEPTS.md` explains the sigmoid, cross-entropy and gradient descent that lesson 7 rebuilds. `05_PLAN_COMBINED_DATASETS.md` sets out the stages, the label definition, the join rules between incidents and stations, and the time-zone warning that lessons 3 to 5 elaborate. Write down every term you do not understand; that list is your study plan for the next three phases.

**Stanford undergrad.** Project documentation is a primary source: it records decisions and their reasons, and it goes stale the moment code diverges from it. Compare the plan with the flowchart document (`21_PLAN_M1_TRAFFIC_MODEL.md`) and list what changed and why: the forward-shifted targets, the train-only profile, the second label, the missing real-time feed. That diff is a lesson in how plans evolve when they meet data.

**Stanford grad.** Read them critically. The original plan paired features at t with the label at t, and computed the profile on all history; both are design errors that would have produced flattering numbers. Finding such issues in a design document before code exists is a core research and engineering skill; practice it by writing a one-page review of `05_PLAN_COMBINED_DATASETS.md` as if it were a submission.

**PhD at Apple.** Design documents are where most of the leverage in an ML project lives, and most teams write them badly. The exercise at your level is to write the version a team could execute: explicit interfaces between stages, success criteria with numbers, risks with owners, and what would make you stop. Use these two files as the before, and the flowchart plan as a partial after.

### Build

- Clone Traffic1, create the venv, install the requirements, get a 511 key, run `02_collect_events.py` and open the CSV.
- Add `data/` and `.env` to `.gitignore`, commit, push. Start `notebook/2026-W37.md` with what you did and what confused you.

### Done when

- You can say in one sentence what each file in the repo root is for.
- One commit of yours is on GitHub, and `.env` is not in it.

### Pitfalls

- Rosetta Python, which runs at half speed and breaks some wheels.
- Reading about tools instead of typing the commands.

## Phase 1: Python that does work

_weeks 1–5 · ≈40 h · builds stage 2, rebuilt_

**Goal.** Read and write the collector scripts fluently.

### What you need to learn

- Variables, numbers, strings, lists, dicts, sets, tuples; `if`, `for`, `while`; functions, arguments, return values, scope.
- Modules and imports, files, CSV and JSON, exceptions, `requests`, `argparse`, f-strings, comprehensions.
- Dates and time zones with `datetime` and `zoneinfo`; what UTC is and why the log stores it.
- The debugger, print-debugging with intent, and `pytest` for small functions.

### Topic: Python: from variables to functions to files

**Goal.** Read and write the kind of Python the collectors are made of: data structures, functions, files, JSON and CSV.

**Age 5.**

Python is a language for telling the computer what to do, one line at a time.

A variable is a box with a name. A list is a row of boxes. A function is a recipe you can use again and again without rewriting it.

_Try this._ Type `name = 'Sam'` and then `print('Hi', name)`. Change the name and run it again.

**Middle school.**

Types: numbers, strings, lists, and dictionaries, which are lookup tables from names to values, perfect for a JSON event. Booleans are true or false.

`if` decides, `for` repeats, `def` makes a function with inputs and an output. Files are opened, read line by line, and written. CSV and JSON are text with rules, and Python's `csv` and `json` modules read them for you.

_Try this._ Read `data/fremont_events_log.csv` with the `csv` module and count rows per `event_type` using a dictionary. Print the counts sorted from largest to smallest.

**High school.**

The skills inside `02_collect_events.py`: `requests.get(url, params=...)` and `.json()`; dictionary access with `.get()` so a missing key does not crash; list comprehensions to filter events; f-strings; `with open(...)`; `argparse` for `--bootstrap`; `try/except` around the network call; `datetime` with `timezone.utc`.

Modules and the `if __name__ == '__main__':` guard, so a file can be both a script and something another script imports.

Style that pays off in pipelines: small functions with one job, names that say what a variable holds, and no clever tricks.

_Try this._ Rewrite `flatten_event()` from its docstring without looking at the original, then run both on three real events and compare the outputs field by field.

**Stanford undergrad.**

Python's data model: everything is an object; mutability (lists versus tuples), identity versus equality, and references, so a list inside a dictionary is shared, not copied. Scope (local, enclosing, global, built-in) and closures.

Iterators and generators (`yield`) let you stream a large file without loading it, which is the pattern stage 5 needs. `pathlib` for paths, type hints and `dataclasses` for records, and the standard library's breadth: `collections.Counter`, `itertools`, `functools`.

Performance: Python loops are slow, so vectorize with numpy and pandas, and profile with `cProfile` before optimizing anything.

_Try this._ Write a generator that yields only Fremont rows from a gzipped PeMS file, line by line, and measure its peak memory with `tracemalloc`. Compare with reading the whole file.

**Stanford grad.**

The interpreter: CPython compiles to bytecode; the global interpreter lock means threads do not speed up CPU-bound code, so use processes or vectorized libraries; reference counting plus a cycle collector manage memory; C extensions such as numpy are the escape hatch.

Packaging with `pyproject.toml`, entry points and wheels. Testing strategy: unit tests for pure functions, fixtures for I/O, property-based tests for parsers. Concurrency for I/O: `asyncio` for polling several endpoints with a rate limiter.

_Try this._ Convert the poller to `asyncio` with an explicit rate limiter and prove, by logging, that it makes exactly the same requests in the same order.

**PhD at Apple.**

In production ML, Python is the glue, not the engine: the arithmetic runs in C, C++ or Metal kernels reached through numpy, PyTorch and Arrow. The engineering concerns are dependency hygiene across a fleet, reproducible numerics, typed interfaces at module boundaries, and knowing when a hot loop should move to a compiled extension.

Python's flexibility is why ML iteration is fast and why production systems need contracts, schemas and tests to survive it.

_Try this._ Find the one function in the pipeline whose cost dominates, implement it twice (pandas and polars), benchmark both on a real file, and write down which you would ship and why.

**Check yourself.**

<details>
<summary>Why `event.get('severity')` rather than `event['severity']`?</summary>

Some 511 events lack the key. `.get` returns None instead of raising, so one odd event does not stop the poll.

</details>

<details>
<summary>What does the `if __name__ == '__main__'` guard do?</summary>

It runs the block only when the file is executed directly, so another script can import its functions without triggering a poll.

</details>

### Topic: Tests, debugging and time zones

**Goal.** Prove small pieces work, find out why they do not, and never be confused by a clock again.

**Age 5.**

A test is a tiny check that asks 'does this still work?' and answers yes or no. Debugging is being a detective when the answer is no.

When it is lunchtime in California it is dinnertime in London. Computers avoid the mix-up by writing times in one shared clock called UTC.

_Try this._ Find out what time it is right now in a city far away. How many hours different is it?

**Middle school.**

`pytest` runs every function whose name starts with `test_` and reports which passed. A good test builds a tiny fake input, such as five made-up events, and checks the exact output.

Debugging: read the error message from the bottom up, print the values you assumed were true, and shrink the input until the bug is obvious.

A naive time has no zone attached and is ambiguous; an aware time carries its zone. Store and compute in UTC; convert to California time only to display.

_Try this._ Write `test_event_touches_fremont` with one event inside the bounding box and one outside, run `pytest`, then break the function on purpose and watch the test fail.

**High school.**

Test anatomy: arrange, act, assert. Fixtures for shared setup, `parametrize` to run one test over many cases, `pytest -x -q` to stop at the first failure.

Debugging tools: `breakpoint()`, the VS Code debugger, and `logging` instead of print for scripts that run unattended.

Time in Python: `zoneinfo.ZoneInfo('America/Los_Angeles')`, `datetime.now(timezone.utc)`, `astimezone()`, ISO 8601 strings ending in `Z`. Daylight saving removes an hour in March and repeats one in November; pandas exposes the choice through `ambiguous` and `nonexistent`.

_Try this._ Write a test that localizes 2026-11-01 01:30 to America/Los_Angeles and shows why `ambiguous` must be specified.

**Stanford undergrad.**

Testing data code: golden files (a stored known-good output) for pipelines, property checks (row counts preserved, no NaN in keys), and boundary cases (daylight saving, empty files, a day with one station). Test doubles for the network so tests run offline.

Debugging as hypothesis testing: form a guess, find the cheapest experiment that could refute it, and bisect the pipeline when the wrong number appears far from its cause.

Time semantics: instants versus wall-clock times, offsets versus zones (a zone has rules, an offset is a number), the epoch, and why `datetime.utcnow()` is a trap (it returns a naive object).

_Try this._ Build the 100-row fixture for `10_clean_pems.py` that includes a daylight-saving boundary and a station boundary, and write the three tests that use it.

**Stanford grad.**

Testing ML systems: data tests (schema, ranges, distributions), model tests (a training run on a fixture reaches a minimum average precision; shifting every timestamp by a week changes nothing), and pipeline tests on a golden day. Nondeterminism from seeds, thread counts and the GPU causes flakiness, so pin what you can and tolerate what you must.

Debugging statistical code is different: the failure is a wrong number, not a crash. Sanity baselines and synthetic data with known answers are the instruments.

The tz database changes when governments change rules; pin `tzdata`. The ambiguous hour is a real loss of information, not a formatting problem.

_Try this._ Write an invariance test that shifts all timestamps by seven days and asserts the feature table is identical except for the calendar columns.

**PhD at Apple.**

Verification at scale: continuous integration on every change, contract tests between pipeline stages, canaries on live data, and the uncomfortable fact that most production ML failures are data failures that no unit test caught.

Time in distributed systems: clock skew, event time versus processing time, watermarks in stream processors, and the point-in-time discipline that makes training match serving.

Research threads: metamorphic testing for ML, and executable data contracts.

_Try this._ Specify the contract between `10_clean_pems.py` and `13_feature_engineering.py` as executable assertions, and decide which assertions should stop the pipeline and which should only warn.

**Check yourself.**

<details>
<summary>Why test with fake events rather than the live API?</summary>

Speed, determinism and no network. The API changes; your fixture does not, so a failure means your code changed.

</details>

<details>
<summary>Why store UTC and convert only for display?</summary>

UTC has no daylight saving, so arithmetic on it is safe. Local time has a missing hour and a repeated hour every year.

</details>

### Lesson 2: The data: detectors, five-minute rows, incidents

_Lives in stages 1, 2 and 4 · 02_collect_events.py, 09_collect_pems_selenium.py._

**Goal.** Know what a PeMS row and a 511 event actually measure, how much of them there are, and where the numbers lie.

**Age 5.**

Under the road there are little wires that feel cars driving over them, like a doorbell mat that rings when you step on it.

Every five minutes the wires write down how many cars went by and how fast they were going. Another helper, called 511, writes down when there is a crash and where.

We collect both into two big lists. The lists are the food our helper learns from.

_Try this._ Count the cars that pass your window for one minute and write the number down. Do it again a few minutes later. The numbers change; that is data.

**Middle school.**

PeMS is California's collection of those road sensors. Each sensor is a 'station'. Every 5 minutes a station makes one row: the time, the station number, how many cars passed (flow), how full the road was (occupancy, the share of time a car was sitting over the sensor), and the average speed.

One day of the Bay Area district is about 1.3 million rows in a text file with no column names. We only care about the roughly 100 stations in Fremont on I-880, I-680 and SR-84.

511.org lists incidents: crashes, stalls, roadwork, each with a time, a road name and a map point. Our collector asks 511 every 15 minutes 'what is happening?' and saves anything near Fremont into a spreadsheet-like file.

The two lists look different: one is a steady drumbeat of numbers, the other is a scattered log of events.

_Try this._ Open `data/fremont_events_log.csv` in a spreadsheet. Sort by `created`. Which hour of the day has the most incidents? Which road?

**High school.**

Three quantities per station per 5 minutes. Flow: vehicles counted. Occupancy: the fraction of time the loop was covered, which stands in for how dense the traffic is. Speed: miles per hour, sometimes measured directly with a pair of loops and sometimes estimated from flow and occupancy with an assumed vehicle length.

The fundamental diagram links them: as density rises, flow rises until the road reaches capacity, then flow falls as the road jams. That is why occupancy matters and not just speed; a road at 60 mph with heavy occupancy is about to break down.

Timestamps in the file are California local time with no zone attached. We convert them to UTC and handle the two daylight-saving switches, where one hour repeats in November and one is missing in March.

Incidents arrive as JSON with `created` and `updated` times, but our poller only looks every 15 minutes, so a short stall can be missed entirely and every start time is fuzzy by up to 15 minutes.

_Try this._ Sketch the fundamental diagram: flow on the vertical axis, density on the horizontal. Mark free flow, capacity and jam. Where on the curve is 'slow'?

**Stanford undergrad.**

Detector physics decides the error model. Single-loop stations infer speed from flow and occupancy through a per-station 'g-factor' (effective vehicle length), so speed error grows at low flow and when truck mix changes. PeMS also imputes missing samples from neighboring stations and history and reports the share of real samples as `% Observed`; aggregate columns are flow-weighted across lanes.

Volume arithmetic: about 100 stations × 288 steps × 365 days is 10.5 million rows a year; at 5 float32 columns that is 210 MB in memory, and with 60 feature columns about 2.5 GB. Memory is rows × columns × bytes, and dtype choices decide whether two years fit a 16 GB machine.

Time semantics: `tz_localize('America/Los_Angeles')` must be told what to do with the ambiguous hour (fall back) and the nonexistent hour (spring forward); the plan drops the ambiguous hour rather than guess.

The 511 feed is an event stream sampled every 15 minutes, so incident start and end times are interval-censored. `created` is when CHP logged the report, which lags the physical event by an unknown, skewed delay.

_Try this._ Compute the memory of a year of features as rows × columns × 4 bytes. Then compute the same for float64 and for a naive pandas pipeline that holds three copies during a groupby-shift.

**Stanford grad.**

Write the measurement model: observed speed = true speed + detector noise + imputation error, where imputation is itself a model of neighbors and history. Imputed rows are therefore correlated with neighboring stations' values, which leaks spatial information into what looks like an observation and flatters any spatial model evaluated on them.

Spatial sampling sets what is predictable. Station spacing of 0.3 to 0.6 miles and backward-forming shock speeds around 10 to 15 mph (kinematic wave theory) mean a jam crosses one station every two to four minutes, so a 30-minute horizon corresponds to roughly two to six stations of upstream propagation.

Temporal aggregation at 5 minutes filters the high-frequency dynamics; the data cannot resolve a 90-second stall. The incident covariate has measurement error of its own: reporting delay, missing geography, unknown severity, and a selection effect, since incidents are reported more readily when congestion is already visible.

_Try this._ Estimate how many stations upstream a queue's tail passes in 30 minutes at a 12 mph shockwave with half-mile spacing, and what that implies for the value of two-hop versus five-hop neighbor features.

**PhD at Apple.**

Sensor networks in production mix fixed loops, radar and probe data, and the fixed loops come with known pathologies: g-factor drift, chattering detectors, lane-assignment errors. In a district it is normal for a fifth to a third of detectors to be unhealthy at any time, so a serious pipeline carries a detector-health model as a component in its own right, and the imputation mask becomes a feature rather than an embarrassment.

Probe data introduces the privacy layer: on-device aggregation, differential privacy on segment speeds, and coverage that is a function of device share. The statistical problem becomes estimating a segment speed from a biased, sparse sample and propagating that uncertainty.

Here the binding constraint is not modelling but data availability: a 305-second rate limit and a one-day publication lag. Recognizing that a project is data-bound rather than model-bound is most of senior judgment.

Research threads: learned spatio-temporal imputation (graph recurrent imputation networks) against the risk that the downstream model learns the imputer; representing incidents as a marked point process joined to a regular grid; and asynchronous multi-rate sensor fusion.

_Try this._ Design a detector-health score from `% Observed`, flow-occupancy consistency against the station's own fundamental diagram, and agreement with neighbors. Say how you would validate it without ground truth.

**Guided walkthrough.**

1. Look at a raw file before you trust any description of it.

   ```bash
   zcat data/pems_raw/d04_text_station_5min_2026_03_02.txt.gz | head -3
   zcat data/pems_raw/d04_text_station_5min_2026_03_02.txt.gz | head -1 | tr ',' '\n' | wc -l
   ```

2. Read the first twelve columns for a hundred thousand rows and describe them.

   ```python
   import pandas as pd
   cols = ["ts","station","district","fwy","dir","lane_type","length","samples","pct_obs","flow","occ","speed"]
   df = pd.read_csv("data/pems_raw/d04_text_station_5min_2026_03_02.txt.gz", header=None,
                    usecols=range(12), names=cols, nrows=100_000)
   print(df.describe().T)
   print(df.station.nunique(), "stations in the first 100k rows")
   ```

3. Count rows per station. A healthy station has 288 per day.

   ```python
   print(df.groupby("station").size().describe())
   ```

4. Parse the local timestamp and convert to UTC with explicit DST handling.

   ```python
   ts = pd.to_datetime(df.ts, format="%m/%d/%Y %H:%M:%S")
   df["timestamp"] = ts.dt.tz_localize("America/Los_Angeles", ambiguous="NaT", nonexistent="shift_forward").dt.tz_convert("UTC")
   ```

5. Open the incident log and look at the gap between `created` and `pulled_at` for a few rows. That gap is the reporting and polling delay you will live with.

   ```python
   ev = pd.read_csv("data/fremont_events_log.csv", parse_dates=["created","updated","pulled_at"])
   print((ev.pulled_at - ev.created).describe())
   ```

**Check yourself.**

<details>
<summary>Why does occupancy matter if you already have speed?</summary>

Speed alone cannot tell a nearly empty road from a dense one moving at the same speed. Occupancy stands in for density, and density is what predicts breakdown.

</details>

<details>
<summary>What does `% Observed` = 20 mean for a row?</summary>

Only a fifth of the samples in that 5 minutes were real; the rest of the row was filled in by PeMS's imputer. Treat the row as missing for labels and carry the fraction as a feature.

</details>

<details>
<summary>Why can the poller miss an incident entirely?</summary>

It samples the feed every 15 minutes; an event that is created and cleared between two polls never appears. Start times are also only known to within one poll.

</details>

**Misconceptions.**

- 'The file has headers.' It does not; column order must be verified against the documentation on a real file.
- 'Speed is measured.' Often it is estimated from flow and occupancy with an assumed vehicle length.
- 'Incidents are the labels.' They are inputs. The label comes from the detectors.

### Read and watch, explained at every level

#### Automate the Boring Stuff with Python, chapters 1–13 and 16–18

_free online; practical from page one_

**Age 5.** A friendly book that teaches Python by making the computer do chores for you, like renaming lots of files at once or reading a spreadsheet. It is free to read on the internet, and every chapter ends with something useful you made.

**Middle school.** Al Sweigart's book starts from zero: numbers, text, lists, dictionaries, functions, and then practical chapters on files, spreadsheets, CSV and JSON, and the web. Chapters 1 to 13 are the language basics; 16 to 18 cover CSV and JSON, time and scheduling, and sending email and text messages, which are close to what our collector script does. Type every example rather than copying it, and do the practice projects at the end of the chapters.

**High school.** The best first Python book because every chapter ends in a program that does something real. For this project the payoff chapters are 9 and 10 (reading, writing and organizing files), 13 (spreadsheets), 16 (CSV and JSON, the two formats the pipeline lives on), and 17 (time and scheduling, which the poller needs). After it you can read `02_collect_events.py` end to end and change it with confidence. Its style, small scripts with clear names, is the right style for pipeline code.

**Stanford undergrad.** Its limits are exactly what the official tutorial and later phases fill: little on the data model (mutability, references), nothing on numpy, and almost nothing on testing. Use it for fluency and for the habit of automating anything you do twice, then move on. The chapter on debugging (chapter 11 in the current edition) is the one to reread when you are stuck on a stage 5 bug at midnight.

**Stanford grad.** Pedagogically it works through immediate utility, and that is the model for how to bring domain experts (a traffic engineer, a city planner) onto a data project: give them a script that saves them an hour on day one. If you already program, skip it yourself; recommend it to collaborators and write the project-specific chapter it lacks, 'automating the PeMS download and cleaning'.

**PhD at Apple.** The book's persistence as the default beginner text, across editions and free online, reflects a fact about ML teams: the bottleneck is often basic scripting fluency among the people who understand the domain. Building that capability across a team is a higher-leverage act than most modelling work, and this book is the cheapest tool for it.

#### The official Python tutorial, sections 3–9

_free; the reference once Sweigart runs out_

**Age 5.** The instruction book written by the people who make Python. It is for looking things up when the friendly book did not say.

**Middle school.** Sections 3 to 9 cover numbers and strings, control flow (`if`, `for`, `while`, functions), data structures (lists, dictionaries, sets, tuples), modules, reading and writing files, errors and exceptions, and classes. It is terser than Sweigart and assumes you will experiment. Read a section when you meet that topic in the code, then try its examples in the interpreter.

**High school.** The tutorial is the authoritative short version of the language: how `for` really works (iterators), how arguments are passed to functions, what a module is and how imports find it, how exceptions propagate up the call stack, and what a class is. Read it once fully after Sweigart, then keep the language reference for details. Section 5 on data structures and section 7 on input and output are the ones the pipeline code exercises most.

**Stanford undergrad.** This is where the data model becomes precise: mutable versus immutable objects, `is` versus `==`, how comprehension scope works, `*args` and `**kwargs`, `with` statements and context managers, and the exception hierarchy. Pair it with the library reference pages for `collections`, `itertools`, `pathlib` and `dataclasses`, which are the standard-library tools a clean pipeline uses.

**Stanford grad.** Beyond the tutorial, the language reference and the PEPs are where design rationale lives: PEP 8 for style, PEP 484 and successors for typing, PEP 20 for the philosophy. Reading a few accepted and rejected PEPs teaches how language decisions are argued with evidence, which is a useful model for arguing design decisions in your own project.

**PhD at Apple.** For a team the tutorial's value is as the shared baseline that code-review standards can point to. Beyond it, the investment that turns research code into production code is static typing (mypy or pyright on every module), packaging, and a linter configuration checked into the repository so style is never a review comment.

#### Kaggle Learn: Python

_free, short, with exercises in the browser_

**Age 5.** A website where you learn Python in small steps and it checks your answers right away, like a game with levels.

**Middle school.** Seven short lessons in the browser with exercises that are graded the moment you run them. No installation. It covers the basics you also meet in Sweigart: functions, booleans, lists, loops, strings, dictionaries, and importing libraries. Use it for practice and repetition, not as your only source; finish it in a weekend.

**High school.** Good for building confidence through repetition. Each lesson pairs a short tutorial notebook with an exercise notebook that checks your answers. Because it runs in the cloud it hides environment setup, which is convenient here and exactly why phase 0 has you set up your own machine, so you learn the setup you will actually need.

**Stanford undergrad.** Kaggle's notebooks are a fine scratchpad and a poor habit if they replace a local environment: no version control, no tests, no reproducible dependencies. Take the practice, leave the workflow. The course's exercises on functions and on list handling are worth doing; the rest is a review.

**Stanford grad.** The Learn courses are a funnel into Kaggle competitions. Their pedagogy, tiny graded steps, is effective for syntax and weak for design and debugging, the two skills that decide whether a pipeline survives. Use them accordingly and move on quickly.

**PhD at Apple.** Kaggle as a platform matters more than this course. It is where tabular machine-learning practice evolved: gradient-boosted ensembles, feature engineering, leakage detection, and careful validation. Reading winning solutions' write-ups for time-series competitions is a research-adjacent habit that this course is the first step toward.

#### The `zoneinfo` and `datetime` documentation

_free; read the section on aware versus naive datetimes twice_

**Age 5.** The instructions for the part of Python that knows about clocks, dates and the different times in different places.

**Middle school.** `datetime` handles dates and times; `zoneinfo` knows the time zones of the world. Read the section that explains aware versus naive times (aware means the time knows its zone), then the examples that convert a time from one zone to another. That is the whole trick behind storing every time in UTC.

**High school.** The key pages: `datetime.datetime` (aware versus naive, `astimezone`, `isoformat`, `fromisoformat`), `timedelta` for arithmetic, and `zoneinfo.ZoneInfo`. Read the aware-versus-naive section twice, because most time bugs are a naive time mistaken for a UTC one. Then learn the pandas equivalents, `tz_localize` and `tz_convert`, which apply the same ideas to a whole column at once.

**Stanford undergrad.** The docs explain `fold`, the attribute that distinguishes the first and second occurrence of the repeated hour when daylight saving ends. That is how Python represents the November ambiguity that pandas exposes as the `ambiguous` argument. Understanding fold makes the cleaning stage's choice to drop the ambiguous hour a deliberate decision rather than a default.

**Stanford grad.** Time zone rules come from the IANA database through the `tzdata` package and change when governments change rules, so pin the version. The docs' note that naive datetimes are interpreted as local time is a trap on servers set to UTC, and `datetime.utcnow()` returning a naive object is a second trap that the docs now warn about. Both have caused real outages.

**PhD at Apple.** Time handling is a recurring production failure class across the industry. The documentation is the specification; the engineering answer is a house rule (UTC everywhere, aware objects only, conversions at the edges) enforced by a linter rule and a test, so the specification never has to be reread under pressure.

### Build

- Read `02_collect_events.py` top to bottom and write one comment above each function saying what it does and why.
- Rewrite `event_touches_fremont()` from its description without looking, then test it with pytest on five hand-made events (one on I-880 with no coordinates, one inside the box on an unnamed road, one outside both).
- Write a script that prints events per road per day from the CSV, using only the standard library.
- Make the stage 2 change from the plan: record `first_seen_at` and `last_seen_at` per event id, and write each poll's raw JSON to `data/events_raw/`.

### Done when

- Your filter passes your tests and the repo's behaviour is unchanged for real events.
- You can explain why `created` is a string in the CSV and where it must become a datetime.

### Pitfalls

- Tutorial hell: watching without typing. Every chapter ends with you closing the book and writing the code.
- Skipping tests because the script 'obviously works'.

## Phase 2: The math the model uses

_weeks 3–11, alongside phases 1 and 3 · ≈60 h · builds stage 3 from a blank file_

**Goal.** Follow every symbol in `00_CONCEPTS.md` and `03_train_from_scratch.py`, and derive the update rule yourself.

### What you need to learn

- Linear algebra: vectors, dot product, matrix times vector, matrix times matrix, transpose, norms; what `x @ weights + bias` computes for a whole batch; a linear model as a hyperplane.
- Calculus: the derivative as a slope, the chain rule, partial derivatives and the gradient; why stepping against the gradient lowers the loss; derivatives of the sigmoid and the log.
- Probability: random variables, Bernoulli and normal distributions, expectation, variance, conditional probability and Bayes, independence; the logistic function as a probability.
- Statistics: mean, median and percentiles (the label uses a median; free flow is an 85th percentile), sampling, the law of large numbers, confidence intervals, what a p-value is and is not, correlation versus causation.
- Putting it together: binary cross-entropy from maximum likelihood, and its gradient for logistic regression, then the same lines in the code.

### Topic: Linear algebra: vectors, dot products, matrices

**Goal.** See a row of features as a vector, a batch as a matrix, and a model's score as one multiplication.

**Age 5.**

A vector is a list of numbers that belong together, like three apples and two pears written as (3, 2). Adding two lists means adding the matching numbers.

A dot product is 'multiply the pairs and add them up'. That is how we mix clues with how much each clue matters.

_Try this._ Compute (2, 3) · (1, 4). Multiply the first numbers, multiply the second numbers, add the two results.

**Middle school.**

Vectors can be arrows or lists; they have a length and a direction. The dot product x·w = x₁w₁ + x₂w₂ + … is large when two vectors point the same way.

A matrix is a grid of numbers. Multiplying a matrix by a vector does many dot products at once, one per row. That is how a whole batch of rows gets scored in one line of code, `X @ w`.

_Try this._ With X = [[1, 2], [3, 4]] and w = (1, −1), compute X @ w by hand. You should get two numbers.

**High school.**

Shapes must match: an n×d matrix times a d×k matrix gives n×k. The transpose flips rows and columns; the identity leaves things unchanged. Geometrically, a matrix is a linear map.

The dot product equals |x||w|cos θ, so it measures alignment. The set of points where w·x + b = 0 is a line or a plane, which is the decision boundary of a linear model.

In numpy: `@` for products, `.T` for transpose, and broadcasting for adding a bias to every row.

_Try this._ Write the logistic regression score for a whole batch as one matrix product and check the shape of the result against the number of rows.

**Stanford undergrad.**

Vector spaces, linear independence, rank and column space. Projections and least squares through the normal equations XᵀX w = Xᵀy. Eigenvalues and eigenvectors, symmetric matrices and the spectral theorem, the singular value decomposition and principal components.

Positive semidefinite matrices matter here because the Hessian of the logistic loss is one, which is what makes the loss convex. Conditioning explains why standardizing features speeds up gradient descent.

_Try this._ Compute the Hessian Xᵀ diag(p(1 − p)) X for a small X and show it is positive semidefinite by checking vᵀHv ≥ 0 for random v.

**Stanford grad.**

Matrix calculus: gradients of quadratic forms and the chain rule in matrix form. Graph Laplacians and spectral graph convolution (the Chebyshev filters in STGCN), random-walk matrices in diffusion convolution (DCRNN), and the power method.

Numerical linear algebra: floating point, stability, and why you never invert a matrix when you can solve a system.

_Try this._ Build the normalized adjacency for the Fremont road graph, compute its eigenvalues, and interpret the smallest ones in terms of connected pieces of road.

**PhD at Apple.**

Linear algebra is the compute substrate: BLAS and LAPACK, the Accelerate framework on Apple silicon, matrix-multiply throughput as the number that sets training speed, and the distinction between bandwidth-bound and compute-bound kernels. Mixed precision is a linear-algebra decision.

Randomized linear algebra (sketching) for large least-squares problems; tensor decompositions for spatio-temporal data.

_Try this._ Measure GFLOP/s of `X @ W` on the M1 CPU and on MPS for several sizes, plot both curves, and explain their shapes.

**Check yourself.**

<details>
<summary>Why does `X @ w` compute every row's score?</summary>

Each row of X dotted with w is one row's weighted sum; the product does all rows in one operation.

</details>

<details>
<summary>What does a positive semidefinite Hessian buy you?</summary>

Convexity. Every local minimum is global, so gradient descent cannot get stuck.

</details>

### Topic: Calculus: derivatives, the chain rule, gradients

**Goal.** Know why stepping against the gradient reduces the loss, and derive the one gradient every model in this course uses.

**Age 5.**

A derivative tells you how fast something changes, like how steep a hill is right where you are standing.

If you want to reach the bottom, you step downhill. Learning is stepping downhill on the 'mistake' hill.

_Try this._ Draw a hill. Mark where it is steep and where it is flat. Where would a ball roll fastest?

**Middle school.**

The slope of a line is rise over run. The derivative is the slope of a curve at one point. If the function goes up when x goes up, the derivative is positive.

The chain rule: if y depends on u and u depends on x, then the change in y per x is the change in y per u times the change in u per x.

Gradient descent: move x a little against the slope and repeat.

_Try this._ For f(x) = x², the derivative is 2x. Start at x = 3 with step size 0.1. Take three steps of x ← x − 0.1 · 2x and watch x shrink toward 0.

**High school.**

Rules: power, product, chain. Derivatives of eˣ and ln x, and the sigmoid's own derivative σ′ = σ(1 − σ). Partial derivatives when there are several inputs, and the gradient as the vector of them; it points uphill.

The update rule w ← w − η∇L. Applied to one row's cross-entropy, the derivative with respect to the score z is p − y, which is the term at the heart of lesson 7.

_Try this._ Differentiate the one-row cross-entropy with respect to z and confirm you get p − y for y = 1 and for y = 0.

**Stanford undergrad.**

Multivariable calculus: Jacobians, the chain rule as a product of Jacobians (which is exactly backpropagation), Taylor expansion to second order (the Hessian), convexity from the Hessian, and Lagrange multipliers for constraints.

Optimization: gradient descent converges under a Lipschitz gradient with a step size bounded by the inverse of the Lipschitz constant; momentum as a heavy ball; why the learning rate that works depends on the scale of the features.

_Try this._ Derive the backward pass of a one-hidden-layer network as a chain of Jacobian-vector products, and count the multiplications.

**Stanford grad.**

Automatic differentiation: forward mode versus reverse mode, and why reverse mode suits many parameters and one loss. Non-smooth points (ReLU, subgradients). Stochastic gradients as unbiased estimators with variance, convergence rates for convex, strongly convex and non-convex objectives, second-order methods and their scaling limits, natural gradient.

_Try this._ Implement reverse-mode automatic differentiation for scalar operations (a small micrograd) and use it to train lesson 7's model to the same weights.

**PhD at Apple.**

Differentiable programming as a paradigm: implicit differentiation through optimization layers, differentiating through simulators (a differentiable traffic model), gradient estimators for discrete decisions, and the numerics of mixed precision.

_Try this._ Sketch how you would differentiate through a kinematic-wave traffic simulator to fit its parameters to PeMS data, and what would make the gradient uninformative.

**Check yourself.**

<details>
<summary>Why the negative gradient?</summary>

The gradient points toward increasing loss. Learning wants the opposite direction.

</details>

<details>
<summary>What is backpropagation in one sentence?</summary>

The chain rule applied in reverse order along the computation graph, reusing intermediate results.

</details>

### Topic: Probability and statistics: distributions, medians, likelihood

**Goal.** Speak the language the labels, the metrics and the loss are written in.

**Age 5.**

Probability is how likely something is, from 0 (never) to 1 (always). Flip a coin many times and about half come up heads.

Statistics is looking at lots of numbers and finding the typical one and how spread out they are.

_Try this._ Flip a coin twenty times and count the heads. Is it exactly ten? Is it close?

**Middle school.**

Random variables and distributions: a coin has two outcomes, a die six, and speeds spread out in a bell-like shape. Mean, median and mode are three kinds of 'typical'; the median resists outliers.

Spread: range and standard deviation. Percentiles: the 85th percentile speed is the speed that 85 % of readings fall below.

Conditional probability: the chance of slow given rain is not the chance of slow.

_Try this._ From a list of twenty speeds, find the median and the 85th percentile by sorting.

**High school.**

Bernoulli, binomial and normal distributions; expectation and variance and their rules; independence. Bayes' rule with a traffic example: P(crash | slow) is not P(slow | crash).

The law of large numbers, sampling variability and confidence intervals. Correlation is not causation: rain and incidents both cause slowness, so incidents and rain correlate.

The likelihood is the probability of the data given the parameters. Maximum likelihood picks the parameters that make the data most probable, and for a yes-or-no outcome that is exactly the cross-entropy loss.

_Try this._ Simulate a thousand coin flips with p = 0.3 in Python and plot the running fraction of heads. Watch it settle.

**Stanford undergrad.**

Probability spaces and random variables formally; joint, marginal and conditional distributions; expectation as an integral; the central limit theorem. Estimators and their bias, variance and consistency; maximum likelihood and its asymptotics; hypothesis tests and the misuse of p-values; the bootstrap; Bayesian inference as prior times likelihood.

The statistical view of learning: regression estimates E[y | x], classification estimates P(y = 1 | x), and the bias-variance decomposition explains why simpler models sometimes win.

_Try this._ Derive the maximum-likelihood estimate of a Bernoulli p, then show that the cross-entropy loss is the negative log-likelihood of the logistic model.

**Stanford grad.**

Dependence: stochastic processes, stationarity, autocorrelation and effective sample size; time series as dependent data. Robust statistics and the median's breakdown point. Quantile regression and the pinball loss. Proper scoring rules and calibration as a conditional-expectation property. Exchangeability and conformal prediction. Causal inference basics: confounding and the difference between P(y | do(x)) and P(y | x).

_Try this._ Compute the effective sample size of one station's speed series from its autocorrelation and compare it with the raw row count.

**PhD at Apple.**

Statistics for ML at scale: high-dimensional inference, concentration inequalities and generalization bounds; sequential experiments (bandits, switchbacks) with interference; Bayesian deep learning and its approximations; uncertainty with guarantees (conformal, PAC-Bayes); causal machine learning under shift.

_Try this._ Write the estimand for 'the effect of an alert on travel time' and list the assumptions needed to identify it from logged data.

**Check yourself.**

<details>
<summary>Why is the profile a median and not a mean?</summary>

A few outage days with speed 0 or an imputed 70 would drag a mean; the median ignores them.

</details>

<details>
<summary>What does the likelihood measure?</summary>

How probable the observed data are under a candidate model. Maximizing it picks the model that explains the data best.

</details>

### Lesson 7: Logistic regression from scratch: the math the model runs on

_Lives in stage 3 and stage 9 · 03_train_from_scratch.py, 15_train_logreg.py, 00_CONCEPTS.md._

**Goal.** Derive, implement and check the training loop that every later model generalizes: weighted sum, sigmoid, cross-entropy, gradient step.

**Age 5.**

Imagine a seesaw. Each clue pushes it one way ('slow!') or the other ('fine'). Some clues are heavy, like a crash just ahead, and some are light, like a Tuesday.

Learning means finding out how heavy each clue should be. We look at lots of old days, and every time the seesaw tips the wrong way we make the wrong clues a bit lighter and the right clues a bit heavier.

_Try this._ Guess a friend's favorite color. When you are wrong, change your guess a little toward what they said, not all the way. Keep going until you are right most of the time.

**Middle school.**

Give each clue a weight, a number. Add up clue × weight for every clue, plus one extra number called the bias. That total is the score.

Squash the score into 0 to 1 with an S-shaped curve called the sigmoid, so it reads as a probability: a big positive score becomes nearly 1, a big negative score nearly 0, and zero becomes 0.5.

To learn the weights, take an old row, compute the probability, compare with what really happened (1 or 0), and nudge each weight a small step in the direction that would have made the guess better. Do that thousands of times and the weights settle.

_Try this._ One clue: 'speed dropped in the last 15 minutes' is 1 or 0. Weight 2, bias −1. Compute the score and the sigmoid for both clue values. Which one says 'probably slow'?

**High school.**

Score z = w·x + b. Probability p = σ(z) = 1 / (1 + e^(−z)). Loss for one row, the cross-entropy: −[y ln p + (1 − y) ln(1 − p)], which is small when confident and right and huge when confident and wrong.

The gradient is surprisingly clean: ∂loss/∂w = (p − y)·x and ∂loss/∂b = (p − y). The sigmoid's derivative is p(1 − p), and the chain rule cancels it against the log's derivative, leaving error times input.

The update is w ← w − η (p − y) x with a small learning rate η. That line is the whole of `03_train_from_scratch.py`; everything in lesson 9 is this with more layers.

_Try this._ Take x = (1, 0.5), y = 1, w = (0, 0), b = 0, η = 0.1. Compute p, the gradient, and the new w and b. Then compute p again with the new weights and confirm it moved toward 1.

**Stanford undergrad.**

Logistic regression is maximum likelihood for a Bernoulli outcome with a logit-linear mean. The negative log-likelihood is convex in (w, b) (its Hessian X^T diag(p(1 − p)) X is positive semidefinite), so gradient descent with a suitable step reaches the global optimum; L2 regularization is a Gaussian prior and gives the MAP estimate.

Standardizing features conditions the problem; the safe learning rate scales inversely with the largest eigenvalue of the Hessian. Mini-batch SGD trades noise for speed; Adam is unnecessary for a convex problem but harmless.

Without regularization the first-order conditions force the mean predicted probability to equal the observed rate, so the model is calibrated in-sample given an intercept. The decision boundary is a hyperplane; interactions must be built by hand.

_Try this._ Derive the gradient from the negative log-likelihood using the chain rule through σ, and show from the stationarity condition on the intercept that the mean residual is zero at the optimum.

**Stanford grad.**

As a generalized linear model, logistic regression uses the canonical link of the Bernoulli family, which is why the gradient has the score-residual form and why iteratively reweighted least squares (Fisher scoring) converges in a few steps. The MLE is asymptotically normal, giving Wald and likelihood-ratio tests, and quasi-complete separation sends coefficients to infinity unless regularized.

Class reweighting is a prior shift: multiplying positive weights by r shifts the learned intercept by about ln r and destroys calibration; undo it with a log-odds offset before reporting probabilities.

Logistic regression is a one-layer network with sigmoid output; softmax is its multiclass form; the maximum-entropy view derives the same model from moment constraints. Its limits are representational, not statistical.

_Try this._ Show that reweighting positives by a factor r changes the asymptotic log-odds by ln r, and derive the correction that restores calibration to the original prevalence.

**PhD at Apple.**

At scale the same model appears as sparse high-dimensional logistic regression in click-through prediction, trained with distributed SGD or L-BFGS, feature hashing, and per-coordinate learning rates; it persists in production for latency, interpretability and calibration properties that deeper models lack.

The loss lives on in every network as binary cross-entropy with logits, computed through log-sum-exp for numerical stability. Recent theory clarifies what gradient descent does on this loss: on separable data it converges in direction to the maximum-margin classifier (Soudry et al., 2018), which is why early stopping acts as regularization.

Open questions of practical weight: the interaction between class reweighting, calibration and decision thresholds under label shift, and generalization for linear classifiers in the interpolating regime.

_Try this._ State and sketch the proof that gradient descent on logistic loss over linearly separable data converges in direction to the max-margin solution, and explain what it implies for the choice of stopping time.

**Guided walkthrough.**

1. Standardize with training statistics only.

   ```python
   feats = ["speed_lag_0", "speed_lag_3", "speed_roll_mean_30", "down1_speed", "incident_active", "min_sin", "min_cos"]
   mu, sd = tr[feats].mean(), tr[feats].std() + 1e-6
   Xtr = ((tr[feats] - mu) / sd).fillna(0).to_numpy(np.float32); ytr = tr.y_30.to_numpy(np.float32)
   Xva = ((va[feats] - mu) / sd).fillna(0).to_numpy(np.float32); yva = va.y_30.to_numpy(np.float32)
   ```

2. The model, the loss and the gradient in eight lines.

   ```python
   def sigmoid(z): return 1 / (1 + np.exp(-z))
   def loss(w, b, X, y):
       p = np.clip(sigmoid(X @ w + b), 1e-7, 1 - 1e-7)
       return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
   def grad(w, b, X, y):
       p = sigmoid(X @ w + b); e = p - y
       return X.T @ e / len(y), e.mean()
   ```

3. Check the gradient numerically before trusting it.

   ```python
   w = np.zeros(Xtr.shape[1], np.float32); b = 0.0
   gw, gb = grad(w, b, Xtr[:5000], ytr[:5000])
   eps = 1e-4; num = []
   for j in range(len(w)):
       wp = w.copy(); wp[j] += eps; wm = w.copy(); wm[j] -= eps
       num.append((loss(wp, b, Xtr[:5000], ytr[:5000]) - loss(wm, b, Xtr[:5000], ytr[:5000])) / (2 * eps))
   print(np.max(np.abs(np.array(num) - gw)))     # should be ~1e-5 or smaller
   ```

4. Mini-batch gradient descent with a validation check every epoch.

   ```python
   rng = np.random.default_rng(0); eta = 0.1
   for epoch in range(20):
       idx = rng.permutation(len(ytr))
       for i in range(0, len(idx), 4096):
           j = idx[i:i + 4096]; gw, gb = grad(w, b, Xtr[j], ytr[j]); w -= eta * gw; b -= eta * gb
       print(epoch, loss(w, b, Xva, yva), ap(yva, sigmoid(Xva @ w + b)))
   ```

5. Compare with scikit-learn on the same matrix; the two should agree closely.

   ```python
   from sklearn.linear_model import LogisticRegression
   sk = LogisticRegression(C=1e6, max_iter=1000).fit(Xtr, ytr)
   print(np.round(sk.coef_[0], 3)); print(np.round(w, 3))
   ```

**Check yourself.**

<details>
<summary>Why cross-entropy and not squared error?</summary>

Cross-entropy is the negative log-likelihood of the Bernoulli model and is convex in the weights; squared error on a sigmoid output is not convex and its gradient vanishes when the model is confidently wrong.

</details>

<details>
<summary>What does the bias do?</summary>

It sets the probability when all features are zero (after standardization, the average row). Without it the model cannot match the base rate.

</details>

<details>
<summary>Why does the gradient check matter if the code is short?</summary>

A sign error or a missing division by n produces a loop that still decreases the loss slowly, and you would never know. Finite differences catch it in one line.

</details>

**Misconceptions.**

- 'A bigger learning rate learns faster.' Past a limit set by the curvature it diverges; the loss goes up, not down.
- 'The weights say which feature matters most.' Only after standardization, and only for the linear part; correlated features share weight arbitrarily.
- 'Logistic regression is obsolete.' It is the calibration-friendly baseline every other model must beat, and it is inside every neural network's output layer.

### Read and watch, explained at every level

#### Khan Academy: Linear algebra, Differential calculus, Statistics and probability

_free; do the exercises, skip nothing marked 'gradient'_

**Age 5.** Video lessons and practice questions that teach math one small step at a time, with a friendly voice, and you earn points for practicing.

**Middle school.** Three free courses, each made of short videos and exercises that get harder as you improve. Linear algebra is about vectors and grids of numbers, calculus is about slopes and how things change, and statistics is about averages, spread and chance. Do the exercises after every video; the videos alone will not stick.

**High school.** Use them selectively rather than front to back. In linear algebra: vectors, dot products, matrix multiplication. In differential calculus: derivatives, the chain rule, and then the multivariable course's unit on partial derivatives and the gradient. In statistics: distributions, the normal curve, sampling, and probability rules including conditional probability. Skip nothing with the word gradient in it; that is the heart of lesson 7.

**Stanford undergrad.** Khan is remedial in depth but excellent for gaps, and its mastery system is spaced repetition, which is why it works. Pair each unit with the corresponding chapter of Mathematics for Machine Learning to see the same idea at university level, and use Khan's diagnostics to find the specific skill that is rusty rather than redoing a whole subject.

**Stanford grad.** Its limitation is proof and abstraction: it teaches computation, not why the computation is right. For that, Mathematics for Machine Learning and 3Blue1Brown. It remains the fastest way to rebuild one specific skill, such as partial derivatives, in an evening before you need it.

**PhD at Apple.** For a research group, mastery-style practice is an underrated onboarding tool for people arriving from adjacent fields. Its per-skill diagnostics let you prescribe precisely (three units, one evening) instead of sending someone away for a semester.

#### 3Blue1Brown: Essence of linear algebra, Essence of calculus

_free; the intuition the exercises lack_

**Age 5.** Videos with beautiful moving pictures that show what math looks like, so arrows and slopes become things you can see instead of just numbers.

**Middle school.** Two YouTube series by Grant Sanderson. The linear algebra one shows vectors as arrows and matrices as ways of stretching and turning space. The calculus one shows slopes and areas built from tiny pieces. Watch each video, then pause and redraw the picture yourself on paper; the drawing is where it sinks in.

**High school.** Watch the linear algebra series before matrix homework: it turns matrix multiplication into composing transformations, the determinant into a scaling factor, and eigenvectors into directions that do not turn. The calculus series makes the chain rule and the derivative of e to the x intuitive rather than memorized. Neither replaces practice; both make practice make sense.

**Stanford undergrad.** The series supply the geometric intuition textbooks omit: the dot product as projection, the derivative as a local linear approximation, change of basis as a change of viewpoint. That intuition is what you will lean on when reading about singular value decomposition, principal components and Jacobians, and when debugging a shape error in a matrix product.

**Stanford grad.** Sanderson's method, visual first and symbols second, is a model for explaining your own work. His later videos on neural networks and on the Fourier transform extend the same approach to topics you meet in phases 6 and 8, and his video on backpropagation is the one to show someone who has read the equations and still does not see it.

**PhD at Apple.** The channel is evidence that rigorous intuition scales to millions of viewers. For someone who will teach and present, the animation library he built (manim) is also a practical tool for figures in talks, and his explanations of attention are a good benchmark for how clearly a research idea can be communicated.

#### Mathematics for Machine Learning (Deisenroth, Faisal, Ong), chapters 2–7

_free PDF; reference, not a read-through_

**Age 5.** A big free book about the math inside learning machines. It is for looking things up, not for reading from the first page to the last.

**Middle school.** A university textbook, free as a PDF. Chapters 2 to 7 are linear algebra, geometry, matrix decompositions, vector calculus, probability, and optimization. When a lesson mentions a word you do not know, find it in the index and read that section. The pictures and worked examples are good; the exercises are hard and worth trying.

**High school.** Use it as the reference after Khan Academy and 3Blue1Brown. Chapter 5 on vector calculus explains gradients and the chain rule in the form machine learning uses; chapter 6 covers distributions, expectations and Bayes; chapter 7 explains gradient descent and convexity. Do the exercises for chapters 5 and 7; they are the two chapters that lesson 7 and lesson 9 lean on.

**Stanford undergrad.** The book's second half applies the math: work chapter 9 on linear regression fully to see maximum likelihood, maximum a posteriori and Bayesian regression in one place, then chapter 10 on principal components for the decomposition view. Its notation matches most ML papers, which makes it the bridge from courses to the literature.

**Stanford grad.** Coverage stops before measure-theoretic probability and before optimization theory proper (convergence rates, duality in depth). For those, Boyd and Vandenberghe on convex optimization and a graduate probability text. It is the right prerequisite for Murphy's Probabilistic Machine Learning, and its exercises are a reasonable self-test before that book.

**PhD at Apple.** As a reference for writing, it fixes consistent notation for gradients, Jacobians and distributions that reviewers recognize. Its free availability makes it the safe citation for prerequisites in a paper or an internal design document, and its chapter on optimization is a compact refresher before reading anything on second-order methods.

#### Think Stats (Downey)

_free; statistics with Python from the first page_

**Age 5.** A free book that teaches statistics by playing with real numbers in Python, instead of only formulas.

**Middle school.** Allen Downey's book uses real data and Python code from the first chapter: distributions, averages, spread, probability, and how to tell whether a difference is real or just chance. Free online. Run the code as you read; each chapter's exercises use the same data, so you build up a small toolkit.

**High school.** The approach is computational: build a distribution from data, compute a percentile, run a simulation instead of memorizing a formula. That is exactly how the profile (a median per slot) and the bootstrap (resampling days) work in this project. The chapters on distributions, probability, estimation and hypothesis testing are the core; the one on correlation is a good antidote to reading too much into a scatter plot.

**Stanford undergrad.** Downey's simulation-first stance, resampling for confidence intervals and permutation tests for differences, is the modern practical approach and maps directly onto the block bootstrap for average-precision intervals. The treatment is light on theory; pair with chapter 6 of Mathematics for Machine Learning for the formal side.

**Stanford grad.** Compare its resampling viewpoint with classical inference and understand when each is valid: the bootstrap fails for dependent data unless it resamples blocks, which is why lesson 6 bootstraps over days. Downey's Think Bayes is the natural next book for the uncertainty track in phase 8.

**PhD at Apple.** The book is an argument that computation has changed how statistics should be taught, and it is right. For someone designing evaluation methodology, its emphasis on effect sizes and simulation over p-values is the correct instinct, and its examples are a template for the kind of small, honest analysis that should accompany every metric in a report.

### Build

- Derive the logistic regression gradient on paper, then check it numerically with finite differences against the repo's update.
- Write the training loop from a blank file and match the repo's weights to four decimals.
- Compute the median and the 85th percentile of a real speed column with numpy, by hand and with the library, and write a notebook page titled 'why the median' for stage 6.
- Plot the sigmoid and the log loss; explain in one paragraph why the loss is not squared error.

### Done when

- Your loop reproduces `bottleneck_weights.npz` and you can say what each weight means.
- You can derive the gradient on a whiteboard without notes.

### Pitfalls

- Trying to finish all of math first. This phase runs alongside phases 1 and 3 on purpose.
- Memorizing formulas you have not derived once.

## Phase 3: Data handling at scale

_weeks 8–14 · ≈50 h · builds stage 5 for real_

**Goal.** Build stage 5, the cleaning stage, on a real PeMS file, inside 2 GB.

### What you need to learn

- numpy: arrays, dtypes, broadcasting, memory layout; why float32 halves everything.
- pandas: Series and DataFrame, indexing, `groupby`, `merge`, time indexes, `shift`, `rolling`, `resample`, categoricals, `read_csv` with `chunksize`, Parquet through pyarrow.
- Time zones and daylight saving in practice: `tz_localize`, ambiguous and nonexistent times.
- Data quality: missingness, bounds, duplicates, gaps, and PeMS's `% Observed` column.
- Plotting with matplotlib: line, histogram, scatter; reading a plot before trusting a number.
- Measuring memory and time; polars as a second, faster tool for the same job.

### Lesson 3: Cleaning: from a giant text file to a tidy table

_Lives in stage 5 · 10_clean_pems.py._

**Goal.** Turn 200 MB of headerless text a day into a small, correct, gap-free table without ever loading the whole file.

**Age 5.**

The road sensors send a huge messy pile of numbers, like a toy box with everything dumped in at once.

Cleaning means putting each toy in its bin, throwing out the broken ones, and noticing when a piece is missing so we do not pretend it is there.

When it is done, the pile is a neat shelf where we can find anything fast.

_Try this._ Sort a pile of mixed building blocks by color. Count each color. Set aside any broken piece. Which color had the most?

**Middle school.**

One day's PeMS file is about 200 MB of text. We read it a chunk at a time so it never fills the memory, and we keep only the Fremont stations, which throws away about 97 % of the rows straight away.

Then we convert the times to a common clock (UTC), throw out impossible numbers such as a speed of 250 mph, mark rows that PeMS mostly made up as missing, and fill short gaps of up to 15 minutes by repeating the last value.

We also make sure every station has a row for every 5-minute slot, even if it is blank, so that 'six rows ago' always means 'thirty minutes ago'.

Finally we save the result in Parquet, a format that stores each column separately and loads in a fraction of a second.

_Try this._ In a spreadsheet, write 24 hourly temperatures. Delete two of them. What is a fair way to fill the holes? Now delete ten in a row. Is filling still fair?

**High school.**

Nine steps. Build the station registry from the metadata file: freeway, direction, postmile, coordinates, lanes, and from the postmile order, each station's upstream and downstream neighbor. Read each gzip in chunks with only the 12 columns you need. Filter to registry station ids inside each chunk. Convert timestamps to UTC with explicit daylight-saving handling. Apply bounds (speed 0 to 100, occupancy 0 to 1, flow up to 3000 per lane). Blank rows with `% Observed` under 50. Forward-fill gaps of at most three steps, within a station only. Reindex every station onto the full 5-minute grid. Write monthly Parquet files in float32.

Why float32: it halves memory and disk, and speed to a hundredth of a mile per hour is more precision than the detector has.

Why the full grid: later, a lag of six rows must mean exactly thirty minutes. If a row is missing, positional shifts silently become the wrong time.

_Try this._ Compute the bytes for 10 million rows × 5 columns at float64 and at float32. Then look up how much memory your machine has and see how many years of each would fit.

**Stanford undergrad.**

The principles are those of any data pipeline: idempotence (re-running yields identical output), streaming over in-memory, columnar storage with predicate pushdown, dtype economy, and validation at the boundary (assert the schema, the value ranges, and 288 rows per station-day).

Daylight saving is the classic correctness trap. `America/Los_Angeles` has a nonexistent hour in March and a repeated hour in November. `ambiguous='NaT'` drops the repeated hour rather than mislabelling half of it; `nonexistent='shift_forward'` handles the spring gap deterministically.

Forward fill must be grouped: `df.groupby('station_id').ffill(limit=3)` so that one station's last value never bleeds into the next station's first missing row. The reindex to the full grid must happen before any positional operation.

Testing: a fixture of a hundred rows that includes a DST boundary, a station boundary and a `% Observed` failure. If the fixture passes, the whole year is likely right; if it fails, you saved a day.

_Try this._ Write a pytest that constructs two stations with a gap at the boundary and asserts that forward fill does not cross it. Make it fail first by removing the groupby.

**Stanford grad.**

Imputation is inference. A zero-order hold assumes the process is locally constant; linear interpolation uses the next observed value and therefore looks ahead, which is inadmissible for features at serving time but harmless for cleaning history if labels never use imputed rows. Decide per column which rule applies and write it down.

Missingness mechanism matters: a power outage is missing completely at random; a loop that fails under heavy, stop-and-go traffic is missing not at random, correlated with exactly the state you want to predict. Dropping such rows biases the positive rate downward; filling them biases speed upward.

The registry's neighbor chain assumes a line. Interchanges (I-880 with SR-84, I-680 with Mission) break it; those need explicit edges or the graph model inherits a wrong prior.

_Try this._ Classify each failure cause (power outage, loop failure under congestion, PeMS server outage, roadwork removing a station) by missingness mechanism and state the direction of bias each induces in label prevalence.

**PhD at Apple.**

At scale this stage is a streaming system with schema evolution, late and corrected data, backfills and lineage. The design that survives is layered: raw is immutable and content-hashed (the manifest in this plan), cleaned is a pure function of raw plus code version, features are a pure function of cleaned. Any month can then be rebuilt bit for bit, which is what makes an evaluation trustworthy a year later.

Detector health deserves a model of its own, with the imputation mask exposed downstream as a feature. On unified-memory hardware the cleaning stage is memory-bandwidth-bound long before it is compute-bound; profile with real files rather than reasoning about big-O.

Research threads: learned spatio-temporal imputation versus the downstream model learning the imputer; treating the observation mask as data; and reproducible backfills when the upstream provider re-publishes a corrected day.

_Try this._ Design the backfill procedure for a corrected PeMS day: what is recomputed, what is invalidated (features, models, logged metrics), and how the manifest records it.

**Guided walkthrough.**

1. Build the registry once from the meta file and derive neighbors from postmile order.

   ```python
   meta = pd.read_csv("data/pems_meta/d04_text_meta_2026_03_01.txt", sep="\t")
   fre = meta[(meta.Fwy.isin([880, 680, 84])) & meta.Latitude.between(37.447, 37.612)
              & meta.Longitude.between(-122.075, -121.888) & (meta.Type == "ML")].copy()
   fre = fre.sort_values(["Fwy", "Dir", "Abs_PM"])
   fre["upstream_id"]   = fre.groupby(["Fwy", "Dir"]).ID.shift(1)
   fre["downstream_id"] = fre.groupby(["Fwy", "Dir"]).ID.shift(-1)
   fre.to_parquet("data/pems_meta/fremont_stations.parquet")
   ```

2. Stream one day, keeping only registry stations in each chunk.

   ```python
   keep = set(fre.ID)
   parts = []
   for chunk in pd.read_csv(path, header=None, usecols=range(12), names=cols, chunksize=200_000):
       parts.append(chunk[chunk.station.isin(keep)])
   day = pd.concat(parts, ignore_index=True)
   ```

3. Convert time, apply bounds, blank imputed rows.

   ```python
   day["timestamp"] = (pd.to_datetime(day.ts, format="%m/%d/%Y %H:%M:%S")
       .dt.tz_localize("America/Los_Angeles", ambiguous="NaT", nonexistent="shift_forward").dt.tz_convert("UTC"))
   day.loc[~day.speed.between(0, 100), "speed"] = np.nan
   day.loc[day.pct_obs < 50, ["speed", "occ", "flow"]] = np.nan
   ```

4. Reindex onto the full grid per station, then forward-fill within station.

   ```python
   grid = pd.MultiIndex.from_product([sorted(keep), pd.date_range(day.timestamp.min().floor("D"), periods=288, freq="5min", tz="UTC")],
                                     names=["station_id", "timestamp"])
   tidy = day.set_index(["station", "timestamp"]).rename_axis(["station_id", "timestamp"]).reindex(grid)
   tidy[["speed", "occ", "flow"]] = tidy.groupby(level="station_id")[["speed", "occ", "flow"]].ffill(limit=3)
   tidy = tidy.astype({"speed": "float32", "occ": "float32", "flow": "float32"})
   ```

5. Validate before writing: 288 rows per station, values in range, and log memory and seconds.

   ```python
   assert (tidy.groupby(level="station_id").size() == 288).all()
   tidy.reset_index().to_parquet("data/pems_processed/2026-03.parquet")
   ```

**Check yourself.**

<details>
<summary>Why filter inside each chunk rather than after concatenating?</summary>

Concatenating first means holding the whole district-day in memory, about 1.3 million rows and every column. Filtering per chunk keeps the peak under a few hundred megabytes.

</details>

<details>
<summary>What goes wrong if you forward-fill without grouping by station?</summary>

The last value of station A fills the first missing rows of station B, inventing data that looks real and is completely wrong.

</details>

<details>
<summary>Why drop the ambiguous November hour instead of resolving it?</summary>

Half of those rows belong to the first 1 am and half to the second; the file does not say which. Guessing mislabels an hour of data every year; dropping loses one hour per year, honestly.

</details>

**Misconceptions.**

- 'Parquet is just a faster CSV.' It is columnar and typed; the dtypes you write are the dtypes you get back, and only the columns you ask for are read.
- 'Filling gaps is harmless.' Every fill is a model; a fill that looks ahead leaks the future into features.
- 'Cleaning is a one-off.' It is the stage that runs on every new day forever, and the one whose silent errors are costliest.

### Read and watch, explained at every level

#### Python for Data Analysis, 3rd edition (McKinney), chapters 4–8 and 10–11

_free online; the pandas book by pandas's author_

**Age 5.** A book by the person who invented pandas, the tool that turns big tables of numbers into things you can sort, count and draw.

**Middle school.** Wes McKinney created pandas and wrote this book about it. Chapters 4 to 8 teach numpy and pandas basics: arrays, tables, reading files, cleaning messy values, and combining tables. Chapters 10 and 11 teach grouping and time series, which this project uses everywhere. It is free online; do the examples with the project's own data instead of the book's.

**High school.** The essential chapters for stages 5 and 8: 5 (pandas basics), 7 (cleaning: missing data, duplicates, transformations), 8 (merging and reshaping), 10 (groupby, the split-apply-combine that builds the speed profile), and 11 (time series: resampling, shifting, rolling windows, time zones). Do chapter 11 twice; every lag, rolling mean and DST fix in this pipeline comes from it.

**Stanford undergrad.** The book explains pandas' model (index alignment, dtypes, categoricals) so you can predict behavior instead of guessing. Understanding index alignment prevents the silent misjoins that plague feature engineering, and the numpy chapter's broadcasting and vectorization are the reason loops are avoided. Its appendix on advanced numpy is worth reading before lesson 9's hand-written backward pass.

**Stanford grad.** Its limits are scale and the newest tooling: little on Arrow-backed dtypes, nothing on polars or out-of-core processing. Read the pandas documentation on Copy-on-Write and Arrow strings, and the polars guide, to see where the ecosystem is heading. The book's groupby chapter remains the clearest explanation of split-apply-combine in print.

**PhD at Apple.** pandas' design decisions (label alignment, mutable frames, Python-level groupby) explain its performance ceiling and why Arrow-native tools win the cleaning stage while pandas keeps the analysis stage. McKinney's later work on Apache Arrow is the continuation of the story, and understanding both is what lets you choose a tool per stage rather than per habit.

#### pandas user guide: Time series and date functionality

_free; the `shift` and `rolling` semantics live here_

**Age 5.** The manual page for the part of pandas that understands dates, times and clocks.

**Middle school.** One long page in the pandas documentation. It shows how to make a range of 5-minute timestamps, shift values forward and back, compute rolling averages, and convert between time zones. Read it with a notebook open and try each example on one station's data from the project.

**High school.** The sections you need: date ranges and frequency strings (`freq='5min'`), shifting, `rolling` windows, `resample`, and the time-zone section covering `tz_localize`, `tz_convert`, and the `ambiguous` and `nonexistent` arguments. This page is where the daylight-saving behavior in stage 5 is specified, so read that section with the November example in front of you.

**Stanford undergrad.** Understand the difference between a positional shift and a time-based shift, and why the full-grid reindex in stage 5 makes them agree. Also the exact semantics of `rolling`: windows are right-aligned by default, `center=True` looks ahead and is therefore forbidden for features, and `min_periods` controls when the window starts producing values.

**Stanford grad.** The guide documents edge behaviors (periods versus timestamps, offsets, business-day calendars) that matter for holiday and weekday features, and the performance notes on tz-aware columns. Read the Copy-on-Write page alongside; it changed chained-assignment semantics and quietly breaks old code.

**PhD at Apple.** The precise semantics of window operations are what point-in-time correctness rests on. For an engineering organization the right move is to encode them once in a feature library with tests rather than rereading the guide per project, and to treat any window parameter as part of the model's versioned configuration.

#### Kaggle Learn: Pandas, Data Cleaning, Data Visualization

_free, short_

**Age 5.** Three short online lessons about sorting tables, tidying up messy numbers, and drawing charts from them.

**Middle school.** Browser lessons with instant grading. Pandas: selecting, grouping and summarizing tables. Data Cleaning: missing values, scaling, parsing dates, fixing inconsistent text. Data Visualization: line charts, bar charts and histograms with the seaborn library. About an afternoon each; do them between chapters of McKinney.

**High school.** They rehearse exactly the operations of stage 5 (missing data, parsing dates, deduplication) and the plots you will make of a station's day and week. Their datasets are small and tidy compared with a PeMS file, so treat them as syntax practice and then confront the real 200 MB file, where the same operations need chunking and care.

**Stanford undergrad.** Use them for the mechanics and then go to the source: the matplotlib documentation for control over figures, and a book such as Fundamentals of Data Visualization (Wilke, free online) for how to design a plot that communicates. Learn to read a plot before trusting a metric; the fundamental-diagram scatter and the daily speed line should become habits.

**Stanford grad.** The visualization course stops at defaults. The skills that matter for this project are diagnostic plots: reliability diagrams, lead-time histograms, per-station error maps, and autocorrelation plots. Build a small plotting module for the project so every evaluation produces the same figures.

**PhD at Apple.** Visualization is a research instrument, not decoration. A reliability diagram or a lead-time distribution communicates what a table of metrics hides, and the habit these courses seed, plotting before concluding, is worth more than their content. Wilke's book and Tufte remain the references for the design side.

#### The polars user guide, first three chapters

_free; do stage 5 once in each_

**Age 5.** The manual for a newer, faster tool that does the same job as pandas: turning big tables into answers.

**Middle school.** Polars is a table library written in the Rust language. It is often several times faster than pandas and uses less memory, which matters on a 16 GB machine. The first chapters cover expressions (how you describe what you want), selecting and filtering rows, and grouping. Do stage 5 once in pandas and once in polars, and time both.

**High school.** Its model differs from pandas: expressions instead of index alignment, lazy evaluation where `scan_csv` plans the whole query and runs it once, strict dtypes, and no index. For streaming a district-day file and keeping a hundred stations, the lazy scan with a filter is exactly right and runs in a few seconds where pandas takes twenty.

**Stanford undergrad.** Learn the lazy API and streaming collection, and understand why immutability and the absence of an index make parallel execution straightforward. Interoperation goes through Apache Arrow, so `to_pandas` and `from_pandas` are cheap or zero-copy for most dtypes, which lets you use polars for the heavy stage and pandas for analysis without friction.

**Stanford grad.** Polars represents the Arrow-native, query-optimized direction of data tooling, alongside DuckDB. For pipelines it is the better default; for ad hoc analysis pandas still has the larger ecosystem. Know both and choose per stage; the guide's chapter on the query optimizer explains why the same operation costs so differently.

**PhD at Apple.** The performance gap is architectural (columnar memory, SIMD, multithreading, query optimization) rather than incidental, and on unified-memory hardware both libraries end up limited by memory bandwidth. Benchmark on the real file before choosing, and note that the decision is per stage: the cleaning stage is a query, the modelling stage is not.

### Build

- Run stage 4 for a few days of files. Open one, confirm the column layout against the documented one, and write it into the stage 5 README.
- Write `10_clean_pems.py`: chunked reading, the Fremont station filter from the meta file, DST-safe UTC conversion, bounds, `% Observed`, forward fill of at most three steps, the full 5-minute grid, float32, monthly Parquet.
- Build the station registry with upstream and downstream neighbors from postmiles.
- Plot one station for a day and a week; you should see two rush hours on weekdays and none on Sunday.
- Write `tests/test_clean_pems.py` on a 100-row fixture, including the November repeated hour.
- Log memory and seconds per file; redo it in polars and compare.

### Done when

- One month of Fremont Parquet exists with the plan's schema and the plot shows rush hours.
- The DST test passes and the script stays under 2 GB.

### Pitfalls

- Loading a whole district-day at once.
- `object` dtypes and naive timestamps.
- Forward-filling across a station boundary, which invents data.

## Phase 4: Classical machine learning

_weeks 13–21 · ≈70 h · builds stages 6, 7, 9, 10_

**Goal.** Labels, the incident join, baselines, logistic regression, gradient boosted trees, calibration, and honest row-level evaluation.

### What you need to learn

- The supervised setup: features, target, train, validation, test, and why three.
- Overfitting, bias and variance, regularization, early stopping.
- Metrics: confusion matrix, precision, recall, F1, the precision-recall curve and average precision, ROC-AUC, log loss, Brier score, reliability diagrams; why accuracy lies under imbalance.
- Leakage in its three forms: target, temporal, group. Class weighting.
- Linear and logistic regression with scikit-learn; decision trees; random forests; gradient boosting, and what `num_leaves`, `learning_rate` and `min_data_in_leaf` do in LightGBM; categorical handling.
- Feature engineering as hypothesis-making; feature importance and SHAP values.
- Isotonic and Platt calibration; choosing a threshold from a curve; hyperparameter search with a budget.

### Lesson 4: Labels: what does 'slow' mean?

_Lives in stage 6 · 11_build_labels.py._

**Goal.** Define the target precisely, compute it without leaking, and understand that the definition decides what the model can mean.

**Age 5.**

How do we know a road is slow? We compare it with how it usually is.

If cars usually zoom at 65 and today they crawl at 30, that is slow. If they always crawl at five o'clock, that is just five o'clock.

So the helper first learns what 'usual' looks like for every road and every time of day, and then notices when today is different.

_Try this._ Is it cold today? Compared with what: yesterday, summer, or winter? Notice that the answer changes with what you compare to.

**Middle school.**

For every station and every 5-minute slot of the day, separately for weekdays and weekends, we find the typical speed by taking the median of all the days we have. The median is the middle value when you line the numbers up, so a couple of crazy days do not drag it around the way an average would.

Then 'slow' means today's speed is below 60 % of that typical speed. A slow episode is three slots in a row that are slow, fifteen minutes, so one odd reading does not count.

That definition has a surprise inside it: the evening rush hour, which is slow every day, is not 'slow' by this rule, because slow is normal then. The rule finds unusual slowness.

_Try this._ Line up these speeds: 62, 64, 65, 12, 66, 63, 65. Find the median and the mean. Which one is a better 'typical'? Why?

**High school.**

Three formulas. profile(s, slot, daytype) = median of speed over training days. ratio = speed / profile. is_slow = ratio < 0.6, and label = 1 when is_slow holds for three consecutive steps, the window ending at the row's own time.

Two design choices with consequences. Relative to profile: the label flags unusual slowness, so recurrent congestion is invisible to it. Absolute (speed below 35 mph for three steps): the label flags congestion itself, rush hour included. The plan computes both and makes you pick.

Details that matter: 5-minute slots instead of hours, because a 4 pm median mixes free flow at 4:00 with a jam at 4:55; and free-flow speed is better measured as the station's 85th percentile than as a median, which is why the plan adds `ratio_ff` too.

_Try this._ Make up one day of speeds for a station with a 5 pm rush hour. Compute both labels by hand. Where do they disagree, and which one would a driver want?

**Stanford undergrad.**

The label is a deterministic function of the series and an estimated profile, so the profile is part of the model. Estimating it on all data lets the test period influence training labels: target leakage through the label definition. The profile must be fit on training timestamps only and stored as an artifact.

Quantile estimation: the median has a 50 % breakdown point and, for n days per cell, a standard error near 1.25 σ/√n; with eight weeks of weekdays that is n ≈ 40, so smoothing across neighboring slots trades a little bias for a useful variance reduction, and cells below 20 samples fall back to the hourly median.

Consecutive-run labels: label_t = Π_{k=0..2} is_slow_{t−k}. For horizon h = 3 steps, the label at t + 3 depends on speeds at t + 1, t + 2, t + 3 and does not touch the features at t. Prevalence is a steep function of the 0.6 cutoff and the run length; sweep both once and freeze them in the filename.

_Try this._ Show that for a horizon of two steps the label window at t + 2 would include the row at t itself, overlapping the feature `speed_lag_0`, and explain what that does to a naive evaluation.

**Stanford grad.**

Label design is a modelling decision. The relative label approximates conditional anomaly detection; the absolute label approximates a level target. They induce different class balance, different predictability (recurrent congestion is calendar-predictable, non-recurrent is not), and different product semantics, and a model good at one can be useless at the other.

Alternatives keep more information: regress the speed ratio with a pinball loss and derive probabilities from quantiles; or model episode onset as a point process with covariates and predict hazard. Both avoid the brittle 3-step run and make horizon coherence natural.

Errors-in-variables: detector noise near the 0.6 boundary is heteroscedastic (single-loop speed error grows at low flow), so label noise is concentrated exactly where the decision is made. And because the profile pins the training positive rate near a quantile by construction, real seasonal shifts in congestion are masked in the training window.

_Try this._ Derive how a 5 % multiplicative bias in speed estimation shifts the positive rate near the threshold, given a local density of the ratio around 0.6, and propose a check using stations with double loops.

**PhD at Apple.**

Production traffic systems keep the modelled quantity continuous (speed, travel time) and push any binary definition into a presentation policy. One model then serves several products, and changing what 'slow' means does not trigger retraining or invalidate history.

Research angles: learning when the target itself depends on estimated quantities; semi-supervised regime detection on the fundamental diagram (hidden Markov or change-point models that find breakdown without a hand threshold); and measuring label quality through agreement across definitions and across detector types.

Operationally, version the label definition as you would a schema. A label change invalidates every logged metric and every comparison across model generations.

_Try this._ Propose an A/B design that compares two label definitions on a downstream user metric such as reroute acceptance, and state what confounds the comparison.

**Guided walkthrough.**

1. Choose the split dates first; the profile uses training rows only.

   ```python
   train_end = pd.Timestamp("2026-08-31", tz="UTC")
   tr = df[df.timestamp <= train_end].copy()
   ```

2. Compute the profile per station, 5-minute slot and day type.

   ```python
   def daytype(ts):
       d = ts.dt.dayofweek
       return np.where(d < 5, "wd", np.where(d == 5, "sat", "sun"))
   for x in (df, tr):
       x["slot"] = (x.timestamp.dt.tz_convert("America/Los_Angeles").dt.hour * 12 + x.timestamp.dt.tz_convert("America/Los_Angeles").dt.minute // 5)
       x["daytype"] = daytype(x.timestamp.dt.tz_convert("America/Los_Angeles"))
   prof = tr.groupby(["station_id", "daytype", "slot"]).speed.median().rename("profile_speed").reset_index()
   prof.to_parquet("models/profile_train2026.parquet")
   ```

3. Join the profile back and compute the ratio and the run-based label.

   ```python
   df = df.merge(prof, on=["station_id", "daytype", "slot"], how="left")
   df["speed_ratio"] = df.speed / df.profile_speed
   df["is_slow"] = (df.speed_ratio < 0.6).astype("float32")
   df = df.sort_values(["station_id", "timestamp"])
   df["label"] = (df.groupby("station_id").is_slow.rolling(3).sum().reset_index(level=0, drop=True) == 3).astype("int8")
   ```

4. Compute the absolute label and the free-flow ratio alongside.

   ```python
   df["label_abs"] = (df.groupby("station_id").speed.transform(lambda s: (s < 35).rolling(3).sum()) == 3).astype("int8")
   ff = tr.groupby("station_id").speed.quantile(0.85).rename("free_flow_85")
   df = df.merge(ff, on="station_id", how="left"); df["ratio_ff"] = df.speed / df.free_flow_85
   ```

5. Look at prevalence by hour for both labels. The relative label should be flat across the day; the absolute label should peak at rush hour. If not, something is wrong.

   ```python
   print(df.groupby(df.timestamp.dt.tz_convert("America/Los_Angeles").dt.hour)[["label", "label_abs"]].mean())
   ```

**Check yourself.**

<details>
<summary>Why does the relative label ignore the rush hour?</summary>

Because the profile is the typical speed at that slot, and at 5 pm the typical speed is already slow. A ratio near 1 is 'normal for now', whatever the absolute speed.

</details>

<details>
<summary>What happens to the label if the profile is computed on all data?</summary>

Test-period speeds shape the profile that defines training labels, a leak. The evaluation then overstates skill in a way that is very hard to see.

</details>

<details>
<summary>Why three consecutive steps?</summary>

To reject single noisy readings. It also shifts the label window: the label at t + h depends on speeds at t + h − 2 through t + h, which must not overlap the features at t.

</details>

**Misconceptions.**

- 'The median profile is the free-flow speed.' It is the typical speed, which at rush hour is far below free flow. Free flow is the 85th percentile.
- 'Threshold 0.6 is a fact.' It is a choice that sets prevalence and difficulty; version it.
- 'A label is ground truth.' It is a derived quantity with its own estimation error, concentrated at the boundary.

### Lesson 5: Features: turning the past into columns

_Lives in stage 8 · 13_feature_engineering.py._

**Goal.** Build about sixty columns that a model can use, every one of them known at prediction time, and the targets shifted into the future.

**Age 5.**

To guess what the road will do, we look at clues: how fast it was a minute ago, how fast the road just ahead is, whether there was a crash, whether it is a school day, whether it is raining.

Each clue is a number we write in a row. Lots of rows make a table. The helper learns which clues matter by reading the table.

_Try this._ What clues tell you it is almost dinner time? A smell, the clock, your tummy. Which clue is the most reliable?

**Middle school.**

A feature is one column of clues. The row for a station at a time has about sixty of them: the speed now and 5, 10, 15, 30, 60 minutes ago (lags); the average of the last 30 minutes (a rolling mean); the speed at the next station downstream and upstream; whether an incident is active nearby and how far away; the time of day, the day of the week, holidays, and rain.

The one rule: every clue must be something you could actually know at that moment. Speed ten minutes from now is not a clue, it is the answer.

The target columns are the answers, copied from the future: the label 15, 30 and 60 minutes later.

_Try this._ Which of these is a legal clue at 4:00 pm for predicting 4:30? Speed at 3:55. Speed at 4:10. Tomorrow's weather. Today's date. Yesterday's speed at 4:30. Say why for each.

**High school.**

Lags come from a shift within each station on the full grid: `df.groupby('station_id')['speed'].shift(k)`. Rolling statistics from `.rolling(6).mean()` where 6 steps is 30 minutes. The slope is `speed_lag_0 − speed_lag_3`.

Time is a circle: 23:55 is next to 00:00, but the number 1435 is far from 0. So minute-of-day becomes two columns, sin(2π m / 1440) and cos(2π m / 1440), which put midnight next to itself. Weekday gets the same treatment.

Neighbors come from the registry: look up the upstream and downstream station ids and merge their speed at the same timestamp. Incidents come from lesson 7's join. Targets are `label.shift(−h)` per station, where a negative shift looks forward.

Before training, one assert: no feature column may use any row after t. If you cannot say how a column is computed from the past alone, delete it.

_Try this._ Draw the sine and cosine of the hour on a circle for hours 0, 6, 12, 18, 23. Why do you need both columns rather than just one?

**Stanford undergrad.**

Every feature must be F_t-measurable, a function of information available at t. That single principle rules out centered rolling windows, target encodings computed on the same rows, and profiles fit on the future.

Mechanics: `groupby` before every shift and rolling to prevent cross-station contamination; `min_periods` to control early-window NaNs; the full grid so a positional shift is a time shift. Trees consume NaN natively and learn a default direction; linear models and networks need imputation plus a missing indicator, and standardization with training-set statistics only.

Categorical station identity: native categorical for LightGBM, one-hot or embedding elsewhere. Interactions (rain × rush hour) can be hand-built for linear models; trees discover them.

Expect the correlation between `speed_lag_0` and the target to fall with the horizon. That decay is the shape of the problem: momentum dominates at 15 minutes, neighbors and calendar at 60.

_Try this._ Compute the Pearson correlation of `speed_lag_0` with `y_15`, `y_30`, `y_60` on one month and plot it. Then do the same for `down1_speed`. Which decays slower, and why?

**Stanford grad.**

Feature sets carry bias-variance trade-offs that depend on the learner: a dozen correlated lags inflate variance for a linear model through multicollinearity and cost nothing for trees. Rolling summaries plus a few lags are the robust compromise.

Neighbor features encode a hand-specified graph prior (one or two hops along the postmile line); the spatio-temporal network learns the graph instead. Both are legitimate; the hand prior is cheaper to monitor.

Leakage is subtle: a centered rolling mean uses t + 1 through t + k; a target encoding of `station_id` must be computed out-of-fold in time; weather joined on the realized hour is fine for training but the serving loop only has the forecast, so training should use the forecast issued at t too, or the model will over-trust weather.

Lag features are stationary by construction; calendar and static features drift with roadwork and detector replacement.

_Try this._ Construct a toy series, compute a centered 30-minute rolling mean, use it as a feature for a 15-minute target, and quantify the inflation in average precision compared with the trailing mean.

**PhD at Apple.**

The industry problem this stage solves is point-in-time correctness: computing training features exactly as they would have been available at serving time. Feature stores with time-travel joins exist for this; the plan's answer at small scale is one feature module used by both paths and a golden-file test that compares them bit for bit.

Representation learning replaces hand features at scale (learned embeddings for segments, time and weather), yet hand features remain the strongest baseline for tabular pipelines and are far cheaper to monitor, explain and debug. Choosing which to ship is an engineering decision about maintenance, not only accuracy.

Research threads: causal feature selection under shift (incident features and congestion share a latent cause), invariance-based training, and the marginal value of exogenous streams such as roadwork feeds, event calendars and school schedules.

_Try this._ Specify the point-in-time join for the weather forecast: the feature at t is the forecast for hour t + 1 issued at or before t, not the realized weather. Say how you would store forecast vintages to make training match serving.

**Guided walkthrough.**

1. Lags and rolling statistics, grouped by station, on the full grid.

   ```python
   g = df.groupby("station_id")
   for k in (0, 1, 2, 3, 6, 12):
       for c in ("speed", "occ", "flow"):
           df[f"{c}_lag_{k}"] = g[c].shift(k).astype("float32")
   df["speed_roll_mean_30"] = g.speed.transform(lambda s: s.rolling(6, min_periods=3).mean()).astype("float32")
   df["speed_roll_std_30"]  = g.speed.transform(lambda s: s.rolling(6, min_periods=3).std()).astype("float32")
   df["speed_slope_15"] = df.speed_lag_0 - df.speed_lag_3
   ```

2. Cyclic time and calendar flags.

   ```python
   import holidays
   loc = df.timestamp.dt.tz_convert("America/Los_Angeles")
   m = loc.dt.hour * 60 + loc.dt.minute
   df["min_sin"] = np.sin(2 * np.pi * m / 1440).astype("float32"); df["min_cos"] = np.cos(2 * np.pi * m / 1440).astype("float32")
   df["dow_sin"] = np.sin(2 * np.pi * loc.dt.dayofweek / 7).astype("float32"); df["dow_cos"] = np.cos(2 * np.pi * loc.dt.dayofweek / 7).astype("float32")
   us_ca = holidays.US(state="CA", years=[2026])
   df["is_holiday"] = loc.dt.date.isin(us_ca).astype("int8")
   ```

3. Neighbor features from the registry.

   ```python
   reg = pd.read_parquet("data/pems_meta/fremont_stations.parquet")[["ID", "upstream_id", "downstream_id"]]
   cur = df[["station_id", "timestamp", "speed", "occ"]]
   for side in ("upstream", "downstream"):
       nb = cur.rename(columns={"station_id": f"{side}_id", "speed": f"{side[:2 if side=='upstream' else 4]}1_speed", "occ": f"{side[:2 if side=='upstream' else 4]}1_occ"})
       df = df.merge(reg[["ID", f"{side}_id"]].rename(columns={"ID": "station_id"}), on="station_id", how="left")
       df = df.merge(nb, on=[f"{side}_id", "timestamp"], how="left")
   ```

4. Targets shifted forward, then drop rows whose target is unknown.

   ```python
   for h, steps in ((15, 3), (30, 6), (60, 12)):
       df[f"y_{h}"] = g.label.shift(-steps)
   train_table = df.dropna(subset=["y_15", "y_30", "y_60"])
   ```

5. The leakage assert: recompute one feature by hand for a random row and compare.

   ```python
   r = train_table.sample(1, random_state=0).iloc[0]
   past = df[(df.station_id == r.station_id) & (df.timestamp < r.timestamp)].tail(3)
   assert np.isclose(r.speed_lag_3, past.speed.iloc[0])   # three rows back
   ```

**Check yourself.**

<details>
<summary>Why sin and cos of the minute rather than the minute itself?</summary>

The minute jumps from 1439 to 0 at midnight; a model would see a cliff. The pair (sin, cos) is continuous around the circle and two columns are needed because sin alone maps two different times to the same value.

</details>

<details>
<summary>Why does the target use a negative shift?</summary>

`shift(−6)` pulls the value six rows later into the current row: the label thirty minutes ahead, aligned with the features at t.

</details>

<details>
<summary>Why keep NaN for LightGBM but impute for the network?</summary>

Trees route missing values down a learned branch; a network multiplies NaN into every weight. Impute, and add an indicator so the network can still know the value was missing.

</details>

**Misconceptions.**

- 'More features always help.' Correlated features hurt linear models and slow trees; every feature is another thing to monitor for drift.
- 'Realized weather is a fine training feature.' At serving time only the forecast exists. Train on what you will have.
- 'Rolling windows are safe.' Only trailing windows are; centered windows look ahead.

### Lesson 6: Baselines and honest evaluation

_Lives in stages 9 and 11 · 14_train_baselines.py, 18_evaluate_lead_time.py._

**Goal.** Score a forecast so that the number means something: against a baseline, on later data, with metrics that imbalance cannot fool.

**Age 5.**

Before we say our helper is smart, we compare it with a very simple guess: 'the road will be like it is right now'. If the helper cannot beat that, it is not helping.

We also test it on days it has never seen, like a real quiz, not on the days it practiced on.

_Try this._ For one week, guess tomorrow's weather as 'same as today'. Keep score. How often is the lazy guess right? That is the score a smart guesser has to beat.

**Middle school.**

Two simple guesses, called baselines. Persistence: whatever the road is doing now, it keeps doing. Profile rate: how often this station is slow at this time of day, on average.

Two measurements. Of all the times we shouted 'slow!', how often was it really slow? That is precision. Of all the real slow times, how many did we catch? That is recall.

Because slow times are rare, maybe 5 % of rows, a helper that always says 'not slow' is 95 % 'accurate' and completely useless. That is why accuracy is not the score.

And we test on later weeks than we trained on, never on the same days, because two rows five minutes apart look almost the same and would make the quiz too easy.

_Try this._ There are 100 rows and 5 are slow. A helper says 'slow' 10 times, and 4 of those are right. Compute precision and recall. Now compute accuracy for a helper that never says 'slow'.

**High school.**

The confusion matrix gives true and false positives and negatives. Precision = TP / (TP + FP); recall = TP / (TP + FN); F1 is their harmonic mean. Sliding the threshold traces the precision-recall curve; its area is average precision (AP), the headline metric here. ROC-AUC exists too but looks flattering under imbalance because true negatives are cheap.

For probabilities: log loss and the Brier score punish confident wrong answers. Calibration asks whether, among rows predicted at about 0.7, about 70 % turn out positive; a reliability diagram shows it.

The split is chronological: train on months 1 to 8, validate on 9 to 10, test on 11 to 12, with a one-day gap so lags and horizons never straddle a boundary. Skill is reported relative to persistence at each horizon.

_Try this._ Rank ten predictions by probability with three positives among them, and compute average precision by hand as the mean of precision at each positive's rank.

**Stanford undergrad.**

Shuffled K-fold is wrong for a time series because autocorrelation places near-copies of every test row in the training fold, and because the estimand is future performance, which only forward validation approximates. Use blocked or rolling-origin validation with a gap of at least the longest lag plus the horizon.

AP equals the mean over positives of the precision at their rank, and its no-skill value equals the prevalence. Log loss and Brier are proper scoring rules: they are minimized in expectation only by the true probabilities, which is what makes them honest. The Brier score decomposes into calibration plus refinement, and the reliability diagram with expected calibration error shows the first term.

Uncertainty: rows are dependent, so bootstrap over blocks (days) rather than rows to get intervals for AP.

_Try this._ Implement rolling-origin evaluation with three folds on one station's data and compare the spread of AP across folds with the spread from shuffled 3-fold. Explain the difference.

**Stanford grad.**

Evaluation is the main scientific control in this project. Effective sample size under spatio-temporal dependence is small; threshold and label sweeps create multiple comparisons; and per-row metrics hide what a user experiences, which is per-episode: lead time from first alert to onset, share of episodes caught, and false alarms per station-day.

Forecasting practice reports skill scores against persistence and climatology (the profile rate); the Brier skill score can be negative while AP looks respectable, and both should be shown. Calibration drifts between validation and test months, so recalibration cadence is itself a design parameter.

A single test window is one draw from a non-stationary process. Rolling folds give a distribution of scores; report it.

_Try this._ Derive the Brier skill score against the profile-rate climatology and construct an example where AP improves over the baseline while the skill score is negative.

**PhD at Apple.**

Offline metrics are proxies. The product metric is what drivers do (reroutes, realized travel time), and its correlation with AP is unknown until measured; A/B tests in traffic suffer interference because routing changes the traffic everyone else sees, a textbook case of performative prediction.

Evaluation infrastructure at scale: metric definitions versioned with the label, backtests reproducible from logged features, slicing by segment, time and weather to surface failure modes, and guardrails such as never regressing worst-slice recall.

Research threads: evaluating forecasts under intervention, distribution-free control of alert rates (risk-controlling prediction sets), and calibration under covariate and label shift using unlabeled target data.

_Try this._ Design an interleaving or switchback experiment that measures the usefulness of alerts without exposing every driver to a possibly bad model, and state the interference assumptions it needs.

**Guided walkthrough.**

1. Chronological split with a one-day gap.

   ```python
   t = train_table.timestamp
   tr = train_table[t <= "2026-08-30"]; va = train_table[(t >= "2026-09-01") & (t <= "2026-10-30")]; te = train_table[t >= "2026-11-01"]
   ```

2. Baselines: persistence and profile rate.

   ```python
   from sklearn.metrics import average_precision_score as ap
   rate = tr.groupby(["station_id", "daytype", "slot"]).y_30.mean().rename("profile_rate").reset_index()
   va = va.merge(rate, on=["station_id", "daytype", "slot"], how="left")
   print("prevalence", va.y_30.mean())
   print("persistence AP", ap(va.y_30, va.label))          # label now as the forecast
   print("profile-rate AP", ap(va.y_30, va.profile_rate.fillna(0)))
   ```

3. Precision-recall curve and a threshold at 0.5 precision.

   ```python
   from sklearn.metrics import precision_recall_curve
   p, r, thr = precision_recall_curve(va.y_30, va.p_30)          # p_30 from any model
   ok = np.where(p[:-1] >= 0.5)[0]
   tau = thr[ok[0]] if len(ok) else 1.0
   print("tau", tau, "recall at tau", r[ok[0]] if len(ok) else 0)
   ```

4. Reliability diagram numbers.

   ```python
   from sklearn.calibration import calibration_curve
   frac_pos, mean_pred = calibration_curve(va.y_30, va.p_30, n_bins=10, strategy="quantile")
   for a, b in zip(mean_pred, frac_pos): print(f"predicted {a:.2f}  observed {b:.2f}")
   ```

5. Block bootstrap over days for an interval on AP.

   ```python
   days = va.timestamp.dt.floor("D").unique()
   rng = np.random.default_rng(0); scores = []
   for _ in range(200):
       pick = rng.choice(days, size=len(days), replace=True)
       sub = pd.concat([va[va.timestamp.dt.floor("D") == d] for d in pick])
       scores.append(ap(sub.y_30, sub.p_30))
   print(np.percentile(scores, [2.5, 97.5]))
   ```

**Check yourself.**

<details>
<summary>Why is persistence such a strong baseline at 15 minutes?</summary>

Speed five minutes ago is highly autocorrelated with speed in fifteen; jams rarely appear from nothing inside one step. Skill over persistence grows with horizon.

</details>

<details>
<summary>Why bootstrap over days rather than rows?</summary>

Rows within a day are dependent; resampling rows pretends they are independent and gives intervals that are far too narrow.

</details>

<details>
<summary>What does a high ROC-AUC with a low AP tell you?</summary>

The model ranks negatives well below positives on average, but the top of the ranking, which is what alerts use, is still polluted with false positives. Under imbalance ROC-AUC is the flattering one.

</details>

**Misconceptions.**

- 'Cross-validation is always better than a single split.' Shuffled CV on a time series leaks and reports fantasy scores; only forward splits are valid here.
- 'Beating the baseline once is enough.' Report it per horizon and per fold; a model can beat persistence at 60 minutes and lose at 15.
- 'Calibration is a detail.' It is the difference between a probability and a score, and the threshold rule assumes a probability.

### Lesson 8: Gradient boosted trees

_Lives in stage 9 · 16_train_gbt.py, LightGBM._

**Goal.** Understand why hundreds of small trees, each fixing the last one's mistakes, are the production model for tabular data, and how to tune them without fooling yourself.

**Age 5.**

A decision tree is a game of twenty questions. Is the road ahead slower than 40? Is it raining? Is it five o'clock? Each answer takes you down a branch to a guess.

Boosting is playing many tiny question-games in a row. Each new game only tries to fix what the games before it got wrong, and at the end all the little answers are added up.

_Try this._ Play twenty questions to guess an animal. Notice that the best questions split the possibilities in half. A bad question ('is it a zebra?') wastes a turn.

**Middle school.**

One tree splits the rows by asking questions about the clues and ends in leaves, each with a score. One tree is rough and jumpy.

Boosting builds hundreds of small trees. Each is trained on the mistakes left over by the ones before, and its score is shrunk by a 'learning rate' before being added, so no single tree can dominate. The sum of all the shrunken trees is the model.

LightGBM is a fast program that does this. It bins each clue into 255 buckets so it can try splits quickly, and it can stop adding trees when the score on held-out data stops improving.

_Try this._ Write ten rows with a speed and whether the road was slow 30 minutes later. Try the split 'speed < 40' and count slow and not-slow on each side. Try 'speed < 55'. Which split separates them better?

**High school.**

A tree chooses the split that most reduces the mix of classes in its leaves (the gain). Depth and the number of leaves control how detailed it can be. Boosting: start with a constant F₀, then repeatedly fit a small tree h_m to the residuals (for logistic loss, y − p) and set F_m = F_{m−1} + η h_m.

The knobs that matter: `num_leaves` (63 here), `learning_rate` (0.05), `min_data_in_leaf` (200, so a leaf cannot fit noise), `feature_fraction` and `bagging_fraction` (randomness that reduces overfitting), and early stopping on validation average precision. `is_unbalance` reweights the rare positives. `station_id` is passed as a categorical, not one-hot.

Thousands of trees at a small learning rate usually beat hundreds at a large one, and early stopping picks the count for you.

_Try this._ Explain in your own words why 3000 trees at η = 0.05 can beat 300 trees at η = 0.5, and what early stopping is protecting against.

**Stanford undergrad.**

Gradient boosting is gradient descent in function space (Friedman, 2001). Modern libraries use the second-order expansion of the loss: with per-row gradient g and Hessian h, a leaf's optimal value is −Σg / (Σh + λ) and a split's gain is the increase in Σ(Σg)² / (Σh + λ) across the two children. For logistic loss g = p − y and h = p(1 − p).

Histogram-based split finding bins features to 255 values (one byte per cell), which sets memory at roughly rows × features bytes; LightGBM grows leaf-wise (best leaf first) rather than level-wise, and adds gradient-based one-side sampling and exclusive feature bundling for speed.

Regularization is λ, minimum child weight, shrinkage and subsampling. Importance by gain is biased toward high-cardinality features; TreeSHAP gives exact per-row attributions.

_Try this._ Derive the optimal leaf weight and the split gain from the second-order Taylor expansion of a general loss, then specialize to logistic loss.

**Stanford grad.**

Boosted trees win on tabular data for identifiable reasons (Grinsztajn et al., 2022): they are invariant to feature scaling and rotation-non-invariant in a way that suits heterogeneous columns, and they are robust to uninformative features. Their failure modes are just as identifiable: piecewise-constant extrapolation outside the training range, sensitivity of split thresholds to distribution shift, and poorly calibrated raw scores, hence the isotonic step.

Monotone constraints encode physics: the probability of a slow episode should not increase with downstream speed. Class weighting, threshold moving and focal loss are three ways to handle imbalance with different effects on calibration.

Hyperparameter search must use the chronological validation split, and choosing among many configurations on that split is itself a fit; nested rolling-origin evaluation is the honest cost.

_Try this._ Show that for a well-specified model class weighting is equivalent to shifting the decision threshold, then explain concretely why the equivalence fails for finite trees with early stopping.

**PhD at Apple.**

In production, boosted ensembles are compiled to branch-free code or shipped on device through Core ML tree ensembles, quantized, and watched for drift through the histogram of values arriving at each split. Retraining at scale uses warm starts and careful control of tree count so that latency budgets hold.

Research threads: margin-based theory of boosting; hybrids with networks (differentiable trees, GBT-initialized MLPs, deep forests); uncertainty for ensembles (natural-gradient boosting, quantile boosting, conformal on top); and the persistent empirical finding that on ten-million-row engineered tabular data a boosted ensemble is the strongest and cheapest baseline.

The senior judgment is choosing when not to replace it: a model that is 0.01 AP better and ten times harder to monitor is a worse product.

_Try this._ Propose a set of monotone constraints for this problem, argue which are physically justified from kinematic wave theory and which are merely plausible, and describe how you would test that a constraint does not cost skill.

**Guided walkthrough.**

1. Prepare LightGBM datasets with categoricals declared.

   ```python
   import lightgbm as lgb
   feats = [c for c in train_table.columns if c not in ("timestamp", "label", "label_abs", "y_15", "y_30", "y_60", "speed_t30")]
   cats = ["station_id", "fwy", "dir"]
   dtr = lgb.Dataset(tr[feats], tr.y_30, categorical_feature=cats, free_raw_data=False)
   dva = lgb.Dataset(va[feats], va.y_30, categorical_feature=cats, reference=dtr)
   ```

2. Train with early stopping on average precision.

   ```python
   params = dict(objective="binary", is_unbalance=True, num_leaves=63, min_data_in_leaf=200, learning_rate=0.05,
                 feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, max_bin=255,
                 num_threads=8, metric="average_precision", verbose=-1)
   m = lgb.train(params, dtr, num_boost_round=3000, valid_sets=[dva], callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)])
   print("best iteration", m.best_iteration)
   ```

3. Score validation and compare with the baselines from lesson 6.

   ```python
   va["p_30"] = m.predict(va[feats], num_iteration=m.best_iteration)
   print("LightGBM AP", ap(va.y_30, va.p_30), "persistence AP", ap(va.y_30, va.label))
   ```

4. Importance by gain and a SHAP summary for one horizon.

   ```python
   imp = pd.Series(m.feature_importance("gain"), index=feats).sort_values(ascending=False)
   print(imp.head(15))
   import shap
   ex = shap.TreeExplainer(m); sv = ex.shap_values(va[feats].sample(20_000, random_state=0))
   shap.summary_plot(sv, va[feats].sample(20_000, random_state=0), max_display=15)
   ```

5. Save with the label definition in the name, plus the feature list.

   ```python
   m.save_model("models/gbt_h30_thresh0.6_win3.txt", num_iteration=m.best_iteration)
   import json; json.dump(feats, open("models/gbt_h30_features.json", "w"))
   ```

**Check yourself.**

<details>
<summary>What does `min_data_in_leaf = 200` protect against?</summary>

A leaf that isolates a handful of rows fits their noise; 200 forces every leaf to summarize a real population, which is regularization by construction.

</details>

<details>
<summary>Why pass `station_id` as a categorical rather than as an integer?</summary>

As an integer the tree would split on 'id < 4123', which is meaningless. As a categorical it can group stations by behavior.

</details>

<details>
<summary>Why is early stopping done on the validation split and not on training loss?</summary>

Training loss keeps falling as trees are added; validation AP peaks and then declines as the ensemble fits noise. The peak is the model you want.

</details>

**Misconceptions.**

- 'More trees always help.' Past the validation peak they hurt, which is what early stopping finds.
- 'Feature importance shows causation.' It shows what the trees used; a proxy for the cause ranks as high as the cause.
- 'Tree scores are probabilities.' They are ranked well and calibrated badly; isotonic regression fixes that in lesson 11.

### Read and watch, explained at every level

#### Machine Learning Specialization (Andrew Ng), courses 1 and 2

_the classic; audit for free_

**Age 5.** Video classes by a famous teacher who explains how computers learn, slowly and kindly, with drawings on a screen.

**Middle school.** A set of online courses on Coursera that you can watch for free. Course 1 covers linear and logistic regression and gradient descent, the same ideas as lesson 7. Course 2 covers neural networks and decision trees, plus practical advice on what to do when a model does badly. Do the labs; they are small Python notebooks where you fill in the missing pieces.

**High school.** Course 1 is lesson 7's math in video form with numpy labs, and its explanation of the cost function's shape and of feature scaling is the clearest available. Course 2's sections on evaluation, bias versus variance, and error analysis are the practical heart of phase 4, and its decision-tree week is a preview of lesson 8. Watch at normal speed and pause to write the equations yourself.

**Stanford undergrad.** Ng's strength is the practitioner's loop: diagnose whether the problem is bias or variance, then act (more data, more features, more regularization, a different model), and analyze the errors by hand. The treatment is light on theory and on modern tooling, so pair it with ISLP for depth and with the LightGBM documentation for the tools you will actually use.

**Stanford grad.** The specialization's error-analysis method, look at the examples the model gets wrong, categorize them, and prioritize the largest category, is underused in research and is exactly how to improve the traffic model's worst stations: pull the episodes it missed, group them by cause (incident, rain, detector fault), and fix the largest group first.

**PhD at Apple.** Ng's courses shaped how a generation frames ML projects around iterative diagnosis, and his later data-centric argument, that data quality dominates model choice on most real problems, is the thesis of this entire pipeline. For a lead, the course is also the common vocabulary a mixed team can be assumed to share.

#### An Introduction to Statistical Learning with Python (ISLP), chapters 2–4 and 8

_free PDF; the clearest account of trees and boosting_

**Age 5.** A famous textbook, free, that explains learning machines with pictures and real examples, written by teachers who invented some of the methods.

**Middle school.** Written by statisticians, with Python labs. Chapter 2 explains what learning from data is and why a model can be too simple or too complicated. Chapter 3 is linear regression, chapter 4 is classification including logistic regression, and chapter 8 is trees, forests and boosting. Read a chapter, then do its lab in a notebook.

**High school.** The clearest explanation anywhere of bias versus variance (chapter 2, with figures worth redrawing), of logistic regression as a model of probability (chapter 4), and of why boosting works by fitting small trees to what is left over (chapter 8). Its lab for chapter 8 uses the same scikit-learn calls you will make in stage 9.

**Stanford undergrad.** ISLP is the practical face of The Elements of Statistical Learning. Add chapter 5 on cross-validation and the bootstrap, and chapter 6 on model selection and regularization; together with the core chapters they cover phase 4's theory. Its treatment of the bootstrap is the basis for lesson 6's block bootstrap, with the modification that blocks are days.

**Stanford grad.** After ISLP, go to ESL for proofs and depth. ISLP's boosting section gives the additive-model view; combine it with Friedman's 2001 paper for the gradient view that LightGBM implements. Note where ISLP's examples assume independent rows and how each method's assumptions change under temporal dependence.

**PhD at Apple.** As a citation for standard methods and as a teaching text it is the reference, and its evolution from R to Python editions tracks the field's tooling. For research, its framing of methods as estimators with bias and variance you can reason about is the habit that distinguishes a careful ML engineer from a library user.

#### Hands-On Machine Learning (Géron), chapters 1–7

_paid; the practitioner's companion, worth owning_

**Age 5.** A book that shows how to build learning machines with real code, step by step, like a workshop manual with all the tools laid out.

**Middle school.** Aurélien Géron's book (paid, worth owning) walks through a whole project in chapter 2: get data, split it, build features, train a model, evaluate it, tune it. Then classification, training models with gradient descent, support vector machines, decision trees, and ensembles. Chapters 1 to 7 line up with phase 4. Run the notebooks as you read.

**High school.** Chapter 2's project is the template for this pipeline in miniature. Chapter 3 covers precision, recall, the precision-recall curve and ROC with code; chapter 4 trains logistic regression by gradient descent, matching lesson 7; chapters 6 and 7 cover trees, random forests and boosting with scikit-learn. The code is idiomatic scikit-learn and worth imitating.

**Stanford undergrad.** Géron is strong on practice (pipelines, transformers, cross-validation pitfalls, hyperparameter search) and adequate on theory; read alongside ISLP. Be alert to where its examples shuffle data or use `train_test_split` with a random split; for a time series that is the wrong move, as lesson 6 explains, and the fix is a chronological split with a gap.

**Stanford grad.** The later chapters on deep learning overlap with phase 6; choose one framework (PyTorch here) and translate rather than switching. His chapter on sequence models is a useful contrast to Dive into Deep Learning's treatment, and his emphasis on end-to-end pipelines with `Pipeline` and `ColumnTransformer` is a design pattern for the feature module.

**PhD at Apple.** The book's endurance shows the demand for practitioner texts that are correct. Its main lesson for a senior engineer is the scikit-learn API design (fit, transform, predict; composable pipelines), which is worth imitating in an internal feature library so that training and serving call the same objects.

#### Kaggle Learn: Intro to ML, Intermediate ML, Feature Engineering

_free_

**Age 5.** Three short online lessons: build your first learning machine, make it better, and invent better clues for it.

**Middle school.** Intro to ML builds a decision tree and then a random forest on a table of house prices. Intermediate ML covers missing values, categorical columns, pipelines, cross-validation and XGBoost, a cousin of LightGBM. Feature Engineering covers finding useful columns, creating new ones, and encoding categories. Quick, practical, and graded as you go.

**High school.** They give you working code for the exact tools of phase 4 (imputation, categorical handling, gradient boosting with early stopping) in an afternoon. Their cross-validation is shuffled, which is fine for house prices and wrong for traffic; keep lesson 6 in mind and do not copy that part.

**Stanford undergrad.** The Feature Engineering course's target-encoding lesson is a leakage hazard for time series: encode out-of-fold in time or skip it. Its mutual-information lesson is a useful quick screen for which features carry signal. Intermediate ML's pipeline lesson is the pattern to adopt for the feature module.

**Stanford grad.** The courses reflect Kaggle's tabular culture: gradient-boosted trees, engineered features, and careful validation. The competition write-ups they link to are the richer resource; reading three winning solutions for a time-series competition teaches more about leakage and validation than any course.

**PhD at Apple.** Kaggle's tabular consensus (boosted ensembles, careful validation, feature engineering over architecture) has been stable for a decade, and understanding why, which Grinsztajn and colleagues studied systematically, is more valuable than any single course and directly justifies LightGBM as this project's production model.

#### LightGBM documentation: Parameters, Parameters Tuning

_free; read before touching a parameter_

**Age 5.** The instruction sheet for the tree-building program, with all of its knobs and what each one does.

**Middle school.** Two pages of the LightGBM documentation. The first lists every parameter with a one-line meaning; the second is a guide to which knobs to turn for speed, for accuracy, or to stop the model memorizing. Read the tuning page before you change anything, and change one thing at a time.

**High school.** Know these entries: `objective`, `num_leaves`, `min_data_in_leaf`, `learning_rate`, `feature_fraction`, `bagging_fraction` with `bagging_freq`, `lambda_l2`, `max_bin`, `is_unbalance`, `categorical_feature`, and early stopping. The tuning page explains leaf-wise growth and why `num_leaves` and `max_depth` interact, which is the single most common source of overfitting in LightGBM.

**Stanford undergrad.** Read the Features page too: the histogram algorithm, gradient-based one-side sampling, exclusive feature bundling, and native categorical handling explain where the speed and the memory footprint come from. The Python API page's `Dataset` semantics (`free_raw_data`, `reference` for consistent binning between train and validation) prevent subtle bugs.

**Stanford grad.** The documentation is terse on interactions among regularizers; the LightGBM paper (Ke et al., 2017) and the XGBoost paper (Chen and Guestrin, 2016) supply the algorithms and the second-order split gain. Monotone constraints and linear trees are documented options directly relevant here, and the parameter aliases are a known source of silent misconfiguration.

**PhD at Apple.** Documentation quality is a production concern. Parameter aliases have caused silent misconfigurations in real systems; the fix is a team configuration schema that validates parameters before training. At scale, also read the distributed-training and GPU pages to understand what changes when the data no longer fits one machine.

#### Interpretable Machine Learning (Molnar), the SHAP chapter

_free online_

**Age 5.** A free book about asking a learning machine why it said what it said, and getting an answer you can understand.

**Middle school.** Christoph Molnar's online book explains ways to understand models. The SHAP chapter shows how to split one prediction into a contribution from each clue, so you can see that 'downstream speed dropped' pushed the model toward slow and 'Sunday' pushed it away. Read the chapter, then make the plots on the project's own model.

**High school.** SHAP values come from game theory: each feature's fair share of the difference between this prediction and the average prediction. For tree models there is an exact and fast algorithm, TreeSHAP. The chapter explains the plots you will make in stage 9: the summary plot (which features matter overall), dependence plots (how a feature's effect changes with its value), and force plots (one prediction explained).

**Stanford undergrad.** Understand what SHAP does and does not say: it attributes the model's output, not the world's causes; correlated features share credit in ways that depend on the method; and global importance from mean absolute SHAP differs from split-gain importance. The chapter's section on caveats is the important part, and the neighboring chapters on partial dependence and permutation importance give alternatives.

**Stanford grad.** Read alongside the original papers (Lundberg and Lee, 2017; the TreeSHAP paper, 2020) and the critiques concerning feature dependence and off-manifold evaluation. Use SHAP for debugging features and for per-station failure analysis, where it excels, and resist presenting it as a causal story about traffic.

**PhD at Apple.** Interpretability in products is a communication and accountability tool as much as a technical one. SHAP's computational cost at scale and its misuse as causal explanation are both live issues, and Molnar's book is the best neutral survey to hand to a product partner or a regulator before a discussion about what the model 'thinks'.

#### scikit-learn user guide: Model evaluation, Probability calibration

_free; the metrics you will report_

**Age 5.** The manual pages that explain how to score a learning machine, and how to make its confidence honest.

**Middle school.** Two pages of the scikit-learn documentation. Model evaluation lists every scoring function (precision, recall, average precision, log loss, Brier score) with its formula and an example. Probability calibration explains reliability curves and the two fixes, Platt scaling and isotonic regression. Read them with the functions open in a notebook.

**High school.** For lessons 6 and 11 the functions are `average_precision_score`, `precision_recall_curve`, `brier_score_loss`, `log_loss`, `calibration_curve` and `IsotonicRegression`. The guide's examples show the pitfalls directly: ROC curves that look fine under imbalance, and calibration measured on training data that looks perfect and means nothing.

**Stanford undergrad.** The evaluation page is a compact textbook on metrics with definitions and references; the calibration page explains the calibration-refinement decomposition and why calibration must be fit on held-out data. Note that `CalibratedClassifierCV` uses shuffled cross-validation by default, which is wrong for time series; fit isotonic regression manually on the chronological validation split instead.

**Stanford grad.** scikit-learn's implementations are the de facto standard, so knowing their exact conventions (how ties are handled in average precision, the interpolation choices in the precision-recall curve, the binning strategies in `calibration_curve`) matters when comparing your numbers with a paper's.

**PhD at Apple.** The guide is also a model of documentation: definitions, references, examples, and caveats in one place. Your team's metric library should meet that bar, with the label definition and the split protocol versioned alongside the metric code so a number can always be traced to how it was computed.

### Build

- `11_build_labels.py` with the train-only profile and both labels; `12_join_incidents.py` with distance and upstream flags.
- `14_train_baselines.py`: persistence, profile rate, ratio rule.
- `15`: your numpy logistic regression and `sklearn.linear_model.LogisticRegression` on the same standardized features; they must agree.
- `16`: LightGBM per horizon with early stopping on validation average precision; `16b`: isotonic calibration and thresholds.
- The row-level half of `18_evaluate_lead_time.py` with reliability bins, and a SHAP summary for one horizon.
- A leakage experiment: shuffled split versus chronological split on the same features; write down the gap and why it exists.

### Done when

- One table: persistence, profile rate, logistic, LightGBM, three horizons, chronological validation.
- A reliability diagram near the diagonal after calibration.
- You explained every number in that table to someone else, and they could repeat it.

### Pitfalls

- Judging by accuracy.
- Tuning on the test split, even once.
- Celebrating an AUC that persistence also gets.

## Phase 5: Time series and forecasting

_weeks 19–25 · ≈45 h · builds stage 8 targets, stage 11 episodes_

**Goal.** Make the model forecast, and evaluate it as a forecast.

### What you need to learn

- Autocorrelation and the ACF; stationarity, trend, seasonality (daily and weekly here).
- Lag and rolling features done right; the forecast horizon; direct versus recursive multi-horizon strategies.
- Persistence and seasonal-naive baselines; rolling-origin backtests; leakage through time.
- ARIMA and exponential smoothing at reading level: what they are, why trees on lag features replaced them for this problem.
- Spatial correlation and the road graph; propagation of a jam upstream.
- Event-level metrics: episode onset, lead time, false alarms per station-day; anomaly versus level, the two labels.

### Lesson 10: Jams move: sequences, neighbors and the road graph

_Lives in stages 5, 8 and 9 · the registry, neighbor features, 17b._

**Goal.** See why a slowdown downstream predicts your station's future, and how models that read the road as a sequence and a graph exploit that.

**Age 5.**

When the front car in a line stops, the stopping spreads backwards, car by car, like a wave running down a slinky.

So if the road just ahead of you got slow, you will be slow soon. Our helper watches the road ahead to see the wave coming.

_Try this._ Line up toy cars and stop the first one. Watch how the 'stop' moves backwards along the line. Which car stops last?

**Middle school.**

Jams start at a bottleneck, a crash or a busy merge, and grow backwards against the direction of travel at roughly 10 to 15 mph. A station two miles upstream of a jam will be inside it in about ten minutes.

That is why the neighbor clues, especially the downstream station's speed, predict so well at 15 and 30 minutes, and why a model that looks at all the stations together, a 'graph' model, can learn the wave itself instead of being told about it.

_Try this._ A jam's tail moves backwards at 12 mph. How long until it reaches a station 3 miles upstream? What about 6 miles? Which horizon does each match?

**High school.**

Time series vocabulary: autocorrelation (speed now versus five minutes ago correlates at about 0.9), seasonality (daily and weekly), lags as features; persistence is strong precisely because autocorrelation is strong.

Space: the road is a line graph, and the registry's postmiles give each station an upstream and a downstream neighbor. A sequence model (GRU) reads the last 12 steps in order and keeps a memory; graph mixing lets each station's memory hear its neighbors' memories.

The physics: between two traffic states the boundary moves at w = Δflow / Δdensity. When a road goes from flowing to jammed, that slope is negative, a backward-moving wave, and its size sets how far ahead a station's future can be read from stations downstream.

_Try this._ Plot the autocorrelation of one station's speed for lags 1 to 48 (four hours). Describe the shape and say what it implies for the lag features.

**Stanford undergrad.**

Kinematic wave theory (Lighthill-Whitham-Richards): conservation ∂k/∂t + ∂q/∂x = 0 with a fundamental diagram q = Q(k). A discontinuity between states 1 and 2 moves at w = (q₂ − q₁)/(k₂ − k₁); in the congested branch w is about −10 to −15 mph, and queues discharge at capacity when the bottleneck clears.

Implication for prediction: the horizon over which a station's future is readable from its neighbors is roughly the distance to the nearest downstream disturbance divided by |w|. Beyond that, the calendar and incidents carry the signal.

Models along the ladder: ARIMA on one series, vector autoregression across stations, a GRU as a nonlinear state-space model, graph convolution as spatial smoothing with learned weights, and direct multi-horizon heads.

_Try this._ From one station's scatter of flow against occupancy, estimate capacity and the slope of the congested branch, and convert the slope into a shockwave speed in mph.

**Stanford grad.**

The spatio-temporal literature on PeMS data: DCRNN (diffusion convolution with forward and backward random-walk matrices on a directed graph, encoder-decoder GRU, scheduled sampling), STGCN (gated temporal convolutions with Chebyshev graph convolutions), Graph WaveNet (adaptive adjacency with dilated causal convolutions), then transformer variants. Benchmarks report MAE, RMSE and MAPE at 15, 30 and 60 minutes on METR-LA and PEMS-BAY.

The stable findings: gains over persistence grow with horizon; learned or adaptive graphs help; simple models are competitive at 15 minutes. The plan's two-hop mixing corresponds to a K = 2 diffusion step, and respecting wave direction (separate upstream and downstream weights) is a physically motivated inductive bias.

_Try this._ Implement diffusion convolution with the registry's directed adjacency and compare its validation average precision at 30 and 60 minutes with LightGBM's hand-made neighbor features.

**PhD at Apple.**

City-scale graphs have hundreds of thousands to millions of segments with uneven probe coverage; practical systems use supersegments, embedding tables and hierarchy (the Google Maps ETA graph network is the public reference point), and update hidden states incrementally in streaming rather than recomputing windows.

Research threads: coupling learned models with simulators (cell transmission, microsimulation) for counterfactual questions such as rerouting a share of drivers; causal effects of incidents on propagation; transfer across cities; and wave physics as a regularizer in the loss.

_Try this._ Sketch a hybrid where the GRU predicts a residual over a kinematic-wave forecast, and argue where along the road and at which horizons the residual should carry the most signal.

**Guided walkthrough.**

1. Autocorrelation of one station's speed.

   ```python
   s = df[df.station_id == sid].set_index("timestamp").speed.interpolate(limit=3)
   acf = [s.autocorr(lag) for lag in range(1, 49)]
   import matplotlib.pyplot as plt; plt.plot(range(1, 49), acf); plt.xlabel("lag (5-min steps)"); plt.ylabel("autocorrelation"); plt.show()
   ```

2. The fundamental diagram from one month of one station.

   ```python
   x = df[df.station_id == sid]
   plt.scatter(x.occ, x.flow, s=2, alpha=.3); plt.xlabel("occupancy"); plt.ylabel("flow per 5 min"); plt.show()
   ```

3. Directed adjacency matrices from the registry.

   ```python
   ids = list(reg.ID); pos = {i: k for k, i in enumerate(ids)}; N = len(ids)
   A_up = np.zeros((N, N), np.float32); A_down = np.zeros((N, N), np.float32)
   for _, r in reg.iterrows():
       if pd.notna(r.upstream_id):   A_up[pos[r.ID], pos[r.upstream_id]] = 1
       if pd.notna(r.downstream_id): A_down[pos[r.ID], pos[r.downstream_id]] = 1
   ```

4. Windows of 12 steps × N stations × 6 channels, one per timestamp.

   ```python
   chan = ["speed_ratio", "occ", "flow_scaled", "incident_active", "min_sin", "min_cos"]
   wide = df.pivot(index="timestamp", columns="station_id", values=chan)        # (T, 6*N)
   X = np.stack([wide.to_numpy(np.float32).reshape(len(wide), len(chan), N)[t - 12:t] for t in range(12, len(wide))])  # (T-12, 12, 6, N)
   ```

5. A minimal spatio-temporal model.

   ```python
   class STNet(nn.Module):
       def __init__(self, C=6, H=64):
           super().__init__(); self.gru = nn.GRU(C, H, batch_first=True); self.mix1 = Mix(H); self.mix2 = Mix(H); self.head = nn.Linear(H, 3)
       def forward(self, x, A_up, A_down):                     # x: (B, 12, C, N)
           B, T, C, N = x.shape
           seq = x.permute(0, 3, 1, 2).reshape(B * N, T, C)     # every station is its own sequence
           h = self.gru(seq)[1][0].reshape(B, N, -1)
           h = self.mix2(self.mix1(h, A_up, A_down), A_up, A_down)
           return self.head(h)                                  # (B, N, 3) logits for 15/30/60
   ```

**Check yourself.**

<details>
<summary>Why is the downstream neighbor more informative than the upstream one for the next 30 minutes?</summary>

Queues grow backwards. A slow downstream station means the wave is heading toward you; a slow upstream station means it has already passed.

</details>

<details>
<summary>What limits how far ahead neighbors can help?</summary>

Distance to the disturbance divided by the wave speed. Beyond that horizon nothing on the road yet carries the information.

</details>

<details>
<summary>Why share the GRU across stations?</summary>

One set of weights sees a hundred times more data, and the dynamics of a queue are the same physics everywhere; station identity can be added as an embedding if needed.

</details>

**Misconceptions.**

- 'Traffic is random.' Propagation is mechanical and fast to learn; what is hard is the onset of the first disturbance.
- 'The graph model replaces the physics.' It learns the physics from data; encoding wave direction helps it learn faster.
- 'Longer windows are better.' Beyond an hour the extra history mostly adds noise; the calendar carries the slow signal.

### Lesson 11: From scores to alerts: calibration, thresholds, lead time

_Lives in stages 10 and 11 · 16b_calibrate_select.py, 18_evaluate_lead_time.py._

**Goal.** Make the probabilities honest, choose when to warn, and measure the warning the way a driver would: how early, and how often wrongly.

**Age 5.**

Our helper says '70 % chance of slow'. If it says that a hundred times, it should be right about seventy of them. If it is only right forty times, its sureness is broken and we fix it.

Then we decide how sure it has to be before we shout 'slow!'. Shout too easily and people stop listening; shout too rarely and they get stuck.

_Try this._ Guess ten coin flips and say how sure you are each time. Keep score. Was your 'very sure' really more right than your 'not sure'?

**Middle school.**

Calibration means the probabilities are honest. We check by grouping predictions, all the 0.6-to-0.7s together, and counting how many really were slow. If the model is over-confident, we fit a correction curve, called isotonic regression, using data it did not train on.

Then the threshold. Warn at 0.3 and you catch more jams but cry wolf more; warn at 0.7 and the opposite. We pick the threshold that keeps, say, half of the warnings right, and we accept the recall that comes with it.

Lead time is the number of minutes between our first warning and the jam actually starting. That is the number a driver cares about.

_Try this._ Write ten predictions and what happened. Split them into two bins, below 0.5 and above. In each bin, compare the average prediction with the share that were slow.

**High school.**

Reliability diagram: bin rows by predicted probability, plot the observed share of positives per bin; the gap is calibration error. Two fixes fit on the validation split: Platt scaling (a sigmoid on the score) and isotonic regression (a monotone step function). Isotonic is more flexible and needs more data.

The threshold comes from the validation precision-recall curve at the precision the alert has to keep. Episode metrics: an episode begins at the first step of a run of label = 1; the alert is the first t in the two hours before that with p ≥ τ; lead = onset − alert; a false alarm is an alert not followed by an episode within 60 minutes, counted per station-day.

Trade-off: lower τ, longer lead and more false alarms. Plot lead time against false alarms per station-day for several τ and choose with the user in mind.

_Try this._ Make up a sequence of probabilities and labels over two hours. Compute the lead time and count false alarms at τ = 0.5 and τ = 0.3.

**Stanford undergrad.**

Proper scoring rules decompose into calibration and refinement; a model can rank perfectly and be badly calibrated, which is the tree-ensemble case. Isotonic regression is the pool-adjacent-violators algorithm and can overfit on a small validation set, so it needs thousands of positives; Platt scaling is safer with fewer.

Class reweighting shifts the prior, so calibrate after reweighting, on data with the true prevalence. Decision theory gives the threshold for calibrated probabilities: warn when p > C_FA / (C_FA + C_miss). Alerts are a policy, not a classifier: hysteresis (raise at τ_on, clear at a lower τ_off) prevents flapping, and the policy, not the model, is what gets evaluated on episodes.

_Try this._ Prove the cost-optimal threshold formula from expected cost and compute it for a false alarm costing 1 and a miss costing 4. Then explain why the formula is useless if the probabilities are not calibrated.

**Stanford grad.**

Three horizons produce three probabilities that can disagree; combine them into an expected time-to-slow or a hazard curve, or enforce monotone structure. Conformal methods give alert sets with distribution-free control of error rates over episodes, at the cost of exchangeability assumptions that time series violate and must be handled with weighting.

Calibration drifts and per-station calibration is noisy; hierarchical shrinkage toward the pooled map is the practical middle. Human factors from warning science: false-alarm ratios above roughly a half erode compliance, which sets a floor on the precision the policy must keep.

_Try this._ Formulate alerting as a sequential decision with a constraint on false alarms per station-day, and propose a policy with hysteresis that respects the constraint. Say how you would evaluate it on the test split.

**PhD at Apple.**

In products, calibration and thresholds live in a policy layer separate from the model, versioned, and tested with real users; alert budgets are enforced globally and evaluated on outcomes rather than on precision.

Research threads: risk-controlling prediction sets (Angelopoulos, Bates and colleagues), calibration under covariate and label shift with unlabeled target data, and decision-focused learning where the model is trained for the policy's objective rather than for log loss.

_Try this._ Design an experiment that estimates the causal effect of an alert on a driver's reroute decision and travel time, given that alerts change the traffic other drivers experience.

**Guided walkthrough.**

1. Isotonic calibration on validation, per horizon.

   ```python
   from sklearn.isotonic import IsotonicRegression
   iso = IsotonicRegression(out_of_bounds="clip").fit(va.p_30, va.y_30)
   va["q_30"] = iso.predict(va.p_30)
   print("Brier raw", np.mean((va.p_30 - va.y_30) ** 2), "Brier calibrated", np.mean((va.q_30 - va.y_30) ** 2))
   ```

2. Threshold at the required precision.

   ```python
   p, r, thr = precision_recall_curve(va.y_30, va.q_30)
   ok = np.where(p[:-1] >= 0.5)[0]; tau = float(thr[ok[0]]); print("tau_30", tau, "recall", float(r[ok[0]]))
   ```

3. Episode onsets on the test split.

   ```python
   te = te.sort_values(["station_id", "timestamp"])
   te["onset"] = ((te.label == 1) & (te.groupby("station_id").label.shift(1).fillna(0) == 0)).astype(int)
   ```

4. Lead time: first alert in the two hours before each onset.

   ```python
   te["alert"] = (iso.predict(te.p_30) >= tau).astype(int)
   leads = []
   for sid, g in te.groupby("station_id"):
       g = g.reset_index(drop=True); on = g.index[g.onset == 1]
       for i in on:
           w = g.loc[max(0, i - 24):i - 1]
           hit = w.index[w.alert == 1]
           leads.append((i - hit[0]) * 5 if len(hit) else np.nan)
   leads = pd.Series(leads); print("episodes", len(leads), "caught", leads.notna().mean(), "median lead min", leads.median())
   ```

5. False alarms per station-day.

   ```python
   te["fut"] = te.groupby("station_id").label.transform(lambda s: s[::-1].rolling(12, min_periods=1).max()[::-1])
   fa = te[(te.alert == 1) & (te.fut == 0)].groupby([te.station_id, te.timestamp.dt.floor("D")]).size()
   print("false alarms per station-day", fa.sum() / (te.station_id.nunique() * te.timestamp.dt.floor("D").nunique()))
   ```

**Check yourself.**

<details>
<summary>Why calibrate on the validation split and not on training?</summary>

The model has fit the training rows; its confidence there is meaningless. Calibration needs rows the model has not seen, and the test split must stay untouched for the final report.

</details>

<details>
<summary>What does a lead time of zero or negative mean?</summary>

The first alert came at or after the episode began: the model saw the jam, it did not foresee it. Expect that for crash-initiated episodes.

</details>

<details>
<summary>Why count false alarms per station-day and not per row?</summary>

A driver experiences alerts per trip, not per 5-minute row; a station firing twenty times in one hour is one bad hour, not twenty errors.

</details>

**Misconceptions.**

- '0.5 is the natural threshold.' It is only right when a false alarm and a miss cost the same, which they never do.
- 'Calibration fixes ranking.' It is monotone; it changes the numbers, not the order.
- 'A high AP means useful alerts.' Only the episode metrics, lead time and false alarms, say that.

### Read and watch, explained at every level

#### Forecasting: Principles and Practice, the Python edition (Hyndman, Athanasopoulos), chapters 1–5, 7, 9

_free online; the forecasting book_

**Age 5.** A free book about predicting the future from the past, by people who do it for a living, with lots of pictures of wiggly lines.

**Middle school.** The standard forecasting textbook, free online, now with Python. Chapters 1 to 5 cover what forecasting is, how to draw time series, how to split a series into trend and seasonal parts, simple forecasting methods, and how to check whether a forecast is any good. Chapter 7 is regression with time features, chapter 9 is ARIMA. Read 1 to 5 fully and do the exercises.

**High school.** Chapter 5's evaluation (training and test in time order, residual checks, accuracy measures) and its simple baselines (naive, which is persistence; seasonal naive, which is our profile rate) are the heart. Chapter 3's decomposition helps you see the daily and weekly cycles in a station's speed, and chapter 7's Fourier terms are the sine and cosine time features of lesson 5.

**Stanford undergrad.** The book supplies the statistical forecasting vocabulary that ML-first people lack: stationarity, autocorrelation, exponential smoothing, ARIMA, and its 'time series cross-validation', which is the rolling-origin protocol lesson 6 uses. Chapter 9 shows why ARIMA is a linear model on lags with a principled error structure, which clarifies what a gradient-boosted model on lag features adds and gives up.

**Stanford grad.** Compare its classical models with the ML approach: one ARIMA per series does not scale to a hundred stations with spatial dependence, and its hierarchical and global-model chapters bridge toward the global boosted and graph models used here. Its treatment of forecast intervals is the classical route to the uncertainty questions of phase 8.

**PhD at Apple.** Hyndman's wider work (the M4 and M5 forecasting competitions, hierarchical reconciliation, feature-based model selection) is the empirical forecasting literature. The M5 result, gradient-boosted models on lag features winning at scale, is direct evidence for this project's choice of LightGBM as the production model.

#### Kaggle Learn: Time Series

_free; lag features and hybrids in an afternoon_

**Age 5.** A short online lesson about guessing what comes next in a line of numbers.

**Middle school.** Six lessons on store sales data: the trend, seasonality using sine and cosine features, using past values as features (lags), combining two models, and forecasting several steps ahead with machine learning. About half a day, graded as you go.

**High school.** Its lag-feature and Fourier-feature lessons are exactly our lags and the sine and cosine encoding of time. Its hybrid lesson, a simple model for the predictable part plus a boosted model for what is left, is a pattern to try here: profile plus a residual model.

**Stanford undergrad.** Note how the course handles multi-step forecasting (direct versus recursive strategies) and where leakage hides in lag construction. Its examples are single series without spatial structure, so the neighbor features of lesson 5 and the graph models of lesson 10 are beyond it.

**Stanford grad.** The hybrid idea generalizes to 'physics plus residual', which lesson 10's advanced prompt asks about, and its Fourier treatment of seasonality is the linear analogue of what a recurrent network learns. Short courses like this are where practitioner defaults come from, so know what they teach and where they stop.

**PhD at Apple.** The design principle it plants, decompose into predictable structure and a learned residual, is robust across production forecasters and is how a classical model can remain as a fallback when a learned model goes stale.

#### statsmodels documentation: one ARIMA fit on one station

_free; do it once to know what it is_

**Age 5.** A tool that fits an old but famous kind of forecast to a line of numbers. You try it once to see what it does.

**Middle school.** statsmodels is a Python library for statistics. Fit an ARIMA model to one station's speed, look at its forecast for the next hour and at what it got wrong. The point is to know what ARIMA is and how it behaves, not to put it in the pipeline.

**High school.** ARIMA(p, d, q) predicts a value from p past values, d differences, and q past errors. Fit it with `ARIMA(series, order=(2, 0, 1)).fit()`, forecast twelve steps, and compare with persistence. You will see a linear model on lags that does about as well as persistence at short horizons and drifts toward the mean at long ones.

**Stanford undergrad.** Understanding ARIMA clarifies why lag features work and what a recurrent network adds: nonlinearity, exogenous inputs, and parameters shared across stations. SARIMAX adds seasonality and exogenous regressors, which is the classical way to include incidents and rain, and its diagnostics (residual autocorrelation) are useful for any model.

**Stanford grad.** Vector ARIMA and state-space models with the Kalman filter are the classical route to multi-station forecasting, and their scaling limits are what motivated the graph deep-learning literature. statsmodels' state-space module is a good place to see the Kalman filter concretely before reading about recurrent models as nonlinear state-space models.

**PhD at Apple.** Classical models remain the interpretable baseline and a source of principled uncertainty that deep models lack. A production system often keeps a simple statistical model as the fallback when the learned model is stale or its inputs are missing, and statsmodels is where that fallback would be built.

### Build

- `13_feature_engineering.py` with forward-shifted targets `y_15`, `y_30`, `y_60` and the rolling-origin split with a one-day gap.
- The episode-level half of `18_evaluate_lead_time.py`: onset, lead-time histogram, share of episodes caught, false alarms.
- Plot the ACF of speed for one station; compute skill over persistence per horizon.
- Train on the absolute label too; compare prevalence and skill and write the decision.

### Done when

- A lead-time histogram and one sentence per horizon on skill over persistence.
- A written decision on which label the product means, with numbers.

### Pitfalls

- `shift` in the wrong direction; a rolling window that looks ahead.
- Evaluating on rows whose target was imputed.

## Phase 6: Neural networks

_weeks 24–34 · ≈80 h · builds stage 9 networks_

**Goal.** The MLP in numpy with backprop by hand, then the spatio-temporal net in PyTorch on the M1's GPU.

### What you need to learn

- The perceptron and the multilayer perceptron; activations; loss functions; backpropagation for one hidden layer, by hand.
- Gradient descent variants (SGD, momentum, Adam); initialization; batch normalization, dropout, learning-rate schedules; early stopping.
- PyTorch: tensors, autograd, `nn.Module`, `DataLoader`, the `mps` device, saving and loading, `model.eval()`.
- Sequence models: RNN, GRU, LSTM, 1-D temporal convolutions; attention and transformers at reading level.
- Graph neural networks: adjacency, message passing, diffusion on a graph; how DCRNN and STGCN model traffic on PeMS data.
- The practical rules: small data wants small nets; a deep model does not win by default.

### Lesson 9: Neural networks and backpropagation, on the M1

_Lives in stage 9 · 17_train_mlp.py, 17b, PyTorch on MPS._

**Goal.** Build a network by hand once, check its gradients, then let PyTorch do it on the Mac's GPU and know what changed.

**Age 5.**

A neural network is a lot of tiny seesaws stacked in layers. The first layer looks at the clues, the next layer looks at what the first layer said, and so on until the last one gives the answer.

When the answer is wrong, the 'oops' travels backwards through the layers so every seesaw adjusts a little. That is called backpropagation.

_Try this._ Play the whisper game down a line of friends. When the message comes out wrong, work backwards to find where it changed. That is what the network does with its mistake.

**Middle school.**

One layer computes weighted sums of its inputs, exactly like lesson 7, and then applies a simple bend: negatives become zero (ReLU). Stacking layers lets the network learn combinations of clues, such as 'raining AND rush hour', which a single seesaw cannot.

Training is the same loop as before: guess, compare, adjust. The adjusting uses the chain rule to pass the error back through every layer. PyTorch does that calculus for you.

The Mac mini has a small GPU. PyTorch can run the arithmetic there through a backend called MPS. For tiny networks the CPU is nearly as fast, because telling the GPU what to do costs more than the work.

_Try this._ Inputs a = 1, b = 2, weights 0.5 and −1, bias 0.2. Compute the weighted sum, then apply ReLU. Now flip the sign of the first weight and do it again.

**High school.**

Forward pass: h = ReLU(W₁x + b₁), p = σ(w₂·h + b₂). Loss: cross-entropy. Backward pass: ∂L/∂w₂ = (p − y) h; ∂L/∂h = (p − y) w₂; through ReLU multiply by 1 where the pre-activation was positive and 0 elsewhere; ∂L/∂W₁ = δ₁ xᵀ.

Optimizers: plain gradient descent, momentum, Adam (adaptive step per weight). Overfitting controls: early stopping, dropout, weight decay. Practical rules: standardize inputs, start with a learning rate of 1e-3, batches of a few hundred, and make sure the loss falls on a tiny subset before training on everything.

On the M1: `device = torch.device('mps')`, move the model and each batch with `.to(device)`, float32 only.

_Try this._ Implement the one-hidden-layer forward and backward pass in numpy and check every gradient with finite differences at ε = 1e-5. Do not move on until the largest difference is below 1e-6.

**Stanford undergrad.**

Backpropagation is reverse-mode automatic differentiation on a computational graph: one forward and one backward pass give every parameter's gradient at a cost independent of parameter count. Initialization (He for ReLU, Xavier for tanh) keeps activations and gradients at unit scale; without it deep nets vanish or explode.

Batch and layer normalization stabilize training; Adam's bias correction matters in the first steps; warmup and cosine schedules are cheap wins. Early stopping on validation is the main regularizer for small tabular nets and is equivalent to L2 for linear models.

Engineering: keep the dataset on the device, batch in large chunks, avoid per-batch Python work. MPS specifics: fp32 and fp16, no fp64, `PYTORCH_ENABLE_MPS_FALLBACK=1` for unsupported ops, and a launch-overhead floor that makes tiny models CPU-competitive. Measure rows per second at batch 256 and 8192 on both devices before choosing.

_Try this._ Benchmark the MLP on CPU and MPS at batch sizes 256, 2048 and 8192; find the crossover and explain it in terms of kernel launch overhead versus arithmetic.

**Stanford grad.**

Overparameterized networks generalize for reasons still being mapped: the implicit bias of SGD, double descent, and flat minima are the working vocabulary. On tabular data of this size they rarely beat boosted trees (Grinsztajn et al., 2022) unless the data has structure the trees cannot see, which here is the spatial graph.

Sequence models: GRU and LSTM gating, truncated backpropagation through time, dilated temporal convolutions and attention. For traffic, DCRNN couples diffusion convolution on the road graph with a sequence-to-sequence GRU; STGCN alternates gated temporal convolutions with spectral graph convolutions; Graph WaveNet learns an adaptive adjacency. The plan's challenger, a shared GRU plus two rounds of neighbor mixing, is a stripped-down DCRNN.

Training recipe for the 12 × N × 6 windows: whole dataset in memory, batches of timestamps, positive class weight per horizon, early stopping on validation average precision.

_Try this._ Write the message-passing update h ← h + W_u h_up + W_d h_down as a matrix expression with directed adjacency matrices A_up and A_down, and relate two rounds of it to a K = 2 diffusion convolution.

**PhD at Apple.**

On-device deep learning brings constraints that training clusters hide: Core ML export, fp16 on the Neural Engine, static shapes, a supported-operator list, energy per inference, and thermal throttling. The unified-memory model on Apple silicon removes copies between CPU and GPU but makes many workloads bandwidth-bound.

What transfers from the Mac mini to a cluster: recipes, evaluation discipline, and the habit of checking gradients; what does not: batch-size and learning-rate scaling laws.

Research threads: spatio-temporal transformers and adaptive graphs at city scale; physics-informed losses that respect kinematic wave direction; time-series foundation models used zero-shot against small specialized models; and uncertainty through deep ensembles, dropout at inference or conformal wrappers.

_Try this._ Outline the Core ML export path for the spatio-temporal net: which layers convert cleanly, how to handle a variable number of stations, and what you would measure on device before shipping.

**Guided walkthrough.**

1. The numpy MLP with a hand-written backward pass.

   ```python
   rng = np.random.default_rng(0); d, H = Xtr.shape[1], 64
   W1 = rng.normal(0, np.sqrt(2 / d), (H, d)).astype(np.float32); b1 = np.zeros(H, np.float32)
   w2 = rng.normal(0, np.sqrt(1 / H), H).astype(np.float32); b2 = 0.0
   def forward(X):
       z1 = X @ W1.T + b1; h = np.maximum(z1, 0); z2 = h @ w2 + b2; p = 1 / (1 + np.exp(-z2)); return z1, h, p
   def backward(X, y):
       z1, h, p = forward(X); e = (p - y) / len(y)
       gw2 = h.T @ e; gb2 = e.sum()
       dh = np.outer(e, w2) * (z1 > 0)
       gW1 = dh.T @ X; gb1 = dh.sum(0)
       return gW1, gb1, gw2, gb2
   ```

2. Gradient check on a few rows.

   ```python
   def L(X, y): p = np.clip(forward(X)[2], 1e-7, 1 - 1e-7); return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
   X0, y0 = Xtr[:64], ytr[:64]; gW1, gb1, gw2, gb2 = backward(X0, y0)
   eps = 1e-4; i, j = 3, 5
   W1[i, j] += eps; lp = L(X0, y0); W1[i, j] -= 2 * eps; lm = L(X0, y0); W1[i, j] += eps
   print(abs((lp - lm) / (2 * eps) - gW1[i, j]))     # ~1e-6
   ```

3. The same network in PyTorch on MPS.

   ```python
   import torch, torch.nn as nn
   dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
   net = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 1)).to(dev)
   opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
   pos_w = torch.tensor([(1 - ytr.mean()) / ytr.mean()], device=dev)
   lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
   Xt, yt = torch.tensor(Xtr, device=dev), torch.tensor(ytr, device=dev)
   ```

4. Train with early stopping on validation AP.

   ```python
   best, patience = 0, 0
   for epoch in range(30):
       net.train(); perm = torch.randperm(len(yt), device=dev)
       for i in range(0, len(perm), 2048):
           j = perm[i:i + 2048]; opt.zero_grad(); lossf(net(Xt[j]).squeeze(1), yt[j]).backward(); opt.step()
       net.eval()
       with torch.no_grad(): pv = torch.sigmoid(net(torch.tensor(Xva, device=dev)).squeeze(1)).cpu().numpy()
       score = ap(yva, pv); print(epoch, score)
       if score > best: best, patience = score, 0; torch.save(net.state_dict(), "models/mlp_h30.pt")
       elif (patience := patience + 1) >= 5: break
   ```

5. The spatio-temporal step: one round of neighbor mixing as matrix products.

   ```python
   # hidden: (N, H) after a GRU over 12 steps per station; A_up, A_down: (N, N) 0/1 directed adjacency from the registry
   class Mix(nn.Module):
       def __init__(self, H): super().__init__(); self.Wu = nn.Linear(H, H, bias=False); self.Wd = nn.Linear(H, H, bias=False)
       def forward(self, h, A_up, A_down): return h + self.Wu(A_up @ h) + self.Wd(A_down @ h)
   ```

**Check yourself.**

<details>
<summary>Why check gradients numerically when autograd exists?</summary>

For the numpy version, because a wrong backward pass still trains, just badly. For PyTorch, because the check teaches you what autograd is doing and catches a wrong loss or a detached tensor.

</details>

<details>
<summary>Why `BCEWithLogitsLoss` rather than a sigmoid followed by `BCELoss`?</summary>

It computes the loss from logits with log-sum-exp, which is numerically stable for large scores; sigmoid then log can underflow to −inf.

</details>

<details>
<summary>Why is the GPU barely faster for this network?</summary>

Each batch launches a handful of tiny kernels; the launch overhead is comparable to the arithmetic. Larger batches or larger models tip the balance toward MPS.

</details>

**Misconceptions.**

- 'Deep learning needs a data center.' A 0.1 M-parameter network trains in an hour on a Mac mini; what needs a data center is a different problem.
- 'A network will beat the trees.' On this data, usually not; the graph structure is its one real advantage.
- 'Training loss going down means it works.' Validation AP is the only number that means it works.

### Read and watch, explained at every level

#### 3Blue1Brown: Neural networks

_free; four videos, watch before Karpathy_

**Age 5.** Four videos with moving pictures that show what a neural network is and how it learns, using handwritten numbers as the example.

**Middle school.** Watch them before Karpathy's course. They show a network as layers of numbers, what a weight and a bias do, what gradient descent means, and what backpropagation computes, all with animations of the digit-recognition example. Pause often and predict what the next picture will show.

**High school.** The third and fourth videos derive backpropagation with the chain rule visually. Pause and write the equations yourself as they appear; then lesson 9's numpy implementation will feel like transcribing the pictures. The first video's description of a neuron as a weighted sum plus a squashing function is lesson 7 in one image.

**Stanford undergrad.** The series builds intuition for the computation graph and for how gradients flow backward through it; after it, the formal Jacobian-chain view is easy. Its digit example is a plain multilayer perceptron, the same architecture as the numpy network in stage 9, and its remarks on what the hidden layers actually learn are a good caution against over-interpreting them.

**Stanford grad.** Good for teaching; note what it simplifies (no batching, no optimizers beyond plain gradient descent, no regularization, no normalization). The newer videos on transformers and attention extend the same visual method to the models of phase 8, and are the shortest route to understanding attention geometrically.

**PhD at Apple.** Its visual language has become the standard for explaining networks to non-specialists. Reuse the logic of its diagrams when presenting the spatio-temporal network to a product team, and its treatment of what a layer 'sees' as a caution when someone asks what the model has learned.

#### Neural Networks: Zero to Hero (Karpathy), micrograd and makemore

_free; backprop from scratch, the best there is_

**Age 5.** A programmer builds a tiny brain from scratch on video, typing every line, and explains each one as he goes.

**Middle school.** Andrej Karpathy's free YouTube course. micrograd builds automatic differentiation, the machinery that computes gradients, in about a hundred lines of Python. makemore builds models that generate names, starting from counting letter pairs and ending with a small transformer. Code along in a notebook; pause and predict what each line does before he runs it.

**High school.** micrograd is lesson 9's backward pass done generally: every operation records how to pass its gradient back, and the chain rule falls out. Build it, then use it to train lesson 7's logistic regression. makemore's early lessons teach the training loop, learning rates, initialization and batch normalization by showing failures happening on screen and fixing them.

**Stanford undergrad.** The course's method (make the bug, watch it, fix it) teaches debugging networks better than any textbook. Its lesson on the statistics of activations and gradients is a practical tutorial on initialization and normalization. After micrograd, PyTorch's autograd is obvious rather than magical, and you will know what `requires_grad` and `backward` actually do.

**Stanford grad.** makemore's progression to attention is the clearest path to understanding transformers, and the follow-up on building GPT completes it. For traffic the relevant carry-over is the sequence-modelling mindset and the training diagnostics: reading the loss curve, the gradient norms and the activation histograms before changing an architecture.

**PhD at Apple.** Karpathy's teaching sets the bar for from-scratch pedagogy, and his emphasis on reading gradients and activations is the diagnostic culture a research team should share. For a lead, the course is the fastest way to give a strong engineer without deep-learning background a working mental model in a week.

#### PyTorch tutorials: Learn the Basics; the MPS backend notes

_free_

**Age 5.** The official lessons for the tool that runs neural networks, plus a note about making it use the Mac's graphics chip so it goes faster.

**Middle school.** Learn the Basics is eight short pages: tensors (grids of numbers), datasets, building a model, automatic gradients, the training loop, and saving. The MPS notes explain how to run on the Mac's GPU by setting the device to `mps`. Do the tutorial in a notebook and run each cell.

**High school.** After it you can write lesson 9's PyTorch network: `nn.Module`, `DataLoader`, the training loop, `torch.no_grad()` for evaluation, and saving a `state_dict`. The MPS page lists the limits that matter on the Mac mini: float32 and float16 only, no float64, and an environment variable that lets unsupported operations fall back to the CPU.

**Stanford undergrad.** Understand tensors' memory model (views versus copies, contiguity, in-place operations), the autograd graph and `requires_grad`, and why moving data to the device in large chunks matters more than any architectural choice for small models. Read the profiler tutorial to see where time actually goes on MPS; the answer is usually kernel launches, not arithmetic.

**Stanford grad.** The advanced tutorials (custom autograd functions, `torch.compile`, distributed training) are the next layer. For this project, test whether `torch.compile` helps on MPS rather than assuming; support is uneven. The tutorials on saving and loading, and on reproducibility, are the ones that prevent embarrassment at handover.

**PhD at Apple.** PyTorch's eager, Pythonic design is why research iterates fast; its deployment story (TorchScript's decline, ExecuTorch, ONNX and Core ML export) is where production effort goes. Knowing both ends, and the operator-support gaps between them, is what lets you promise a model can ship on a device.

#### Dive into Deep Learning (d2l.ai), chapters 3–5 and 9–10

_free, PyTorch throughout_

**Age 5.** A free online textbook where every idea comes with code you can run, about teaching computers using layers of simple math.

**Middle school.** Chapters 3 to 5 cover linear and softmax regression and multilayer perceptrons with PyTorch code you can execute. Chapters 9 and 10 cover recurrent networks (RNN, GRU, LSTM) and how to train them well. Run the notebooks; each section ends with exercises that are worth doing.

**High school.** Chapter 5's multilayer perceptron from scratch mirrors lesson 9's numpy network; chapter 9 explains why GRUs can remember and forget, which is the sequence half of the spatio-temporal network. The exercises about changing hidden size and learning rate are the experiments to actually run on the traffic data.

**Stanford undergrad.** d2l's coverage of optimization (chapter 12), attention (chapter 11) and computational performance (chapter 13) is deeper than most books and free. Its consistent code style makes it easy to lift components, and its treatment of truncated backpropagation through time explains what a recurrent model can and cannot learn from a 12-step window.

**Stanford grad.** Its graph-neural-network coverage is thin; use CS224W for that. Its attention chapters are the bridge to spatio-temporal transformers, and its chapter on computational performance (asynchrony, memory, multi-device) is the right background before profiling on unified memory.

**PhD at Apple.** A living, open textbook with code in several frameworks is itself a research-infrastructure achievement. Cite it for standard architectures, use its implementations as reference points when reproducing papers, and note its authors' choice to include the messy practical chapters most textbooks omit.

#### Practical Deep Learning for Coders (fast.ai)

_free; the practitioner's counterweight to the theory_

**Age 5.** A free class where you train a real neural network on the very first day and only later learn all the details, like learning to ride a bike before studying how it works.

**Middle school.** Jeremy Howard's course is top-down: you build working models first (on images, text and tables) and learn the theory as you need it. It uses the fastai library built on PyTorch. It is good for confidence and full of practical tricks that make training work.

**High school.** Its tabular lessons (random forests first, then neural networks on tables) and its training tricks (a learning-rate finder, the one-cycle schedule) are directly usable in stage 9. The lesson on embeddings for categorical variables shows how to give station identity to a network, which is how the spatio-temporal net could learn per-station behavior.

**Stanford undergrad.** The course's emphasis on iterating quickly and inspecting data is a counterweight to theory-first study, and its stated view that tree ensembles usually win on tabular data matches the evidence and this project's production choice. Its chapter on data ethics is short and worth the time for anyone whose model will issue alerts to drivers.

**Stanford grad.** The fastai library's abstractions can hide details you need for a custom model such as the spatio-temporal network. Learn the ideas from the course, then implement in plain PyTorch. The course's lessons on collaborative filtering and on tabular embeddings are the clearest introduction to learned representations for categorical data.

**PhD at Apple.** fast.ai's research output (training-efficiency results, transfer learning for text) shows that practitioner-driven groups produce research results. Its philosophy of accessible deep learning shaped how many teams onboard engineers, and its lesson structure is a reasonable template for an internal course on this pipeline.

#### DCRNN (Li et al., ICLR 2018) and STGCN (Yu et al., IJCAI 2018)

_the two traffic papers; first pass now, second in phase 8_

**Age 5.** Two papers where scientists taught computers to predict traffic on real Californian roads, using the same kind of sensors we use.

**Middle school.** Two research papers that started modern traffic forecasting. DCRNN treats traffic as something that spreads along the road (diffusion) and uses a recurrent network to follow it in time. STGCN uses convolutions over time and over the road graph instead. Both are tested on PEMS-BAY, which is Bay Area PeMS data like ours, at 15, 30 and 60 minutes ahead.

**High school.** First pass: the abstract, figure 1, the results table (mean absolute error at 15, 30 and 60 minutes) and the datasets section. Note the baselines (historical average, ARIMA, vector autoregression, a plain LSTM) and how the gains grow with horizon. That table is your yardstick for the spatio-temporal network in stage 9.

**Stanford undergrad.** DCRNN: diffusion convolution with forward and backward random-walk matrices on a directed graph, GRU cells whose matrix multiplies are replaced by graph convolutions, and a sequence-to-sequence architecture with scheduled sampling. STGCN: gated temporal convolutions sandwiching Chebyshev graph convolutions, with no recurrence. Work through each equation; the plan's spatio-temporal net is a two-hop diffusion GRU without the decoder.

**Stanford grad.** Read critically: sensitivity to how the graph is built (a Gaussian kernel on road distance with a threshold), the evaluation protocol (12-step windows, masking missing values), and later critiques showing that simple baselines are competitive at short horizons. Graph WaveNet's adaptive adjacency is the natural next read.

**PhD at Apple.** These papers set the benchmark protocol the field still uses, and their limitations (fixed sensor sets, no incidents or weather, no transfer across networks) are the open problems. Reproducing them on the M1 with a reduced configuration and then adding incident and weather features to the same protocol is a publishable direction.

### Build

- `17a`: a numpy MLP with a hand-written backward pass checked against finite differences.
- The same MLP in PyTorch with matching loss curves.
- `17b`: the spatio-temporal net on MPS with the registry's road graph; a CPU versus MPS timing table.
- Put its validation number next to LightGBM's and apply the plan's 0.02 rule.

### Done when

- Your backward pass matches autograd to 1e-6.
- The net trains without NaN, early-stops, and sits in the report next to the trees.
- You can draw its data flow from memory.

### Pitfalls

- float64 on MPS.
- Forgetting `model.eval()` at prediction time.
- A learning rate ten times too high.
- Believing the deep model must win.

## Phase 7: ML engineering

_weeks 32–40 · ≈50 h · builds stages 12 and 13_

**Goal.** The pipeline runs itself on the Mac mini, and you could hand it to someone.

### What you need to learn

- Reproducibility: seeds, pinned versions, a Makefile, one command from raw to report.
- The one-feature-module rule and train-serve skew; testing ML code with fixtures and golden files.
- Experiment tracking (a JSON log per run, or MLflow locally); a model registry as a folder with a manifest; rollback.
- Scheduling with launchd; logging; alerting when nothing arrives.
- Monitoring: drift with PSI, performance decay, retraining policy; delivered lead versus nominal lead.
- Memory profiling on unified memory; documentation for the next person; what a false alarm costs a real driver.

### Lesson 12: Shipping it on one Mac mini: memory, scheduling, monitoring

_Lives in stages 12 and 13 · 19_predict_live.py, 20_monitor.py, launchd/._

**Goal.** Run the whole thing unattended on 16 GB, know when it is failing, and retrain without breaking what works.

**Age 5.**

Our helper has to run all by itself every five minutes on one small computer that must never fall asleep.

It also has to check its own work. If it starts guessing badly, someone should find out, and it should learn again from the new days.

_Try this._ Set a timer to water a plant every day. What happens if the phone with the timer runs out of battery? How would you notice?

**Middle school.**

The Mac mini has 16 GB of memory shared by everything: the screen, the browser, and our program. A year of Fremont data is about 10 million rows; as 60 columns of numbers that is about 2.5 GB, so it fits, but only if we store numbers compactly and never run two big jobs at once. The Chrome downloader alone takes a gigabyte.

Jobs are scheduled by macOS with launchd: ask 511 every 15 minutes, predict every 5 minutes, check accuracy weekly, retrain monthly.

Monitoring means comparing last week's predictions with what really happened once the PeMS file for those days arrives, and checking that the clues still look like the ones the model trained on.

_Try this._ Compute 10,000,000 rows × 60 columns × 4 bytes. Convert to gigabytes. How many years fit in 12 GB if nothing else is running?

**High school.**

The serving loop: load the three models and the calibration file once; every five minutes build the last 13 steps of features with the same code that built the training table, predict, calibrate, apply the thresholds, write a JSON file, and append to a log.

Memory rules: float32, categorical ids, monthly Parquet parts, one heavy job at a time, three gigabytes kept free. Scheduling rules: launchd plists with `StartInterval`, Energy settings set to never sleep, `caffeinate` around the long download.

Monitoring: population stability index per feature against the training distribution, and average precision recomputed over the last seven days when PeMS posts them. Retrain monthly, or sooner on a real drop. The known gap: there is no real-time speed feed yet, so the loop runs in replay mode on yesterday's file.

_Try this._ Write the launchd plist for the predictor with a 300-second interval, and explain in two sentences why cron is worse on a machine that sleeps.

**Stanford undergrad.**

Train-serve skew is the classic failure: a second implementation of the features drifts within weeks. The defense is one module imported by both paths plus a golden-file test that recomputes one stored day and compares bit for bit. Point-in-time correctness follows: serving must only see what was available at t, and so must the training table.

Operational mechanics: idempotent jobs, atomic model swaps (write to a temporary name, rename), predictions logged with the newest input timestamp they used, and delayed scoring once labels arrive a day later. Drift: PSI = Σ (p_i − q_i) ln(p_i / q_i) over bins; a shift of one standard deviation in a normal feature gives PSI near 0.5, well past the 0.2 alarm.

Memory on unified hardware: measure with `psutil` and `tracemalloc`, set `OMP_NUM_THREADS`, and treat swap as failure. Reproducibility: pinned requirements, seeds, and a manifest with content hashes.

_Try this._ Implement PSI with ten quantile bins from the training distribution and compute it for a feature shifted by 0.25, 0.5 and 1.0 standard deviations. Where would you set the alarm?

**Stanford grad.**

Treat the pipeline as a reliability problem: service objectives for freshness and latency, alerts on missing data (silent failure is the default failure), canary evaluation of a retrained model on the most recent month before it replaces the champion, rollback, and a champion-challenger record.

Retraining is a control problem: frequency against drift speed against cost, with the risk of feedback loops once alerts change behavior. Continual learning without forgetting is unsolved in general; periodic full retrains with the newest month always in the window are the robust default.

Governance: raw immutable, derived reproducible, lineage in the manifest, and metric definitions versioned with the label so that a change in one is visible in the other.

_Try this._ Design the canary rule that decides when a retrained model may replace the old one, using a statistical test that respects temporal dependence, and state its false-promotion rate.

**PhD at Apple.**

At scale the same stage becomes streaming feature pipelines with exactly-once semantics, online feature stores, model registries, shadow deployments and staged rollouts, and on-device deployment with privacy-preserving telemetry. The Mac mini is a faithful miniature of on-device budgets: memory, energy, thermal throttling, and no operator watching.

Research threads: performative prediction and feedback loops (next month's training data contains the effect of this month's alerts), monitoring without labels through uncertainty and density-ratio drift detectors, and continual learning with guarantees.

_Try this._ Propose how you would detect that the model's own alerts are changing the label distribution, and how you would correct the training data or the objective for it.

**Guided walkthrough.**

1. The serving loop's core, using the training feature module.

   ```python
   from features import build_features           # the same module 13_feature_engineering.py uses
   import json, lightgbm as lgb, joblib
   models = {h: lgb.Booster(model_file=f"models/gbt_h{h}_thresh0.6_win3.txt") for h in (15, 30, 60)}
   cal = json.load(open("models/calibration.json")); isos = {h: joblib.load(f"models/iso_h{h}.joblib") for h in (15, 30, 60)}
   buf = load_last_13_steps()                       # replay mode: slice yesterday's Parquet at the wall-clock slot
   X = build_features(buf).iloc[-1 * N:]            # the newest row per station
   out = {"generated_at": pd.Timestamp.utcnow().isoformat(), "input_ts": str(buf.timestamp.max()), "rows": []}
   for h, m in models.items():
       q = isos[h].predict(m.predict(X[feats]))
       for sid, p in zip(X.station_id, q):
           out["rows"].append({"station_id": int(sid), "h": h, "p": float(p), "alert": bool(p >= cal[str(h)]["tau"])})
   json.dump(out, open("output/predictions_latest.json.tmp", "w")); os.replace("output/predictions_latest.json.tmp", "output/predictions_latest.json")
   ```

2. Population stability index against the training distribution.

   ```python
   def psi(train_col, live_col, bins=10):
       edges = np.quantile(train_col.dropna(), np.linspace(0, 1, bins + 1)); edges[0], edges[-1] = -np.inf, np.inf
       p = np.histogram(train_col.dropna(), edges)[0] / train_col.notna().sum() + 1e-6
       q = np.histogram(live_col.dropna(), edges)[0] / live_col.notna().sum() + 1e-6
       return float(np.sum((p - q) * np.log(p / q)))
   ```

3. Delayed scoring once PeMS posts the day.

   ```python
   log = pd.read_parquet("output/predictions_log.parquet"); truth = pd.read_parquet("data/pems_processed/2026-11.parquet")[["station_id", "timestamp", "label"]]
   log["target_ts"] = log.input_ts + pd.to_timedelta(log.h, "min")
   scored = log.merge(truth, left_on=["station_id", "target_ts"], right_on=["station_id", "timestamp"])
   print(scored.groupby("h").apply(lambda g: ap(g.label, g.p)))
   ```

4. A launchd plist for the predictor.

   ```xml
   <?xml version="1.0" encoding="UTF-8"?>
   <plist version="1.0"><dict>
     <key>Label</key><string>com.traffic1.predict</string>
     <key>ProgramArguments</key><array><string>/Users/you/Traffic1/.venv/bin/python</string><string>/Users/you/Traffic1/19_predict_live.py</string></array>
     <key>StartInterval</key><integer>300</integer>
     <key>WorkingDirectory</key><string>/Users/you/Traffic1</string>
     <key>StandardOutPath</key><string>/Users/you/Traffic1/logs/predict.log</string>
     <key>StandardErrorPath</key><string>/Users/you/Traffic1/logs/predict.err</string>
   </dict></plist>
   ```

5. A memory line in every stage's log.

   ```python
   import psutil, time, os
   t0 = time.time(); proc = psutil.Process(os.getpid())
   # ... stage work ...
   print(f"stage=clean file={path} seconds={time.time()-t0:.1f} rss_gb={proc.memory_info().rss/1e9:.2f}")
   ```

**Check yourself.**

<details>
<summary>Why write the JSON to a temporary name and rename?</summary>

A reader that opens the file mid-write sees half a file. Rename is atomic on the same filesystem, so readers see either the old file or the new one.

</details>

<details>
<summary>Why log the input timestamp with every prediction?</summary>

Scoring later needs to know what the model knew; and the gap between input time and generation time is the real latency a user experiences.

</details>

<details>
<summary>What is wrong with retraining every night automatically?</summary>

Nothing checks the new model before it replaces the old one. A bad day of data produces a bad model that ships at midnight. Canary first, then swap.

</details>

**Misconceptions.**

- 'It works on my laptop, so it will run unattended.' Sleep, reboots, a changed login page and a full disk are the normal failures; none show up in a demo.
- 'Monitoring is accuracy.' Freshness and missing data fail first; accuracy is the last thing to notice.
- 'More memory would fix it.' Filtering at ingest, float32 and one job at a time fix it; memory would only hide the design.

### Read and watch, explained at every level

#### Designing Machine Learning Systems (Huyen), chapters 1–3 and 7–9

_paid; the systems view in one book_

**Age 5.** A book about how to make a learning machine work in the real world, day after day, not just once on a laptop.

**Middle school.** Chip Huyen's book (paid). Chapters 1 to 3 cover what an ML system is, how to set up a project, and how data flows through it. Chapters 7 to 9 cover putting a model into service, watching it for problems, and updating it as the world changes. It is the textbook for phase 7.

**High school.** Chapter 8's data distribution shifts and monitoring, and chapter 9's continual learning and testing in production, map directly onto `20_monitor.py`. Chapter 7's deployment patterns (batch versus online prediction, model compression) frame the decision to run in replay mode until a real-time feed exists.

**Stanford undergrad.** The book gives vocabulary and checklists (feature stores, model registries, data validation, drift detection) and case studies; it is light on math and code, so pair it with Made With ML for implementations. Its discussion of training-serving skew is the reason this pipeline has one feature module and a golden-file test.

**Stanford grad.** Its treatment of feedback loops and degenerate feedback (recommendation systems that train on their own outputs) applies to alerts that change traffic. Its chapter on the human side, stakeholders and responsible AI, is practical rather than preachy, and its framing of monitoring as a product requirement rather than an afterthought is the right one.

**PhD at Apple.** The systems view it presents is the current industry consensus, and the research questions it leaves open (monitoring without labels, continual learning with guarantees, evaluation under feedback) are exactly the ones in lesson 12's highest level. For a lead it is also the book to align a team's vocabulary before an architecture discussion.

#### Made With ML (Mohandas)

_free; testing, monitoring and pipelines with code_

**Age 5.** A free website that shows, with real code, how to build a learning machine the careful way, with tests and checks so it keeps working.

**Middle school.** Goku Mohandas's course walks through an ML project end to end: data, modelling, and then the operational parts, testing, monitoring, pipelines and serving, all with code you can read. It is free. Follow the operational sections in phase 7.

**High school.** Its testing lesson (data tests, model tests, behavioral tests) and its monitoring lesson (drift detection with statistical tests) are directly reusable in stage 13. Its pipeline lesson shows orchestration that you can scale down to launchd jobs on one Mac mini without losing the ideas.

**Stanford undergrad.** Compare its tooling with a minimal setup: one Mac mini does not need Airflow or a feature store, but it needs the ideas (idempotent tasks, lineage, validation, versioning). Extract the principles and implement them in a few hundred lines rather than adopting the stack.

**Stanford grad.** Its coverage of experiment tracking and model versioning gives the vocabulary to design a JSON-based registry for this project, and its lessons on scaling with Ray are for later, when the data outgrows one machine. The course's opinionated structure is itself a useful design for an internal curriculum.

**PhD at Apple.** A free, opinionated, code-first MLOps curriculum is rare. Use it to align a team on practices before choosing a platform, and note which of its practices survive the translation to a single-machine deployment: most of them, which says something about what MLOps actually is.

#### Machine Learning Engineering (Burkov)

_free to read online_

**Age 5.** A short, plain book about the whole job of building learning machines, from getting the data to keeping them running.

**Middle school.** Andriy Burkov's book, free to read online under a 'read first, buy later' model. It covers the project lifecycle: collecting data, building features, training, evaluating, deploying, and monitoring. It is concise, and each chapter reads like a checklist you can apply to a stage of this project.

**High school.** Its chapters on data leakage, on evaluation and on deployment are the fastest reads on those topics. The leakage chapter lists exactly the traps lessons 4 to 6 cover: target leakage, temporal leakage, and contamination between train and test. Read it before building the feature table.

**Stanford undergrad.** Burkov's earlier Hundred-Page Machine Learning Book is the companion for the math; together they are a compact reference set. Use the engineering book's checklists as review criteria for each stage: does stage 8 pass the leakage checklist, does stage 12 pass the deployment one.

**Stanford grad.** It is opinionated and short. Where it disagrees with Huyen (for instance, on how much automation a small team needs), both positions are worth thinking through, and the disagreement itself is a useful prompt for deciding what this project's phase 7 should actually build.

**PhD at Apple.** A concise practitioner reference is useful for writing internal guidelines, and its brevity forces prioritization, which senior engineers need more than additional content. Its structure is a reasonable skeleton for an engineering-standards document for an ML team.

#### Hidden Technical Debt in Machine Learning Systems (Sculley et al., 2015)

_free; ten pages, read twice_

**Age 5.** A short paper by people who build big systems, saying: the learning part is small, and the messy parts around it are big and bite you later if you are not careful.

**Middle school.** Ten pages, free. It lists the ways ML systems rot over time: tangled dependencies on data, glue code nobody owns, pipelines nobody understands, feedback loops where the model changes the data it learns from, and configuration mess. Read it once now and once after building phase 7, when every paragraph will sound familiar.

**High school.** Its famous figure, a tiny box labelled ML code surrounded by large boxes of infrastructure, is a picture of this pipeline. Its warnings about correction cascades, undeclared consumers and data dependencies map to concrete rules here: one feature module, a manifest with hashes, versioned labels, and no script that silently reads another's output.

**Stanford undergrad.** Treat each named debt as a checklist for the project: entanglement (changing one feature changes everything downstream), hidden feedback loops (alerts change traffic), pipeline jungles (stages 5 to 8 as separate scripts), dead experimental code paths, and configuration debt. Write one sentence per item about where it could appear here and what prevents it.

**Stanford grad.** Later work, on ML test scores and on data-validation systems for production pipelines, operationalizes the paper's warnings; read it together with those follow-ups and with Huyen's chapter 8. The paper's concept of the CACE principle, changing anything changes everything, is the reason retraining needs a canary.

**PhD at Apple.** A foundational engineering paper whose legacy is the field of ML systems and MLOps. At your level the exercise is to separate which debts are inherent to ML (data dependencies, feedback) from which are ordinary software engineering, and to identify where research, rather than discipline, could reduce the inherent ones.

#### launchd.info and the pytest documentation

_free_

**Age 5.** Instructions for two tools: one that makes the Mac run a program on a schedule by itself, and one that runs all your checks with a single command.

**Middle school.** launchd.info explains macOS's scheduler: you write a small file called a plist saying what to run and how often, and macOS runs it even after a restart. The pytest documentation explains how to write and run tests. Read the getting-started page of each and try the smallest example.

**High school.** launchd: `StartInterval` and `StartCalendarInterval`, `WorkingDirectory`, log paths, `launchctl load` and `unload`, and why it beats cron on a Mac (it handles sleep and wake, runs per user, and restarts jobs). pytest: fixtures, `parametrize`, `-k` to select tests, `-x` to stop early, `tmp_path` and `monkeypatch`, and `conftest.py` for shared fixtures.

**Stanford undergrad.** Understand launchd agents versus daemons and the permission model (Full Disk Access for reading files, the keychain for secrets), because a job that works in the terminal and fails under launchd is almost always a permissions or environment problem. In pytest, fixture scoping and the plugin ecosystem (`pytest-cov`, `hypothesis`) are the next layer.

**Stanford grad.** Scheduling is orchestration at small scale; know its failure modes (overlapping runs, missed intervals after sleep, environment differences) and their mitigations (lock files, idempotent jobs, explicit environment in the plist). For testing, property-based tests with hypothesis are the right tool for parsers and feature code, where the input space is large.

**PhD at Apple.** Tooling choices are context-dependent. The durable skill is specifying a job's contract, idempotence, timeout, alerting, and retry policy, independently of the scheduler, so that migrating from launchd to a cluster scheduler later is mechanical rather than a redesign.

### Build

- `19_predict_live.py` in replay mode over yesterday's file; `20_monitor.py`; four launchd plists.
- `make setup` and `make all` that rebuild features, models and the report from raw.
- A golden-file test: serving features equal training features for one stored day, bit for bit.
- A memory and time line in every stage's log; a README per stage.

### Done when

- From a fresh clone, `make setup && make all` produces the evaluation report.
- The replay loop has run unattended for a week and the monitor has written four weekly files.

### Pitfalls

- A second copy of the feature code.
- cron instead of launchd on a Mac that sleeps.
- No way to roll back a bad model.

## Phase 8: Depth

_weeks 40–44, then onward · open-ended · builds papers, PEMS-BAY, write-ups_

**Goal.** From 'can build this' to 'can design the next one and say why it works'. Rotate the tracks; each is months, not weeks.

### What you need to learn

- Uncertainty: Bayesian thinking, quantile regression, conformal prediction for honest intervals on 'slow in 30 minutes'; causal inference basics, and why incident features are not causal claims.
- Theory: convexity, why boosting works, generalization and the bias-variance decomposition properly; CS229 notes; Elements of Statistical Learning chapters 7, 10 and 15.
- Deep learning depth: transformers properly, attention for time series, graph learning (CS224W), self-supervision; Goodfellow as the reference.
- Benchmarks: reproduce DCRNN or Graph WaveNet on METR-LA and PEMS-BAY, the public PeMS Bay Area datasets, with a reduced config on the M1; compare with the Fremont numbers.
- Practice: two Kaggle competitions on tabular or time-series data, written up honestly; one contribution to a library you used (a docs fix counts).
- Communication: publish a write-up of this project with its limitations; give a talk to a meetup or a friend.

### Topic: Uncertainty: quantiles, Bayesian thinking, conformal prediction

**Goal.** Say how sure the model is, with guarantees you can defend.

**Age 5.**

Sometimes the honest answer is 'probably between 20 and 40 minutes', not one number. A good helper says how sure it is and gives a range.

_Try this._ Guess how many candies are in a jar as a range you are pretty sure about. Was the true number inside your range?

**Middle school.**

A single number hides how wrong it might be. A range ('between 20 and 45, nine times out of ten') is more useful. You can get one by predicting quantiles directly, or by conformal prediction: check on held-out data how wrong the model usually is and widen the range until it covers 90 % of cases.

_Try this._ For ten past predictions, compute the errors, find the ninth largest, and use it as a plus-or-minus width. Check how many of the ten it covers.

**High school.**

Quantile regression with the pinball loss; prediction intervals versus confidence intervals. Bayesian thinking: a prior belief updated by data into a posterior. Conformal prediction: calibrate a score on held-out data, take its (1 − α) quantile, and get guaranteed coverage with almost no assumptions, provided calibration and test data are exchangeable. For time series that assumption is stretched, and adaptive methods exist.

_Try this._ Build a 90 % interval for the 30-minute speed with LightGBM quantile objectives at 0.05 and 0.95, and measure coverage on the test split.

**Stanford undergrad.**

Aleatoric versus epistemic uncertainty; calibration of intervals; Bayesian logistic regression with a Laplace approximation; Monte Carlo dropout and deep ensembles as approximate posteriors; split conformal and conformalized quantile regression with their coverage proof; risk control for binary alerts.

_Try this._ Prove split-conformal coverage from exchangeability of the calibration and test scores.

**Stanford grad.**

Conformal prediction under shift (weighted conformal, adaptive conformal inference for time series), risk-controlling prediction sets, conditional coverage and its impossibility results, hierarchical Bayesian models for per-station effects with partial pooling, and decision-making under uncertainty.

_Try this._ Implement adaptive conformal inference on the rolling predictions and plot realized coverage week by week.

**PhD at Apple.**

Uncertainty in products: what users do with intervals, communicating probability, alert budgets enforced with guarantees, and the research frontier of distribution-free inference under dependence and feedback, Bayesian deep learning at scale, and calibration under performative shift.

_Try this._ Propose a coverage guarantee that survives the feedback loop in which alerts change the traffic they predicted.

**Check yourself.**

<details>
<summary>What does 90 % coverage mean?</summary>

Over many predictions, about 90 % of intervals contain the truth. It does not mean one specific interval has a 90 % chance.

</details>

<details>
<summary>Why does conformal need held-out data?</summary>

It measures how wrong the model is on data it did not fit. Training errors are optimistic.

</details>

### Topic: Theory: why learning works at all

**Goal.** Know what generalization is, why more capacity is not always better, and what assumptions every guarantee rests on.

**Age 5.**

If you practice a game on one level only, you might get great at that level and bad at new ones. Learning that works means being good on new levels too.

_Try this._ Learn a card trick with one deck, then try it with a different deck. Does it still work?

**Middle school.**

A model can memorize its training rows (overfitting) or be too simple to fit them (underfitting). The goal is doing well on new data. More data helps, simpler models overfit less, and checking on held-out data tells you where you are.

_Try this._ Draw eight points. Fit a straight line and a wiggly curve through them. Which guesses a ninth point better?

**High school.**

Training error versus test error; the bias-variance trade-off; complexity (leaves, depth, layers) and regularization; cross-validation to estimate test error, forward-only for time series. The no-free-lunch idea: assumptions, called inductive bias, are what let a model generalize.

_Try this._ Plot validation average precision against `num_leaves` from 7 to 511 and find where it peaks.

**Stanford undergrad.**

Empirical risk minimization; the bias-variance decomposition for squared loss; uniform convergence, VC dimension and Rademacher complexity as intuition; regularization as constraining the hypothesis space; convexity and tractable optimization; margin theory for boosting. Bounds are loose, but they explain the shape of learning curves.

_Try this._ Derive the bias-variance decomposition and identify each term for a k-nearest-neighbors model as k changes.

**Stanford grad.**

Modern puzzles: interpolating models that generalize, double descent, the implicit regularization of stochastic gradient descent, benign overfitting; PAC-Bayes and information-theoretic bounds; and how distribution shift (covariate, label, concept) breaks every bound unless an invariance assumption restores it.

_Try this._ Reproduce a double-descent curve on a small synthetic regression problem and explain the peak.

**PhD at Apple.**

Theory as engineering guidance: which knobs matter (data quality over model size for tabular problems), scaling laws and where they fail, the theory-practice gap in deep learning, and the open problems relevant here: learning under dependence and feedback, and transfer across road networks.

_Try this._ State precisely what assumption would let a model trained on Fremont transfer to San Jose, and design the test that would falsify it.

**Check yourself.**

<details>
<summary>Why does a bigger model not always win?</summary>

Variance. It fits noise unless data or regularization constrains it.

</details>

<details>
<summary>What is inductive bias?</summary>

The assumptions built into a model class. Without any, nothing generalizes.

</details>

### Topic: Reading papers and reproducing benchmarks

**Goal.** Turn a paper into a change you could make to stage 9, and know whether its claim is real.

**Age 5.**

Scientists write down what they tried and what happened so others can try it too. Reading their notes and redoing their experiment is how you learn the newest tricks and check they are real.

_Try this._ Follow a recipe exactly. Then change one thing and see what happens.

**Middle school.**

A paper has an abstract (the summary), a method, experiments with tables, and a conclusion. Read the abstract and the tables first and ask: what did they compare with? Reproducing means running their code, or rewriting it, on the public data and getting close to their numbers.

_Try this._ Find the DCRNN paper online. Write down its three horizons and its best number at 60 minutes.

**High school.**

Three passes: title, abstract, figures and conclusion; then method and experiments with attention to baselines and datasets; then re-derive the key equations and read the code. For traffic, METR-LA and PEMS-BAY are the standard datasets, PEMS-BAY being Bay Area PeMS with 325 sensors, and the metrics are MAE, RMSE and MAPE at 15, 30 and 60 minutes. On the M1, reduce epochs, hidden size and batch, but keep the evaluation protocol identical.

_Try this._ Run the public DCRNN code on PEMS-BAY with a reduced configuration and compare your numbers with the paper's table.

**Stanford undergrad.**

Evaluating claims: is the baseline tuned, is the split honest, is the gain within seed-to-seed noise, and what does it cost in compute? Reproduction discipline: pin versions, record seeds, log everything, report variance. A reproduction report has method, deviations, results and a judgment.

_Try this._ Write a one-page reproduction report of STGCN with a table of the paper's numbers against yours.

**Stanford grad.**

The reproducibility literature shows many published gains vanish under fair baselines. Ablations isolate what matters. A benchmark score is not a deployable model. Read with a question: what would this change in my stage 9? Graph WaveNet and later transformer models are the next reads; understand adaptive adjacency versus a physical graph.

_Try this._ Run an ablation on your spatio-temporal net: remove neighbor mixing and quantify the change in average precision at 60 minutes.

**PhD at Apple.**

Research practice: choosing problems where data is the bottleneck, designing experiments whose failure would be informative, reviewing, and the ethics of claims. For traffic forecasting the open questions are transfer, non-stationarity, causal intervention and coupling with routing, and product impact often comes from data and evaluation rather than architecture.

_Try this._ Write the related-work paragraph a paper on this Fremont model would need: the three papers it must cite and the gap it fills.

**Check yourself.**

<details>
<summary>What is the first thing to check in a results table?</summary>

The baselines, and whether persistence or a simple model is among them.

</details>

<details>
<summary>Why reproduce with a reduced configuration?</summary>

To verify the pipeline and the protocol on your hardware. The absolute number scales with compute you can add later.

</details>

### Read and watch, explained at every level

#### Probabilistic Machine Learning: An Introduction (Murphy)

_free PDF; the modern reference_

**Age 5.** A very big free book that explains almost all of machine learning using the language of chance.

**Middle school.** Kevin Murphy's book, free as a PDF, about a thousand pages. It is a reference, not a read-through: look up a topic (logistic regression, decision trees, calibration, neural networks) and read that section. Its pictures are excellent and its examples are consistent from chapter to chapter.

**High school.** Use its chapters on probability and statistics (2 to 5), on linear models (9 to 11) and on trees and ensembles (18) as the rigorous version of ISLP. Its notation is consistent, and its habit of connecting every method to a likelihood shows that lessons 7, 8 and 9 are the same idea with different function classes.

**Stanford undergrad.** This is the graduate-level unification: every model as a probabilistic model, every loss as a negative log-likelihood, Bayesian and frequentist views side by side. Read the chapters on generalization, on optimization, and on calibration; its treatment of decision theory frames the alert threshold of lesson 11 properly.

**Stanford grad.** The second volume, Advanced Topics, covers inference, generative models and causality; between the two they are the modern reference set. Its treatment of Gaussian processes and Bayesian neural networks is the background for the uncertainty track of phase 8, and its chapters on time series connect state-space models to recurrent networks.

**PhD at Apple.** Murphy's books are the field's encyclopedia. Cite them for standard results and use them to find the original references quickly; for research, their breadth is what lets you place a new idea in the landscape in an afternoon rather than a week.

#### The Elements of Statistical Learning (Hastie, Tibshirani, Friedman)

_free PDF_

**Age 5.** A famous, hard, free book by three statisticians who invented many of the methods it describes.

**Middle school.** The graduate version of ISLP, free as a PDF, with proofs and depth. Read a chapter when you have met the topic in ISLP and want to know why it works: chapter 7 on judging models, chapter 10 on boosting, chapter 15 on random forests. Expect to read slowly and to reread.

**High school.** Chapter 10 is the origin story of gradient boosting written by its inventor, and it explains why fitting small trees to residuals works. Chapter 7 is the honest treatment of bias, variance and cross-validation, including why the test error estimate from cross-validation is itself uncertain. Chapter 15 explains why random forests work through variance reduction.

**Stanford undergrad.** ESL's derivations (boosting as additive modelling in function space, the bias-variance decomposition, effective degrees of freedom, shrinkage) are the background every ML PhD assumes. Work the exercises for chapters 7 and 10; they are the ones that make the later literature readable.

**Stanford grad.** Its coverage predates deep learning's rise and the modern boosting libraries, but its statistical framing (estimators, degrees of freedom, shrinkage, model selection) is what explains why those tools work. Pair chapter 10 with the XGBoost and LightGBM papers to see the second-order approximation that made boosting fast.

**PhD at Apple.** Still the canonical citation for many methods, and its authors' later work on sparsity and the lasso is the natural continuation. For a paper, its notation for loss functions and additive models is safe to adopt, and its chapter 7 remains the reference for arguing that a reported gain is within noise.

#### CS229 lecture notes (Stanford)

_free_

**Age 5.** The notes from a famous university class about learning machines, free for anyone to read.

**Middle school.** Stanford's machine learning course notes: linear regression, logistic regression, a family of related models called generalized linear models, models that describe how data is generated, support vector machines, and learning theory. Dense but clear. Read the first sections after phase 4, one section per sitting.

**High school.** The logistic regression and generalized-linear-model sections give lesson 7's math in the standard notation, including the derivation of the gradient that this course rebuilds by hand. The learning-theory notes on bias, variance and uniform convergence are the source material for the theory ladder in phase 8.

**Stanford undergrad.** The notes are the compact graduate treatment: exponential families unify linear, logistic and Poisson regression as one construction; the learning theory section introduces PAC-style bounds; the notes on regularization and model selection are practical. The public problem sets are worth doing for the theory sections.

**Stanford grad.** Compare CS229's derivations with ESL's, then read the newer notes on deep learning and on generalization. The course's treatment of the EM algorithm and of factor analysis is the background for latent-variable models of traffic regimes, which lesson 4's higher levels gesture at.

**PhD at Apple.** CS229 is what most ML researchers learned from, so knowing it lets you calibrate what a reviewer or a new hire assumes as background, and what they will not have seen (dependent data, forecasting protocols, calibration under shift) and will need explained.

#### CS224W: Machine Learning with Graphs (Stanford)

_free lectures_

**Age 5.** A university class, free to watch, about learning from things that are connected to each other, like roads that join.

**Middle school.** Jure Leskovec's course: how to represent a network of connected things, how to summarize each node with numbers, graph neural networks (where each node learns from its neighbors), and applications. The lectures are free on YouTube. Watch the graph-neural-network lectures after phase 6.

**High school.** The lectures on message passing (graph convolutional networks, GraphSAGE, graph attention) explain what neighbor mixing in the spatio-temporal net is doing and how to generalize it: attention over neighbors, more hops, learned edge weights. The lecture on spatio-temporal graphs connects directly to DCRNN.

**Stanford undergrad.** Understand graph convolutions as generalizations of image convolutions, the role of degree normalization, over-smoothing with depth, and the expressiveness limits given by the Weisfeiler-Leman test. A road is a directed graph, so the operators must be asymmetric, which is what DCRNN's forward and backward diffusion provides.

**Stanford grad.** The course's coverage of heterogeneous graphs, knowledge graphs and scalable training (neighbor sampling) is the toolkit for city-scale models, and its assignments in PyTorch Geometric are a practical route to implementing Graph WaveNet without writing every operator by hand.

**PhD at Apple.** CS224W and the PyTorch Geometric ecosystem are the standard entry to graph machine-learning research. Its open problems (expressiveness, scalability, dynamic graphs, transfer across graphs) overlap with traffic forecasting's own, and the course's reading list is a good map of the field.

#### A Gentle Introduction to Conformal Prediction (Angelopoulos, Bates)

_free; intervals you can defend_

**Age 5.** A short guide about how to make a learning machine say 'I am 90 % sure the answer is in this range' and actually be right 90 % of the time.

**Middle school.** A tutorial paper, free, with code. It explains conformal prediction: measure how wrong your model is on data it has not seen, and use that to make ranges or sets of answers that hit the truth a guaranteed share of the time, no matter what model produced the predictions.

**High school.** Section 1 gives the recipe in a page: compute a score on calibration data, take a quantile of the scores, build the set. Later sections cover intervals for regression (conformalized quantile regression), sets for classification, and the assumptions. Apply the recipe to the 30-minute speed prediction and check coverage on the test split.

**Stanford undergrad.** The coverage guarantee follows from exchangeability in a few lines, which the tutorial proves. Understand why the guarantee is marginal rather than conditional, and what breaks under time-series dependence. The risk-control extension applies to controlling the false-alarm rate of a binary alert, which is lesson 11's problem.

**Stanford grad.** Follow with the authors' work on conformal risk control and with adaptive conformal inference for time series (Gibbs and Candès), and think about how to obtain guarantees for the episode-level false-alarm rate rather than the row-level one, which is what a driver experiences.

**PhD at Apple.** Conformal methods are the practical route to distribution-free guarantees in products. The research frontier, conditional coverage, dependence, and feedback between predictions and outcomes, is where the traffic alert problem sits, and the tutorial's clarity makes it the right document to hand to a product or legal partner.

#### Statistical Rethinking (McElreath)

_paid; the Bayesian track, lectures free_

**Age 5.** A book, with free video lectures, that teaches how to think carefully about chance and evidence, using stories and small models.

**Middle school.** Richard McElreath's Bayesian statistics course: you build a model by writing down what you believe, then update it with data, with lots of simulation and pictures. The book is paid; the lectures are free online. It belongs to the uncertainty track of phase 8.

**High school.** It teaches Bayesian thinking (priors, posteriors, why they help when data are scarce) and causal thinking (diagrams of cause and effect, confounding) in plain language. Both apply here: per-station effects with little data, and rain and incidents as common causes of slowness that confound a naive reading of the incident feature.

**Stanford undergrad.** Hierarchical models with partial pooling are the answer to 'per-station calibration is noisy': share strength across stations while letting each differ. Work through the chapters on multilevel models and on causal inference with the Python ports of the code, and apply partial pooling to the per-station thresholds of lesson 11.

**Stanford grad.** It contrasts with the frequentist habit of point estimates; its emphasis on generative models and posterior predictive checks is a different evaluation culture that is worth learning. Pair with Gelman and colleagues' Bayesian Data Analysis for depth and with the causal-inference literature for the diagrams.

**PhD at Apple.** McElreath's causal-first framing is influential in applied statistics. For traffic it is the right lens for questions prediction alone cannot answer, such as the effect of an alert or of an incident on downstream speeds, and its emphasis on stating assumptions explicitly is the discipline an alerting product needs.

#### Deep Learning (Goodfellow, Bengio, Courville)

_free online_

**Age 5.** The big classic book about deep neural networks, free online, by three of the people who built the field.

**Middle school.** Three parts: the math background, practical deep networks (feedforward nets, regularization, optimization, convolutional and recurrent networks), and research topics. Read chapters from the second part as references after phase 6, when the words will already be familiar.

**High school.** Chapter 6 on feedforward networks and backpropagation and chapter 8 on optimization are the rigorous versions of lesson 9; chapter 10 on sequence modelling covers recurrent networks and gated units; chapter 7 on regularization explains early stopping, dropout and weight decay and why they work.

**Stanford undergrad.** It is the theory reference: backpropagation as a general algorithm on computation graphs, the difficulties of optimization (saddle points, ill-conditioning), and the regularization view of generalization. It predates transformers and modern training recipes, so supplement with Dive into Deep Learning and recent papers.

**Stanford grad.** Part three (representation learning, structured probabilistic models, generative models) frames later research. Some material is dated, but the conceptual chapters remain the shared vocabulary of the field, and its chapter on practical methodology is a good checklist before any training run.

**PhD at Apple.** A canonical citation and a snapshot of the field as of 2016. Knowing what it lacks (attention, scaling laws, modern generative models, the empirical understanding of generalization) is as important as knowing what it contains, and reading it against Murphy's second volume shows how the field's framing has shifted.

#### Graph WaveNet (Wu et al., IJCAI 2019)

_free; the third traffic paper_

**Age 5.** A paper where scientists let the computer figure out for itself which roads affect which, instead of telling it.

**Middle school.** The third traffic paper. Instead of a fixed map of neighbors, Graph WaveNet learns which sensors influence which from the data (an adaptive adjacency), and it uses stacked convolutions over time instead of a recurrent network. It improved on DCRNN and STGCN on the same benchmarks.

**High school.** Read it after DCRNN and STGCN: what changes is the graph (learned from node embeddings) and the temporal model (dilated causal convolutions, cheaper to train than recurrence). Its ablation shows the learned graph helps even when a physical graph is available, which is a hint that the postmile line misses some dependencies.

**Stanford undergrad.** The adaptive adjacency, a softmax over the product of two learned embedding matrices, is a learned low-rank graph; combined with the diffusion operators it captures dependencies the road line misses, such as interchanges and parallel routes. Dilated convolutions give a long receptive field with few layers, which is why it trains fast.

**Stanford grad.** Later work (MTGNN, spatio-temporal transformers) generalizes the learned-graph idea. Evaluate whether learned graphs are interpretable and stable across retrains, which is a practical concern for monitoring: a graph that changes every month is hard to trust and hard to explain.

**PhD at Apple.** For Fremont the learned graph could reveal cross-freeway coupling (I-880 to I-680 through SR-84 and Mission), and testing whether that improves 60-minute skill is a concrete, publishable experiment that fits on the M1. The paper is also a good example of an ablation done properly.

### Build

- A PEMS-BAY reproduction that runs on the mini.
- A conformal interval on the 30-minute prediction, evaluated for coverage.
- A public write-up of the Fremont model, limitations first.

### Done when

- You read a new traffic-forecasting paper in an evening and say what it would change in stage 9 and what it would not fix.
- The write-up is public and someone else has run your code.

### Pitfalls

- Collecting courses.
- Never finishing a write-up.
- Optimizing a leaderboard instead of understanding.

## How to study so it sticks

- **A weekly rhythm at ten hours.** Three hours of reading or lectures with notes. Five hours building in the repo. One hour rewriting the week's notes from memory. One hour explaining it to someone, or to a file if no one is around.
- **The three passes.** Run it. Change it and predict what will happen before you run it. Rebuild it from a blank file. Only the third pass counts as knowing.
- **A lab notebook.** `notebook/YYYY-WW.md`: what you tried, what broke, what you learned, the number. Numbers without the setup that produced them are noise a month later.
- **A baseline next to every result.** Persistence, the profile rate, the previous version. No exceptions, including for plots.
- **What would fool this metric?.** Ask it of every number. Accuracy under imbalance, AUC with persistence, F1 without a threshold rule, a shuffled split on a time series.
- **Read in this order.** Code, then docs, then papers, then blog posts. Blog posts last because they are the easiest to write badly.
- **The stuck rule.** Thirty minutes alone. Then write the question down precisely, which solves half of them. Then ask, with the exact error and what you already tried.
- **One project.** This one is enough for ten months. A second project before phase 7 is a way to avoid finishing the first.
- **Rest.** Two long sessions beat five short ones for building; five short ones beat two long ones for math. Sleep is when it consolidates.

### Honest notes

- Editions and chapter numbers drift; if a chapter does not match its description, find the topic in the index. Availability of free editions can change.
- The hours assume a beginner. If phase 1 takes twenty hours, take the twenty; if it takes sixty, that is normal too. Reorder nothing.
- The math in phase 2 is the part people skip and the part that separates using ML from understanding it. Derive the gradient once by hand; everything in phase 6 is that derivation repeated.
- Being good at ML is mostly being good at evaluation. Phases 4 and 5 are where that is learned; do not shorten them to reach the neural networks sooner.
