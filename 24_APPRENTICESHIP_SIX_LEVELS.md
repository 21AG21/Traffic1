# Six Ladders: everything needed for machine learning, at six levels

_September 2026. Supersedes `22_LEARNING_PATH_ML.md` and `23_LESSONS_SIX_LEVELS.md`; companion to `21_PLAN_M1_TRAFFIC_MODEL.md`._

**Everything needed for machine learning, told six times: twelve phases, thirty-nine lessons and 140 free books, courses and papers, from age five to a PhD at Apple.**

The Fremont traffic model is the spine: you build it stage by stage and learn each idea the week it is needed. Around that spine the course covers the rest of the field, from SQL and regression to transformers, reinforcement learning, causal inference and fairness, so nothing needed for real machine-learning work is left out. Every lesson, topic and resource is explained at six levels, and every resource is free. Pick a level with the switcher and read the whole course at that height, or turn on every level and watch one idea gain precision rung by rung. Tick things off as you go; progress is saved in this browser.

The six levels, in order: Age 5, Middle school, High school, Stanford undergrad, Stanford grad, PhD at Apple. Every lesson, topic and resource below carries all six. Links were checked in September 2026.

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
- Phase 3 · Data handling, files and SQL (≈60 h)
  - Lesson 3: Cleaning: from a giant text file to a tidy table
  - Topic: SQL and databases: where data usually lives
- Phase 4 · Classical machine learning, the core (≈70 h)
  - Lesson 4: Labels: what does 'slow' mean?
  - Lesson 5: Features: turning the past into columns
  - Lesson 6: Baselines and honest evaluation
  - Lesson 8: Gradient boosted trees
- Phase 5 · Classical machine learning, the rest (≈50 h)
  - Topic: Regression: predicting a number, and the metrics that judge it
  - Topic: Other classifiers and multiclass: k-nearest neighbors, naive Bayes, support vector machines, softmax
  - Topic: Unsupervised learning: clustering, principal components and anomaly detection
  - Topic: Hyperparameter tuning and model selection
  - Topic: Information theory in machine learning: entropy, cross-entropy, KL divergence, mutual information
- Phase 6 · Time series and forecasting (≈45 h)
  - Lesson 10: Jams move: sequences, neighbors and the road graph
  - Lesson 11: From scores to alerts: calibration, thresholds, lead time
- Phase 7 · Neural networks (≈80 h)
  - Lesson 9: Neural networks and backpropagation, on the M1
- Phase 8 · Deep learning across domains (≈100 h)
  - Topic: Convolutional networks and computer vision
  - Topic: Language: tokens, embeddings, attention and transformers
  - Topic: Large language models: pretraining, fine-tuning, prompting, retrieval and evaluation
  - Topic: Generative models: autoencoders, variational autoencoders, GANs and diffusion
  - Topic: Transfer learning, self-supervision and foundation models
- Phase 9 · ML engineering and scale (≈70 h)
  - Lesson 12: Shipping it on one Mac mini: memory, scheduling, monitoring
  - Topic: APIs, containers and the cloud: serving a model beyond one Mac
  - Topic: Scaling: GPUs, distributed training and mixed precision
- Phase 10 · Beyond supervised learning (≈70 h)
  - Topic: Reinforcement learning: agents, rewards, bandits and policies
  - Topic: Causal inference: from correlation to intervention
  - Topic: Fairness, privacy and ethics: what a model owes people
  - Topic: Recommender systems and ranking
- Phase 11 · Depth (open-ended)
  - Topic: Uncertainty: quantiles, Bayesian thinking, conformal prediction
  - Topic: Theory: why learning works at all
  - Topic: Reading papers and reproducing benchmarks
  - Topic: Bayesian machine learning and Gaussian processes

## Six rules

- **The project is the syllabus.** Every topic in the first seven phases is here because a stage of the traffic model needs it, and you learn it the week you build that stage. The later phases cover the rest of machine learning with the same discipline: each topic ends in something that runs.
- **Build it twice.** First with the library, so it works. Then the core from scratch in numpy, the way `03_train_from_scratch.py` does, so you know what the library hides. Then back to the library, faster.
- **Evaluate before you model.** Baselines, metrics and leakage come before any algorithm past logistic regression. A number without a baseline next to it is not a result.
- **Math just in time.** Linear algebra, calculus and probability in the amounts the stage needs, derived once by hand. Depth comes in the final phase, when you know what it is for.
- **Write it down.** A lab notebook every week, a README per stage, a write-up per phase. If you cannot explain it in plain words you do not have it yet.
- **Checkpoints are things that run.** Each phase ends with code that produces a number or a file, not with a quiz.

### What really good looks like

- **Framing.** Turn a fuzzy question into a unit, a target, a horizon and a metric, and say what the answer cannot mean.
- **Data.** Build the pipeline from raw files or a database, know its failure modes, and measure its memory and time.
- **Models.** Know the whole toolbox, from logistic regression to transformers and graph networks, well enough to say which wins on which data and why, and to implement the core of any of them from a paper.
- **Evaluation.** Design a split that cannot leak, pick metrics the imbalance cannot fool, put a baseline next to every number, and calibrate what you ship.
- **Engineering.** Ship one implementation used by training and serving, containerize it, schedule it, monitor it, retrain it, and roll it back.
- **Acting.** Know what changes when a model's output is a decision: causal questions, feedback loops, fairness and privacy.
- **Reading.** Read a new paper in an evening and say what it would change in your system and what it would not fix.
- **Explaining.** Describe all of the above to a smart person outside the field without hiding behind jargon.

## Phase 0: Setup and habits

_weeks 0–1 · ≈8 h · builds stage 2 runs_

**Goal.** A working machine and a working routine.

### What you need to learn

- The terminal: `cd`, `ls`, `cat`, `head`, pipes, environment variables.
- Git: clone, branch, commit, push, diff, `.gitignore`; what a repository and a README are.
- Native arm64 Python on Apple silicon (miniforge or python.org), virtual environments, `pip`, VS Code, Jupyter.
- The shape of this repo: the numbered scripts, `data/`, `models/`, `.env`, and why secrets never get committed.
- What the traffic model predicts, so every later week has a destination.

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

#### [Software Carpentry: The Unix Shell, Version Control with Git](https://swcarpentry.github.io/shell-novice/)

_course · free · a day · free, half a day each, exactly the commands you will use_

**Age 5.** Two short lessons on the internet that teach you to talk to the computer by typing, and to use the magic notebook that remembers your work. You follow along and type every command yourself, and the lesson tells you what should happen after each one.

**Middle school.** Free workshops written for scientists who have never programmed. Each takes about half a day. The Shell lesson teaches moving between folders, listing files, and joining small commands together with pipes. The Git lesson teaches saving snapshots of your work and pushing them to GitHub so a copy lives online. Do them in order with a terminal open, and do the exercises; watching is not the same as typing.

**High school.** The Carpentries lessons are hands-on with checkpoints every few minutes. The shell lesson covers navigation, wildcards, pipes and filters, loops and simple shell scripts. The git lesson covers `init`, `add`, `commit`, `diff`, `log`, branches, remotes, and collaborating through GitHub, including how to resolve a conflict. That is exactly the set of commands this repository's workflow needs, and nothing you will not use. Budget one day for both and take notes on the commands you find yourself reaching for.

**Stanford undergrad.** These lessons are the minimum viable toolkit for reproducible work: scripting a procedure rather than clicking through it, and versioning every change so any result can be traced to the code that produced it. Their pedagogy, live coding with frequent small exercises, is worth copying when you later teach someone else. After them, the man pages and `git help <command>` become readable, and Pro Git's chapters 1 to 3 will feel like a review with more depth.

**Stanford grad.** Compact enough to finish in a day, they leave deliberate gaps you will fill later: shell robustness (`set -euo pipefail`, quoting, exit codes), git internals and history rewriting, and the difference between a merge and a rebase. Treat them as the vocabulary lesson before the grammar book. Their assessment data (they have been taught to tens of thousands of researchers and iterated on feedback) is a rare example of evidence-based technical teaching.

**PhD at Apple.** Written and maintained by The Carpentries community, these are the lessons to hand to a new hire from a domain science before they touch the pipeline. For a research lead the value is twofold: a shared baseline of tooling across a team, and a template for writing your own onboarding lesson for this repository in the same style, with exercises that run against real PeMS files.

#### [Pro Git, chapters 1–3](https://git-scm.com/book/en/v2)

_book · free · a weekend · free online; enough Git for a year_

**Age 5.** A book about a tool called git. Git is like a magic notebook that remembers every version of your drawing, so if you make a mess you can flip back to yesterday's page. The first three chapters teach you how to save a page, look back at old pages, and share the notebook with a friend.

**Middle school.** Pro Git is the free official book for git, the tool programmers use to save snapshots of their work and share them on GitHub. Chapter 1 explains what git is and installs it, chapter 2 covers the everyday commands (making commits, seeing what changed, undoing mistakes, working with a remote copy), and chapter 3 covers branches, which are parallel copies you can experiment on and merge back. Read it with a terminal open and run each command on a practice folder.

**High school.** The reference for git, free online, by Scott Chacon and Ben Straub. Chapter 1 explains the snapshot model: git stores whole snapshots, not differences, and a commit is a pointer to one. Chapter 2 is the daily toolkit: `init`, `clone`, `add`, `commit`, `status`, `diff`, `log`, `remote`, `push`, `pull`, plus tagging and aliases. Chapter 3 is branching and merging, including what a fast-forward is and how to handle a conflict. That is enough for a year of solo and small-team work; the later chapters (rebase, hooks, internals) can wait until you need them.

**Stanford undergrad.** The value of the first three chapters is the mental model rather than the commands: the three areas (working tree, index, repository), commits as immutable objects in a graph, and branches as movable pointers to commits. Once that picture is in your head, every confusing git situation becomes a diagram you can draw, and `git log --graph --oneline --all` becomes your debugger. Read chapter 3 twice; branching is where most people's understanding is vague.

**Stanford grad.** Chapter 10 on internals is the next read: the content-addressed object store (blobs, trees, commits named by hash) is the same idea as the manifest hashes in this pipeline and as data-versioning tools such as DVC and lakeFS. Git's model of history is what makes reproducible experiments possible: a commit hash plus a data hash names a result forever. Chapter 7's tools (stash, bisect, reflog, rewriting history) are what you reach for when an experiment goes wrong.

**PhD at Apple.** At scale git's limits show: monorepos, large binary artifacts (git LFS), and the mismatch between versioning code and versioning data or models, which is why model registries and data lakes grew their own lineage systems. Still, the discipline the book teaches, small atomic commits with messages that explain why rather than what, is what makes a research codebase reviewable and a production incident debuggable at 3 am. For a team, the book is also the reference for a written branching policy.

#### The repo's own `00_CONCEPTS.md` and `05_PLAN_COMBINED_DATASETS.md`

_docs · free · an hour, twice · read them once now, again in phase 4_

**Age 5.** The project's own notes, written by the people who built it. They explain what the helper is for and how the pieces fit together, like the instruction sheet that comes in a toy box.

**Middle school.** Two documents inside this repository. The concepts file explains the ideas behind the first model, including the math of the logistic regression, in plain language. The plan file describes how the PeMS sensor data and the 511 incident feed will be combined into one training table. Read them once now, even if half of it is unclear, and again after phase 4, when all of it will make sense.

**High school.** Read them like a map of the project. `00_CONCEPTS.md` explains the sigmoid, cross-entropy and gradient descent that lesson 7 rebuilds. `05_PLAN_COMBINED_DATASETS.md` sets out the stages, the label definition, the join rules between incidents and stations, and the time-zone warning that lessons 3 to 5 elaborate. Write down every term you do not understand; that list is your study plan for the next three phases.

**Stanford undergrad.** Project documentation is a primary source: it records decisions and their reasons, and it goes stale the moment code diverges from it. Compare the plan with the flowchart document (`21_PLAN_M1_TRAFFIC_MODEL.md`) and list what changed and why: the forward-shifted targets, the train-only profile, the second label, the missing real-time feed. That diff is a lesson in how plans evolve when they meet data.

**Stanford grad.** Read them critically. The original plan paired features at t with the label at t, and computed the profile on all history; both are design errors that would have produced flattering numbers. Finding such issues in a design document before code exists is a core research and engineering skill; practice it by writing a one-page review of `05_PLAN_COMBINED_DATASETS.md` as if it were a submission.

**PhD at Apple.** Design documents are where most of the leverage in an ML project lives, and most teams write them badly. The exercise at your level is to write the version a team could execute: explicit interfaces between stages, success criteria with numbers, risks with owners, and what would make you stop. Use these two files as the before, and the flowchart plan as a partial after.

#### [The Missing Semester of Your CS Education (MIT)](https://missing.csail.mit.edu/)

_course · free · a week · free; the tools nobody teaches, in eleven lectures_

**Age 5.** A short free class about the tools programmers use every day but nobody teaches: the typing window, the magic notebook, and how to make the computer do boring jobs for you.

**Middle school.** MIT students made a course about the things regular classes skip: the shell, editors, git, debugging, command-line tools, and automating tasks. Eleven lectures with notes and exercises, all free. Do the shell, editors and git lectures during phase 0.

**High school.** The lectures on the shell and shell tools, on editors (Vim), on data wrangling (pipes, grep, sed, awk), on git, and on debugging and profiling are directly useful. The git lecture's model of commits as a graph is the best short explanation there is; the data-wrangling lecture teaches you to inspect a PeMS file without Python.

**Stanford undergrad.** Its value is the mental models: the shell as a programming environment, git as a content-addressed graph, profiling before optimizing. The exercises are small and worth doing; the notes are good references for the flags you forget.

**Stanford grad.** A compact syllabus of practitioner tooling that most graduate students still lack; its debugging and profiling lecture is a fair summary of how to approach a slow or wrong pipeline stage.

**PhD at Apple.** For onboarding engineers and researchers alike, this is the shortest path to shared tooling literacy; its lecture on metaprogramming (build systems, testing, continuous integration) is the bridge to the engineering phase.

#### [Elements of AI (University of Helsinki)](https://www.elementsofai.com/)

_course · free · about 30 hours · free; the big picture with no math, before the phases begin_

**Age 5.** A gentle online course that explains what artificial intelligence is and is not, with pictures and simple quizzes, made for everyone.

**Middle school.** A free course from the University of Helsinki, no math required: what AI means, how problems are solved by search, how probability handles uncertainty, what machine learning is, how neural networks work, and what AI does to society. About thirty hours, spread over a couple of weeks. A good first map before the phases begin.

**High school.** Do it in the first week for the vocabulary and the big picture: the difference between rules and learning, why probability matters, what a neural network does in outline, and why AI's effects are a social question. The chapter on machine learning is a preview of phase 4 with no code.

**Stanford undergrad.** It is deliberately non-technical; use it to calibrate how you explain the course to others, and note where it simplifies (it treats learning as fitting without discussing evaluation).

**Stanford grad.** A model of public education about a technical field: careful about hype, honest about uncertainty, and concrete about societal effects.

**PhD at Apple.** Worth knowing as the reference for what a general audience can be expected to understand about AI, which matters when you write for policy makers or the public.

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

**Goal.** Read and write the collector scripts fluently, test them, and never be confused by a clock.

### What you need to learn

- Variables, numbers, strings, lists, dicts, sets, tuples; `if`, `for`, `while`; functions, arguments, return values, scope.
- Modules and imports, files, CSV and JSON, exceptions, `requests`, `argparse`, f-strings, comprehensions.
- Dates and time zones with `datetime` and `zoneinfo`; what UTC is and why the log stores it.
- The debugger, print-debugging with intent, and `pytest` for small functions.
- What a PeMS row and a 511 event actually measure.

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

#### [Automate the Boring Stuff with Python, chapters 1–13 and 16–18](https://automatetheboringstuff.com/)

_book · free · weeks · free online; practical from page one_

**Age 5.** A friendly book that teaches Python by making the computer do chores for you, like renaming lots of files at once or reading a spreadsheet. It is free to read on the internet, and every chapter ends with something useful you made.

**Middle school.** Al Sweigart's book starts from zero: numbers, text, lists, dictionaries, functions, and then practical chapters on files, spreadsheets, CSV and JSON, and the web. Chapters 1 to 13 are the language basics; 16 to 18 cover CSV and JSON, time and scheduling, and sending email and text messages, which are close to what our collector script does. Type every example rather than copying it, and do the practice projects at the end of the chapters.

**High school.** The best first Python book because every chapter ends in a program that does something real. For this project the payoff chapters are 9 and 10 (reading, writing and organizing files), 13 (spreadsheets), 16 (CSV and JSON, the two formats the pipeline lives on), and 17 (time and scheduling, which the poller needs). After it you can read `02_collect_events.py` end to end and change it with confidence. Its style, small scripts with clear names, is the right style for pipeline code.

**Stanford undergrad.** Its limits are exactly what the official tutorial and later phases fill: little on the data model (mutability, references), nothing on numpy, and almost nothing on testing. Use it for fluency and for the habit of automating anything you do twice, then move on. The chapter on debugging (chapter 11 in the current edition) is the one to reread when you are stuck on a stage 5 bug at midnight.

**Stanford grad.** Pedagogically it works through immediate utility, and that is the model for how to bring domain experts (a traffic engineer, a city planner) onto a data project: give them a script that saves them an hour on day one. If you already program, skip it yourself; recommend it to collaborators and write the project-specific chapter it lacks, 'automating the PeMS download and cleaning'.

**PhD at Apple.** The book's persistence as the default beginner text, across editions and free online, reflects a fact about ML teams: the bottleneck is often basic scripting fluency among the people who understand the domain. Building that capability across a team is a higher-leverage act than most modelling work, and this book is the cheapest tool for it.

#### [The official Python tutorial, sections 3–9](https://docs.python.org/3/tutorial/)

_docs · free · an afternoon · free; the reference once Sweigart runs out_

**Age 5.** The instruction book written by the people who make Python. It is for looking things up when the friendly book did not say.

**Middle school.** Sections 3 to 9 cover numbers and strings, control flow (`if`, `for`, `while`, functions), data structures (lists, dictionaries, sets, tuples), modules, reading and writing files, errors and exceptions, and classes. It is terser than Sweigart and assumes you will experiment. Read a section when you meet that topic in the code, then try its examples in the interpreter.

**High school.** The tutorial is the authoritative short version of the language: how `for` really works (iterators), how arguments are passed to functions, what a module is and how imports find it, how exceptions propagate up the call stack, and what a class is. Read it once fully after Sweigart, then keep the language reference for details. Section 5 on data structures and section 7 on input and output are the ones the pipeline code exercises most.

**Stanford undergrad.** This is where the data model becomes precise: mutable versus immutable objects, `is` versus `==`, how comprehension scope works, `*args` and `**kwargs`, `with` statements and context managers, and the exception hierarchy. Pair it with the library reference pages for `collections`, `itertools`, `pathlib` and `dataclasses`, which are the standard-library tools a clean pipeline uses.

**Stanford grad.** Beyond the tutorial, the language reference and the PEPs are where design rationale lives: PEP 8 for style, PEP 484 and successors for typing, PEP 20 for the philosophy. Reading a few accepted and rejected PEPs teaches how language decisions are argued with evidence, which is a useful model for arguing design decisions in your own project.

**PhD at Apple.** For a team the tutorial's value is as the shared baseline that code-review standards can point to. Beyond it, the investment that turns research code into production code is static typing (mypy or pyright on every module), packaging, and a linter configuration checked into the repository so style is never a review comment.

#### [Kaggle Learn: Python](https://www.kaggle.com/learn/python)

_course · free · a weekend · free, short, with exercises in the browser_

**Age 5.** A website where you learn Python in small steps and it checks your answers right away, like a game with levels.

**Middle school.** Seven short lessons in the browser with exercises that are graded the moment you run them. No installation. It covers the basics you also meet in Sweigart: functions, booleans, lists, loops, strings, dictionaries, and importing libraries. Use it for practice and repetition, not as your only source; finish it in a weekend.

**High school.** Good for building confidence through repetition. Each lesson pairs a short tutorial notebook with an exercise notebook that checks your answers. Because it runs in the cloud it hides environment setup, which is convenient here and exactly why phase 0 has you set up your own machine, so you learn the setup you will actually need.

**Stanford undergrad.** Kaggle's notebooks are a fine scratchpad and a poor habit if they replace a local environment: no version control, no tests, no reproducible dependencies. Take the practice, leave the workflow. The course's exercises on functions and on list handling are worth doing; the rest is a review.

**Stanford grad.** The Learn courses are a funnel into Kaggle competitions. Their pedagogy, tiny graded steps, is effective for syntax and weak for design and debugging, the two skills that decide whether a pipeline survives. Use them accordingly and move on quickly.

**PhD at Apple.** Kaggle as a platform matters more than this course. It is where tabular machine-learning practice evolved: gradient-boosted ensembles, feature engineering, leakage detection, and careful validation. Reading winning solutions' write-ups for time-series competitions is a research-adjacent habit that this course is the first step toward.

#### [Exercism, Python track](https://exercism.org/tracks/python)

_interactive · free · ongoing, a few a week · free practice with tests and human mentors_

**Age 5.** A website with little puzzles to solve in Python. Each one is a small job, like counting words, and when you finish, a real person can look at your answer and give you tips.

**Middle school.** Exercism is free and has more than a hundred Python exercises that grow from very easy to medium. You solve them on your own computer, run the tests they give you, and submit. Volunteers mentor you if you ask. Do two or three a week during phase 1; they turn reading about Python into writing it.

**High school.** Each exercise comes with a test file, so you practice the exact loop the project uses: read a description, write a function, run `pytest`, fix, repeat. The track's concept exercises teach one idea at a time (loops, dicts, string methods, classes). Aim for the exercises on parsing and on dates, which rhyme with the collector script.

**Stanford undergrad.** The value is deliberate practice with feedback: small problems with tests and human review are how skill compounds. Use it to practice idioms you meet in the pipeline code (comprehensions, generators, `dataclasses`), and read other people's published solutions afterward to see cleaner versions of yours.

**Stanford grad.** Compared with competitive-programming sites, Exercism's emphasis is on readable, idiomatic code, which is the skill a research codebase needs. Its mentor feedback is uneven but often excellent; the community solutions are a corpus of Python style worth studying.

**PhD at Apple.** For a team, a shared practice track is a low-cost way to align coding style among people from different backgrounds; the exercises are also a fair, low-stakes pair-programming warm-up for onboarding an intern.

#### [CS50's Introduction to Programming with Python (Harvard)](https://cs50.harvard.edu/python/)

_course · free · ten weeks part-time · free; structured lectures with checked problem sets_

**Age 5.** A famous free class where a teacher explains Python from the very beginning, with problems to solve after every lecture.

**Middle school.** Harvard's Python course, free online: functions, conditionals, loops, exceptions, libraries, unit tests, file input and output, regular expressions, and classes. Ten lectures with problem sets that are checked automatically. It is more structured than Sweigart and pairs well with it.

**High school.** Its problem sets are the best part: each is a small program with tests. The weeks on exceptions, on libraries, on unit tests and on file I/O map onto the collector script's needs. Do the problem sets in order; the automatic checker is strict in a useful way.

**Stanford undergrad.** The course teaches testing early (its unit-testing week is unusual for an intro course) and its problems are designed to be misread, which trains careful reading of requirements. Its final project is a chance to build something around the traffic data.

**Stanford grad.** A well-engineered course: autograders, consistent difficulty, and problems that scale. Its design is worth studying if you ever teach.

**PhD at Apple.** For a team, CS50P is a fair baseline for what 'can program in Python' should mean before someone touches pipeline code.

#### [The `zoneinfo` and `datetime` documentation](https://docs.python.org/3/library/zoneinfo.html)

_docs · free · an hour · free; read the section on aware versus naive datetimes twice_

**Age 5.** The instructions for the part of Python that knows about clocks, dates and the different times in different places.

**Middle school.** `datetime` handles dates and times; `zoneinfo` knows the time zones of the world. Read the section that explains aware versus naive times (aware means the time knows its zone), then the examples that convert a time from one zone to another. That is the whole trick behind storing every time in UTC.

**High school.** The key pages: `datetime.datetime` (aware versus naive, `astimezone`, `isoformat`, `fromisoformat`), `timedelta` for arithmetic, and `zoneinfo.ZoneInfo`. Read the aware-versus-naive section twice, because most time bugs are a naive time mistaken for a UTC one. Then learn the pandas equivalents, `tz_localize` and `tz_convert`, which apply the same ideas to a whole column at once.

**Stanford undergrad.** The docs explain `fold`, the attribute that distinguishes the first and second occurrence of the repeated hour when daylight saving ends. That is how Python represents the November ambiguity that pandas exposes as the `ambiguous` argument. Understanding fold makes the cleaning stage's choice to drop the ambiguous hour a deliberate decision rather than a default.

**Stanford grad.** Time zone rules come from the IANA database through the `tzdata` package and change when governments change rules, so pin the version. The docs' note that naive datetimes are interpreted as local time is a trap on servers set to UTC, and `datetime.utcnow()` returning a naive object is a second trap that the docs now warn about. Both have caused real outages.

**PhD at Apple.** Time handling is a recurring production failure class across the industry. The documentation is the specification; the engineering answer is a house rule (UTC everywhere, aware objects only, conversions at the edges) enforced by a linter rule and a test, so the specification never has to be reread under pressure.

### Build

- Read `02_collect_events.py` top to bottom and write one comment above each function saying what it does and why.
- Rewrite `event_touches_fremont()` from its description without looking, then test it with pytest on five hand-made events.
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

- Linear algebra: vectors, dot product, matrix times vector, matrix times matrix, transpose, norms; what `x @ weights + bias` computes for a whole batch.
- Calculus: the derivative as a slope, the chain rule, partial derivatives and the gradient; derivatives of the sigmoid and the log.
- Probability: random variables, Bernoulli and normal distributions, expectation, variance, conditional probability and Bayes; the logistic function as a probability.
- Statistics: mean, median and percentiles, sampling, the law of large numbers, confidence intervals, what a p-value is and is not, correlation versus causation.
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

#### [Khan Academy: Linear algebra, Differential calculus, Statistics and probability](https://www.khanacademy.org/math)

_course · free · weeks · free; do the exercises, skip nothing marked 'gradient'_

**Age 5.** Video lessons and practice questions that teach math one small step at a time, with a friendly voice, and you earn points for practicing.

**Middle school.** Three free courses, each made of short videos and exercises that get harder as you improve. Linear algebra is about vectors and grids of numbers, calculus is about slopes and how things change, and statistics is about averages, spread and chance. Do the exercises after every video; the videos alone will not stick.

**High school.** Use them selectively rather than front to back. In linear algebra: vectors, dot products, matrix multiplication. In differential calculus: derivatives, the chain rule, and then the multivariable course's unit on partial derivatives and the gradient. In statistics: distributions, the normal curve, sampling, and probability rules including conditional probability. Skip nothing with the word gradient in it; that is the heart of lesson 7.

**Stanford undergrad.** Khan is remedial in depth but excellent for gaps, and its mastery system is spaced repetition, which is why it works. Pair each unit with the corresponding chapter of Mathematics for Machine Learning to see the same idea at university level, and use Khan's diagnostics to find the specific skill that is rusty rather than redoing a whole subject.

**Stanford grad.** Its limitation is proof and abstraction: it teaches computation, not why the computation is right. For that, Mathematics for Machine Learning and 3Blue1Brown. It remains the fastest way to rebuild one specific skill, such as partial derivatives, in an evening before you need it.

**PhD at Apple.** For a research group, mastery-style practice is an underrated onboarding tool for people arriving from adjacent fields. Its per-skill diagnostics let you prescribe precisely (three units, one evening) instead of sending someone away for a semester.

#### [3Blue1Brown: Essence of linear algebra, Essence of calculus](https://www.3blue1brown.com/topics/linear-algebra)

_video · free · an afternoon each · free; the intuition the exercises lack_

**Age 5.** Videos with beautiful moving pictures that show what math looks like, so arrows and slopes become things you can see instead of just numbers.

**Middle school.** Two YouTube series by Grant Sanderson. The linear algebra one shows vectors as arrows and matrices as ways of stretching and turning space. The calculus one shows slopes and areas built from tiny pieces. Watch each video, then pause and redraw the picture yourself on paper; the drawing is where it sinks in.

**High school.** Watch the linear algebra series before matrix homework: it turns matrix multiplication into composing transformations, the determinant into a scaling factor, and eigenvectors into directions that do not turn. The calculus series makes the chain rule and the derivative of e to the x intuitive rather than memorized. Neither replaces practice; both make practice make sense.

**Stanford undergrad.** The series supply the geometric intuition textbooks omit: the dot product as projection, the derivative as a local linear approximation, change of basis as a change of viewpoint. That intuition is what you will lean on when reading about singular value decomposition, principal components and Jacobians, and when debugging a shape error in a matrix product.

**Stanford grad.** Sanderson's method, visual first and symbols second, is a model for explaining your own work. His later videos on neural networks and on the Fourier transform extend the same approach to topics you meet in phases 6 and 8, and his video on backpropagation is the one to show someone who has read the equations and still does not see it.

**PhD at Apple.** The channel is evidence that rigorous intuition scales to millions of viewers. For someone who will teach and present, the animation library he built (manim) is also a practical tool for figures in talks, and his explanations of attention are a good benchmark for how clearly a research idea can be communicated.

#### [Immersive Linear Algebra (Ström, Åström, Akenine-Möller)](https://immersivemath.com/ila/)

_interactive · free · a few evenings · free, interactive; touch what the videos show_

**Age 5.** An online book where the pictures move. You drag arrows around and the math updates, so you can feel what adding arrows and stretching space means.

**Middle school.** A free web book with interactive figures in every chapter: vectors, dot products, matrices, determinants, and what a matrix does to a picture. Play with each figure until the text makes sense. It pairs well with the 3Blue1Brown videos; the videos show you, this lets you touch.

**High school.** Work through chapters 2 to 6 (vectors, dot product, vector product, matrices, determinants) alongside Khan Academy exercises. The interactive matrix figures make the row-times-column rule and the idea of a transformation concrete, which is what you need to read `X @ w` and the neighbor-mixing matrices without hesitation.

**Stanford undergrad.** The interactive treatment of change of basis and of the determinant as a volume factor supports the geometric view of eigenvectors and of the singular value decomposition. Use it as the intuition layer under Mathematics for Machine Learning's chapter 2 to 4.

**Stanford grad.** It stops before numerical linear algebra and before spectral graph theory; its value is the mental imagery. For the graph operators in lesson 10 you will need the algebraic view as well, but a picture of what a matrix does to a set of points is what you will actually use when a shape bug appears.

**PhD at Apple.** A good example of what interactive explanation can do for a technical topic; if you write explanatory material for a team, this and Distill are the reference points for the medium.

#### [Seeing Theory (Brown University)](https://seeing-theory.brown.edu/)

_interactive · free · an evening · free, interactive; probability you can see_

**Age 5.** A website that shows chance with little animations: you flip coins, roll dice and watch the results pile up into shapes.

**Middle school.** Free, in the browser, with six chapters: basic probability, compound probability, distributions, frequentist inference, Bayesian inference and regression. Each idea has an animation you control. Spend an evening on it before Think Stats and the distribution shapes will already be familiar.

**High school.** Its chapters on the central limit theorem, confidence intervals and Bayesian inference give the pictures behind lesson 6's bootstrap and lesson 11's calibration. The regression chapter animates least squares, which is the ancestor of every loss you will minimize.

**Stanford undergrad.** Use it as a visual glossary while reading Blitzstein or Mathematics for Machine Learning chapter 6: when a definition is opaque, find its animation. It does not replace derivations, but it shortens the time between reading a definition and believing it.

**Stanford grad.** Its coverage is introductory by design; its role for you is teaching, not learning. When explaining calibration or intervals to a non-specialist, its animations are the fastest bridge.

**PhD at Apple.** Another instance of explanation as interactive design; worth knowing when you brief a product team on what a confidence interval is not.

#### [Introduction to Probability (Blitzstein, Hwang) and Stat 110](https://probabilitybook.net/)

_book · free · a semester · free book and lectures; the rigorous first course_

**Age 5.** A book and a set of video classes about chance, taught by a professor who explains coin flips and dice with stories and jokes.

**Middle school.** Joe Blitzstein's Harvard course Stat 110 is free on YouTube, and the book with Jessica Hwang is free as a PDF. It covers probability from counting through random variables, expectation, conditional probability and the famous distributions, with lots of worked problems. It goes slower and deeper than Khan Academy.

**High school.** The chapters on conditional probability and Bayes' rule, on expectation and variance, and on the normal and Bernoulli distributions are the ones the project uses. The book's habit of checking answers by simulation in R (translate to Python) is exactly the computational stance of Think Stats. The strategic practice problems are excellent.

**Stanford undergrad.** This is the standard first rigorous probability course: it is careful about what a random variable is, why conditioning is the heart of probability, and how the named distributions relate (the 'distribution family tree'). Its treatment of the law of large numbers and the central limit theorem makes the bootstrap and confidence intervals of lesson 6 intelligible.

**Stanford grad.** Blitzstein's stories (the 'fundamental bridge' between probability and expectation, symmetry arguments, indicator random variables) are problem-solving tools that reappear in analysis of algorithms and in variance calculations for estimators. The later chapters on Markov chains are the background for lesson 10's state-space view and for reinforcement learning.

**PhD at Apple.** Stat 110 is a model of pedagogy for a technical subject; its problem sets are the reference bank for probability interviews and qualifying exams alike. For research that leans on concentration inequalities and dependence, this is the base layer beneath the measure-theoretic course you take later.

#### [Mathematics for Machine Learning (Deisenroth, Faisal, Ong), chapters 2–7](https://mml-book.github.io/)

_book · free · reference · free PDF; reference, not a read-through_

**Age 5.** A big free book about the math inside learning machines. It is for looking things up, not for reading from the first page to the last.

**Middle school.** A university textbook, free as a PDF. Chapters 2 to 7 are linear algebra, geometry, matrix decompositions, vector calculus, probability, and optimization. When a lesson mentions a word you do not know, find it in the index and read that section. The pictures and worked examples are good; the exercises are hard and worth trying.

**High school.** Use it as the reference after Khan Academy and 3Blue1Brown. Chapter 5 on vector calculus explains gradients and the chain rule in the form machine learning uses; chapter 6 covers distributions, expectations and Bayes; chapter 7 explains gradient descent and convexity. Do the exercises for chapters 5 and 7; they are the two chapters that lesson 7 and lesson 9 lean on.

**Stanford undergrad.** The book's second half applies the math: work chapter 9 on linear regression fully to see maximum likelihood, maximum a posteriori and Bayesian regression in one place, then chapter 10 on principal components for the decomposition view. Its notation matches most ML papers, which makes it the bridge from courses to the literature.

**Stanford grad.** Coverage stops before measure-theoretic probability and before optimization theory proper (convergence rates, duality in depth). For those, Boyd and Vandenberghe on convex optimization and a graduate probability text. It is the right prerequisite for Murphy's Probabilistic Machine Learning, and its exercises are a reasonable self-test before that book.

**PhD at Apple.** As a reference for writing, it fixes consistent notation for gradients, Jacobians and distributions that reviewers recognize. Its free availability makes it the safe citation for prerequisites in a paper or an internal design document, and its chapter on optimization is a compact refresher before reading anything on second-order methods.

#### [Think Stats (Downey)](https://allendowney.github.io/ThinkStats/)

_book · free · two weeks · free; statistics with Python from the first page_

**Age 5.** A free book that teaches statistics by playing with real numbers in Python, instead of only formulas.

**Middle school.** Allen Downey's book uses real data and Python code from the first chapter: distributions, averages, spread, probability, and how to tell whether a difference is real or just chance. Free online. Run the code as you read; each chapter's exercises use the same data, so you build up a small toolkit.

**High school.** The approach is computational: build a distribution from data, compute a percentile, run a simulation instead of memorizing a formula. That is exactly how the profile (a median per slot) and the bootstrap (resampling days) work in this project. The chapters on distributions, probability, estimation and hypothesis testing are the core; the one on correlation is a good antidote to reading too much into a scatter plot.

**Stanford undergrad.** Downey's simulation-first stance, resampling for confidence intervals and permutation tests for differences, is the modern practical approach and maps directly onto the block bootstrap for average-precision intervals. The treatment is light on theory; pair with chapter 6 of Mathematics for Machine Learning for the formal side.

**Stanford grad.** Compare its resampling viewpoint with classical inference and understand when each is valid: the bootstrap fails for dependent data unless it resamples blocks, which is why lesson 6 bootstraps over days. Downey's Think Bayes is the natural next book for the uncertainty track in phase 8.

**PhD at Apple.** The book is an argument that computation has changed how statistics should be taught, and it is right. For someone designing evaluation methodology, its emphasis on effect sizes and simulation over p-values is the correct instinct, and its examples are a template for the kind of small, honest analysis that should accompany every metric in a report.

#### [MIT 18.06 Linear Algebra (Strang), OpenCourseWare](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/)

_course · free · ten lectures, two weeks · free; the classic lectures, the first ten alongside Khan_

**Age 5.** Video lessons by a beloved professor who explains grids of numbers with a chalkboard and a lot of enthusiasm.

**Middle school.** Gilbert Strang's MIT course, free on OpenCourseWare with videos, notes and problem sets. It covers vectors, matrices, solving equations, the four fundamental subspaces, projections, determinants, eigenvalues and singular values, in a way that makes the ideas feel concrete. Watch the first ten lectures alongside Khan Academy.

**High school.** Lectures 1 to 4 (vectors, elimination, matrix multiplication), 14 to 16 (projections and least squares), and 21 to 22 (eigenvalues) are the ones the course needs. Strang's picture of the four subspaces is the mental model behind least squares, which is the ancestor of every loss function you will minimize.

**Stanford undergrad.** Strang teaches through geometry and examples rather than proofs; his treatment of least squares, of positive definite matrices and of the singular value decomposition is exactly the linear algebra of regression, of the Hessian's convexity argument, and of principal components.

**Stanford grad.** The later lectures on positive definiteness, similar matrices and the SVD connect to optimization and to spectral graph methods; his companion course on matrix methods for data analysis continues into machine learning directly.

**PhD at Apple.** A canonical course; for research communication, Strang's habit of stating the idea in one sentence before any symbol is the standard to aim for.

#### [OpenIntro Statistics](https://www.openintro.org/book/os/)

_book · free · weeks · free textbook; inference done carefully_

**Age 5.** A free statistics textbook with lots of real examples, written so that anyone can follow it.

**Middle school.** A free, openly licensed introductory statistics book with videos and exercises: data collection, summarizing data, probability, distributions, inference with confidence intervals and hypothesis tests, and regression. Slower and more thorough than Khan Academy on inference.

**High school.** The chapters on probability, on distributions, and on inference (sampling distributions, confidence intervals, hypothesis tests and their misuse) give the statistical grounding that lesson 6 assumes. Its regression chapters preview the regression topic.

**Stanford undergrad.** OpenIntro is careful about experimental design and about what a p-value is, which most machine-learning material glosses. Read it for the inference vocabulary and for its treatment of confounding, then Blitzstein for the probability depth.

**Stanford grad.** It is introductory by design; its value is a correct foundation for the language of uncertainty that the evaluation and causal-inference topics build on.

**PhD at Apple.** An openly licensed textbook maintained by a nonprofit, usable in teaching without restrictions; its treatment of study design is what you will point a product team to when they propose an evaluation.

### Build

- Derive the logistic regression gradient on paper, then check it numerically with finite differences against the repo's update.
- Write the training loop from a blank file and match the repo's weights to four decimals.
- Compute the median and the 85th percentile of a real speed column with numpy, by hand and with the library, and write a notebook page titled 'why the median' for stage 6.

### Done when

- Your loop reproduces `bottleneck_weights.npz` and you can say what each weight means.
- You can derive the gradient on a whiteboard without notes.

### Pitfalls

- Trying to finish all of math first. This phase runs alongside phases 1 and 3 on purpose.
- Memorizing formulas you have not derived once.

## Phase 3: Data handling, files and SQL

_weeks 8–15 · ≈60 h · builds stage 5 for real_

**Goal.** Build stage 5, the cleaning stage, on a real PeMS file inside 2 GB, and be as fluent in SQL as in pandas.

### What you need to learn

- numpy: arrays, dtypes, broadcasting, memory layout; why float32 halves everything.
- pandas: Series and DataFrame, indexing, `groupby`, `merge`, time indexes, `shift`, `rolling`, `resample`, categoricals, `read_csv` with `chunksize`, Parquet through pyarrow.
- Time zones and daylight saving in practice; data quality: missingness, bounds, duplicates, gaps, PeMS's `% Observed`.
- SQL: `SELECT`, `WHERE`, `GROUP BY`, `JOIN`, window functions; DuckDB over Parquet; when a database beats files.
- Plotting with matplotlib; reading a plot before trusting a number. Measuring memory and time; polars as a second tool.
- The domain data sources: PeMS, 511, Open-Meteo, and what each actually measures.

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

### Topic: SQL and databases: where data usually lives

**Goal.** Ask questions of a database in SQL, know when a database beats files, and never be stuck because the data is not a CSV.

**Age 5.**

Most numbers in the world live in databases, which are like giant well-organized filing cabinets.

SQL is the way you ask the cabinet questions: 'show me every crash on I-880 in March, biggest first.' The cabinet answers with a table.

_Try this._ Think of a question you could ask about the incident list, like 'which road had the most crashes?' Say it as a sentence starting with 'show me'.

**Middle school.**

A database stores tables (rows and columns) and lets many programs read and write them safely. SQL is the language for questions: `SELECT` picks columns, `WHERE` filters rows, `GROUP BY` counts or averages per group, `ORDER BY` sorts, and `JOIN` combines two tables that share a key, like events and stations.

SQLite is a database in a single file with no server; DuckDB is one built for analysis and can query Parquet files directly. Both come free with Python.

_Try this._ Load `fremont_events_log.csv` into DuckDB and run: `SELECT road_names, COUNT(*) AS n FROM events GROUP BY road_names ORDER BY n DESC LIMIT 5`.

**High school.**

The core of SQL in one afternoon: `SELECT`, `WHERE`, `GROUP BY` with aggregates, `HAVING`, `ORDER BY`, `LIMIT`, inner and left `JOIN`, subqueries, and `CASE`. Window functions (`LAG`, `AVG() OVER`) compute lags and rolling means inside the database, which is stage 8 in another language.

Schemas and keys: a primary key identifies a row, a foreign key links tables, an index makes lookups fast. Types matter: store timestamps as timestamps, not text.

When to use what: files (Parquet) for a pipeline that rebuilds from raw; a database when several programs share live data, need transactions, or need fast lookups by key.

_Try this._ Rebuild the lag-3 speed feature with a window function in DuckDB over one month of Parquet and check it equals the pandas version row for row.

**Stanford undergrad.**

The relational model: relations, keys, normalization (and when to denormalize for analytics), the difference between OLTP (many small transactions) and OLAP (large scans and aggregates), and why columnar engines such as DuckDB dominate the second.

Query execution: how a planner turns SQL into scans, joins and aggregates, why indexes help lookups but not scans, and how to read an `EXPLAIN` plan. Transactions and ACID guarantees; concurrency and isolation levels in one paragraph.

From Python: `duckdb.sql()` over DataFrames and Parquet, `sqlite3` in the standard library, SQLAlchemy for a server database. A feature pipeline can be written entirely in SQL over Parquet, which is often faster and easier to audit than pandas.

_Try this._ Write the profile computation (median speed per station, slot and day type on training rows) as one SQL query in DuckDB and compare its runtime and memory with the pandas version.

**Stanford grad.**

Data modelling for ML: point-in-time correctness expressed as `AS OF` joins, slowly changing dimensions for station metadata that changes when detectors are replaced, and event tables versus state tables. Lakehouse formats (Parquet with a table layer) as the current default for analytical data.

Query optimization matters at scale: predicate pushdown, partition pruning, join ordering. The same pipeline can run on DuckDB on one machine and on a distributed engine (Spark, BigQuery) with mostly the same SQL, which is the argument for writing transformations in SQL.

_Try this._ Design a schema for the pipeline's data (raw, cleaned, features, predictions, monitoring) with keys and partitioning, and write the `AS OF` join that attaches the latest station metadata valid at each timestamp.

**PhD at Apple.**

At scale, data platforms are the substrate of ML: streaming ingestion, table formats with time travel, feature stores backed by both an offline warehouse and an online key-value store, and governance (lineage, access control, retention). The hard problems are consistency between offline and online views and reproducibility of training sets from a warehouse that changes.

Research and engineering overlap in query engines for ML (vectorized execution, GPU databases) and in declarative feature definitions compiled to both batch and streaming.

_Try this._ Specify how the training table for one model version could be reconstructed bit for bit from the warehouse a year later, and what the warehouse must guarantee for that to be possible.

**Check yourself.**

<details>
<summary>When does a database beat Parquet files?</summary>

When several programs share live data, when you need transactions, or when you need fast lookups by key. A rebuild-from-raw pipeline is happier with files.

</details>

<details>
<summary>What does a window function give you that GROUP BY does not?</summary>

A value per row computed over neighboring rows, such as the previous value or a rolling average, without collapsing the rows.

</details>

### Read and watch, explained at every level

#### [Python for Data Analysis, 3rd edition (McKinney), chapters 4–8 and 10–11](https://wesmckinney.com/book/)

_book · free · weeks · free online; the pandas book by pandas's author_

**Age 5.** A book by the person who invented pandas, the tool that turns big tables of numbers into things you can sort, count and draw.

**Middle school.** Wes McKinney created pandas and wrote this book about it. Chapters 4 to 8 teach numpy and pandas basics: arrays, tables, reading files, cleaning messy values, and combining tables. Chapters 10 and 11 teach grouping and time series, which this project uses everywhere. It is free online; do the examples with the project's own data instead of the book's.

**High school.** The essential chapters for stages 5 and 8: 5 (pandas basics), 7 (cleaning: missing data, duplicates, transformations), 8 (merging and reshaping), 10 (groupby, the split-apply-combine that builds the speed profile), and 11 (time series: resampling, shifting, rolling windows, time zones). Do chapter 11 twice; every lag, rolling mean and DST fix in this pipeline comes from it.

**Stanford undergrad.** The book explains pandas' model (index alignment, dtypes, categoricals) so you can predict behavior instead of guessing. Understanding index alignment prevents the silent misjoins that plague feature engineering, and the numpy chapter's broadcasting and vectorization are the reason loops are avoided. Its appendix on advanced numpy is worth reading before lesson 9's hand-written backward pass.

**Stanford grad.** Its limits are scale and the newest tooling: little on Arrow-backed dtypes, nothing on polars or out-of-core processing. Read the pandas documentation on Copy-on-Write and Arrow strings, and the polars guide, to see where the ecosystem is heading. The book's groupby chapter remains the clearest explanation of split-apply-combine in print.

**PhD at Apple.** pandas' design decisions (label alignment, mutable frames, Python-level groupby) explain its performance ceiling and why Arrow-native tools win the cleaning stage while pandas keeps the analysis stage. McKinney's later work on Apache Arrow is the continuation of the story, and understanding both is what lets you choose a tool per stage rather than per habit.

#### [pandas user guide: Time series and date functionality](https://pandas.pydata.org/docs/user_guide/timeseries.html)

_docs · free · an afternoon · free; the `shift` and `rolling` semantics live here_

**Age 5.** The manual page for the part of pandas that understands dates, times and clocks.

**Middle school.** One long page in the pandas documentation. It shows how to make a range of 5-minute timestamps, shift values forward and back, compute rolling averages, and convert between time zones. Read it with a notebook open and try each example on one station's data from the project.

**High school.** The sections you need: date ranges and frequency strings (`freq='5min'`), shifting, `rolling` windows, `resample`, and the time-zone section covering `tz_localize`, `tz_convert`, and the `ambiguous` and `nonexistent` arguments. This page is where the daylight-saving behavior in stage 5 is specified, so read that section with the November example in front of you.

**Stanford undergrad.** Understand the difference between a positional shift and a time-based shift, and why the full-grid reindex in stage 5 makes them agree. Also the exact semantics of `rolling`: windows are right-aligned by default, `center=True` looks ahead and is therefore forbidden for features, and `min_periods` controls when the window starts producing values.

**Stanford grad.** The guide documents edge behaviors (periods versus timestamps, offsets, business-day calendars) that matter for holiday and weekday features, and the performance notes on tz-aware columns. Read the Copy-on-Write page alongside; it changed chained-assignment semantics and quietly breaks old code.

**PhD at Apple.** The precise semantics of window operations are what point-in-time correctness rests on. For an engineering organization the right move is to encode them once in a feature library with tests rather than rereading the guide per project, and to treat any window parameter as part of the model's versioned configuration.

#### [Kaggle Learn: Pandas, Data Cleaning, Data Visualization](https://www.kaggle.com/learn/pandas)

_course · free · an afternoon each · free, short_

**Age 5.** Three short online lessons about sorting tables, tidying up messy numbers, and drawing charts from them.

**Middle school.** Browser lessons with instant grading. Pandas: selecting, grouping and summarizing tables. Data Cleaning: missing values, scaling, parsing dates, fixing inconsistent text. Data Visualization: line charts, bar charts and histograms with the seaborn library. About an afternoon each; do them between chapters of McKinney.

**High school.** They rehearse exactly the operations of stage 5 (missing data, parsing dates, deduplication) and the plots you will make of a station's day and week. Their datasets are small and tidy compared with a PeMS file, so treat them as syntax practice and then confront the real 200 MB file, where the same operations need chunking and care.

**Stanford undergrad.** Use them for the mechanics and then go to the source: the matplotlib documentation for control over figures, and a book such as Fundamentals of Data Visualization (Wilke, free online) for how to design a plot that communicates. Learn to read a plot before trusting a metric; the fundamental-diagram scatter and the daily speed line should become habits.

**Stanford grad.** The visualization course stops at defaults. The skills that matter for this project are diagnostic plots: reliability diagrams, lead-time histograms, per-station error maps, and autocorrelation plots. Build a small plotting module for the project so every evaluation produces the same figures.

**PhD at Apple.** Visualization is a research instrument, not decoration. A reliability diagram or a lead-time distribution communicates what a table of metrics hides, and the habit these courses seed, plotting before concluding, is worth more than their content. Wilke's book and Tufte remain the references for the design side.

#### [The polars user guide, first three chapters](https://docs.pola.rs/)

_docs · free · an afternoon · free; do stage 5 once in each_

**Age 5.** The manual for a newer, faster tool that does the same job as pandas: turning big tables into answers.

**Middle school.** Polars is a table library written in the Rust language. It is often several times faster than pandas and uses less memory, which matters on a 16 GB machine. The first chapters cover expressions (how you describe what you want), selecting and filtering rows, and grouping. Do stage 5 once in pandas and once in polars, and time both.

**High school.** Its model differs from pandas: expressions instead of index alignment, lazy evaluation where `scan_csv` plans the whole query and runs it once, strict dtypes, and no index. For streaming a district-day file and keeping a hundred stations, the lazy scan with a filter is exactly right and runs in a few seconds where pandas takes twenty.

**Stanford undergrad.** Learn the lazy API and streaming collection, and understand why immutability and the absence of an index make parallel execution straightforward. Interoperation goes through Apache Arrow, so `to_pandas` and `from_pandas` are cheap or zero-copy for most dtypes, which lets you use polars for the heavy stage and pandas for analysis without friction.

**Stanford grad.** Polars represents the Arrow-native, query-optimized direction of data tooling, alongside DuckDB. For pipelines it is the better default; for ad hoc analysis pandas still has the larger ecosystem. Know both and choose per stage; the guide's chapter on the query optimizer explains why the same operation costs so differently.

**PhD at Apple.** The performance gap is architectural (columnar memory, SIMD, multithreading, query optimization) rather than incidental, and on unified-memory hardware both libraries end up limited by memory bandwidth. Benchmark on the real file before choosing, and note that the decision is per stage: the cleaning stage is a query, the modelling stage is not.

#### [SQLBolt, interactive SQL lessons](https://sqlbolt.com/)

_interactive · free · a weekend · free; SQL fluency in a weekend_

**Age 5.** A website where you type questions to a pretend filing cabinet and it answers right away, with hints when you get stuck.

**Middle school.** Free, in the browser, about twenty short lessons: selecting rows, filtering, sorting, joining two tables, counting and averaging, and changing data. Each lesson has a few exercises against a tiny database. Two evenings and you can read most SQL.

**High school.** Work through lessons 1 to 12 (queries, filters, joins, aggregates) and 13 to 18 (inserting, updating, creating tables). Then repeat the aggregate lessons in DuckDB on the incident log, where the questions are real: incidents per road per hour, longest-lasting events, roads with the most unknown-severity events.

**Stanford undergrad.** It teaches syntax, not the relational model; pair it with a page on normalization and keys, and with `EXPLAIN` in DuckDB to see what a query does. Its exercises are single-table or two-table; practice multi-join and window-function queries on the project's data afterward.

**Stanford grad.** Sufficient for reading analytics SQL; insufficient for schema design, transactions and performance, which come from a database textbook or the documentation of the engine you use. Its value is speed: fluency in a weekend.

**PhD at Apple.** For a team, a shared minimum of SQL fluency is what makes feature definitions reviewable by analysts and engineers alike; an interactive tutorial is the cheapest way to reach it.

#### [DuckDB documentation, the Python API and SQL reference](https://duckdb.org/docs/)

_docs · free · an afternoon · free; SQL over the project's Parquet files_

**Age 5.** The manual for a small, fast database that lives inside your program and can read the project's files directly.

**Middle school.** DuckDB is a free analytical database that runs inside Python with no server. Its documentation shows how to query CSV and Parquet files with SQL in one line, how to hand results to pandas, and the full list of SQL functions including window functions for lags and rolling averages.

**High school.** Read the Python API page (`duckdb.sql`, `read_parquet`, `to_df`), the window-functions page, and the page on the `ASOF` join, which attaches the latest weather or metadata valid at each timestamp. Then rewrite one feature (lag-3 speed) as a window function and check it against pandas.

**Stanford undergrad.** DuckDB is a vectorized, columnar, single-node engine with a query optimizer; its documentation on the execution model and on out-of-core processing explains why it handles a district-day file in seconds within a few hundred megabytes. Its Parquet reader does predicate pushdown and partition pruning, which is what stage 5 needs.

**Stanford grad.** The engine is the local counterpart of cloud warehouses; SQL written for it moves to them with small changes. Its extensions (spatial, JSON, httpfs) let the whole pipeline be expressed as SQL over files, which is easier to audit and version than pandas code.

**PhD at Apple.** Embedded analytical engines changed what one machine can do; their documentation is also a compact account of modern query execution. For a lead, DuckDB is the argument that the cleaning and feature stages should be declarative.

#### [PeMS documentation and the Clearinghouse (Caltrans)](https://pems.dot.ca.gov/)

_docs · free (login) · an afternoon · the source of the sensor data; read before parsing_

**Age 5.** The website where the road sensor numbers come from, with pages that explain what each column of numbers means.

**Middle school.** PeMS is Caltrans's system that collects data from the freeway sensors. After you log in, the Clearinghouse page lets you download one file per day for a district, and the help pages explain the columns: timestamp, station, flow, occupancy, speed, and how many of the samples were real. Read the help page for the 5-minute station data before touching a file.

**High school.** The documentation describes the `station_5min` layout (no header row; timestamp, station id, district, freeway, direction, lane type, station length, samples, percent observed, total flow, average occupancy, average speed, then per-lane columns) and the meta file that gives each station's location and postmile. It also explains that PeMS imputes missing data and how the percent-observed column reports it. Confirm every claim against a real file; the documentation has drifted from the files before.

**Stanford undergrad.** Understand the data provenance: loop detectors report raw 30-second counts and occupancy, the district servers aggregate to 5 minutes, speed is often estimated for single-loop stations, and the imputation algorithm fills gaps from neighbors and history. That chain determines the error model in lesson 2 and the cleaning rules in lesson 3.

**Stanford grad.** PeMS's imputation and its detector-health diagnostics are documented in Caltrans reports and in the traffic-engineering literature (Chen, Kwon, Rice, Skabardonis and Varaiya's work at Berkeley). Reading those explains what percent observed means statistically and why imputed rows correlate with neighbors.

**PhD at Apple.** PeMS is also the source of the public benchmark datasets (PEMS-BAY, PeMSD4, PeMSD8) used across the spatio-temporal literature, which makes it a rare case where your production data and the research benchmarks share a lineage. The rate limits and login are the practical constraint; a data-sharing agreement or a bulk export is what a real deployment would negotiate.

#### [Open-Meteo, historical weather API documentation](https://open-meteo.com/en/docs/historical-weather-api)

_docs · free · an hour · free, no key; the rain feature_

**Age 5.** A free website that tells you what the weather was, hour by hour, for any place, so we can know whether it rained when the road got slow.

**Middle school.** Open-Meteo is a free weather API with no sign-up. Its historical endpoint returns hourly values (rain, temperature, wind, visibility) for a latitude and longitude going back decades, and its forecast endpoint returns the next days. The documentation page builds the request URL for you as you tick boxes.

**High school.** For stage 8, request hourly precipitation for Fremont's coordinates over the training period and join it on the UTC hour. For the live loop, use the forecast endpoint, because at prediction time only a forecast exists. The docs list every variable and its units; note that historical data come from reanalysis models, not from a rain gauge in Fremont.

**Stanford undergrad.** Understand the difference between reanalysis (a model's reconstruction of past weather from observations) and forecasts, and why training on reanalysis while serving on forecasts is a mild train-serve skew that you should measure (compare forecast and reanalysis precipitation on the same hours).

**Stanford grad.** The point-in-time question from lesson 5 applies: the correct training feature is the forecast that would have been available at t, which Open-Meteo also archives (previous-runs endpoints). Whether the extra fidelity matters for a binary rain feature is an experiment worth an afternoon.

**PhD at Apple.** Weather is the cheapest exogenous signal with a real effect on traffic. In a production system the questions become licensing, latency, and fallbacks when the provider is down; Open-Meteo's open model is a useful default and its terms are worth reading before shipping.

#### [511 SF Bay Open Data, traffic API documentation](https://511.org/open-data)

_docs · free (token) · an hour · free with a token; the incident feed's fields and limits_

**Age 5.** The rules for asking the 511 helper about crashes and roadwork, so our program can ask politely and understand the answer.

**Middle school.** 511.org publishes traffic events (incidents, roadwork, closures) through an API that follows a standard called Open511. You register for a free key, then request events as JSON. The documentation lists the fields (id, type, severity, roads, geography, created, updated) and the limits on how often you may ask.

**High school.** Read the events endpoint documentation next to `02_collect_events.py`: which query parameters exist (`status`, bounding box), what each field means, and the request quota. Note which fields are optional; missing geography and unknown severity are normal, which is why lesson 7's join handles them.

**Stanford undergrad.** Open511 is an open specification for road events; reading the spec explains the field semantics that the 511 docs assume, including how `created` and `updated` relate to an event's life and how multiple roads are represented. The quota defines the sampling interval of your incident stream and therefore its censoring.

**Stanford grad.** Event feeds are the interface between human reporting (CHP dispatch) and your model; their delay and severity encoding are measurement properties you should estimate rather than assume. Compare the feed with CHP's own incident page for a day to see what the API drops.

**PhD at Apple.** Public-sector data APIs are governed by terms and by quotas that shape what a product can promise; at scale, direct feeds from agencies or probe-based incident detection replace polling. The Open511 standard is also a small lesson in how interoperability specifications succeed or fail.

#### [Python Data Science Handbook (VanderPlas)](https://jakevdp.github.io/PythonDataScienceHandbook/)

_book · free · weeks · free online; a second explanation of numpy, pandas and matplotlib_

**Age 5.** A free book that shows how to use Python to look at data: sorting it, summarizing it, and drawing pictures of it.

**Middle school.** Jake VanderPlas's book, free online as notebooks: IPython, numpy, pandas, matplotlib and an introduction to scikit-learn. The numpy and pandas chapters are a good second explanation next to McKinney; the matplotlib chapter is the clearest short guide to plotting in Python.

**High school.** Read the numpy chapter for broadcasting and vectorized operations, the pandas chapter for grouping and time series, and the matplotlib chapter before making the station plots. Its scikit-learn chapter is a preview of phase 4 with clear examples of the fit-predict pattern.

**Stanford undergrad.** Its explanations of numpy's memory model and of pandas' index alignment complement McKinney's; the visualization chapter's coverage of matplotlib's object model is what you need to make consistent figures for the evaluation report.

**Stanford grad.** The book predates Arrow-backed pandas and polars; its scikit-learn chapter is introductory. Its lasting value is the clear treatment of the numerical stack's fundamentals.

**PhD at Apple.** A widely used free reference; a good default recommendation for engineers moving from another language to Python's data stack.

#### [Fundamentals of Data Visualization (Wilke)](https://clauswilke.com/dataviz/)

_book · free · weeks · free online; how to design a plot that tells the truth_

**Age 5.** A free book about drawing pictures of numbers so that people understand them, with lots of examples of bad pictures made good.

**Middle school.** Claus Wilke's book, free online, is about the design of plots rather than any one tool: which chart for which question, color, axes, legends, and the mistakes that mislead. Read the chapters on visualizing amounts, distributions and time series, then look at your own plots differently.

**High school.** Its chapters on visualizing uncertainty, on time series, and on the principle of proportional ink apply directly to the evaluation report: reliability diagrams, lead-time histograms and daily speed lines. The chapter on color explains why a sequential scale is right for one thing and wrong for another.

**Stanford undergrad.** Wilke's rules (avoid unnecessary 3-D, order categories meaningfully, show uncertainty honestly) are the review checklist for every figure you publish; its discussion of small multiples is how to show a hundred stations at once.

**Stanford grad.** Pair it with the literature on perception (Cleveland and McGill) that it draws on; its guidance on uncertainty visualization is the practical counterpart of the calibration and interval topics.

**PhD at Apple.** A free, rigorous, tool-agnostic reference for figures in papers and reports; the standard to hold a figure to before a talk.

#### [Kaggle Learn: Intro to SQL, Advanced SQL](https://www.kaggle.com/learn/intro-to-sql)

_course · free · an afternoon each · free; window functions on a real warehouse_

**Age 5.** Two short online lessons about asking questions of a giant table of data, with answers checked as you type.

**Middle school.** Kaggle's SQL courses run against real public datasets in BigQuery from the browser. Intro to SQL covers select, where, group by, order by, and joins; Advanced SQL covers joins in depth, analytic (window) functions and nested data. An afternoon each.

**High school.** Do Intro to SQL after SQLBolt and Advanced SQL's window-functions lesson before rewriting the lag feature in DuckDB. The window-functions lesson is the same idea as `shift` and `rolling` in pandas, expressed in SQL.

**Stanford undergrad.** The courses use a cloud warehouse, which shows what large-scale analytical SQL feels like (partitioning, cost of scans); the ideas transfer to DuckDB unchanged.

**Stanford grad.** Their coverage of nested and repeated fields previews the semi-structured data you meet in event streams such as the 511 feed.

**PhD at Apple.** A free, practical introduction to warehouse SQL for a team that needs analysts and engineers to share feature definitions.

#### [Select Star SQL](https://selectstarsql.com/)

_interactive · free · two hours · free; one dataset, project-shaped practice_

**Age 5.** A free online book that teaches you to ask a database questions by working through one real story from start to finish.

**Middle school.** An interactive SQL book, free and without sign-up, that uses one real dataset throughout and builds up to real questions. Four chapters, about half an hour each. Do it after SQLBolt for a project-shaped practice run.

**High school.** Its single-dataset design mirrors how you will work with the traffic data for months; the chapters on aggregation and on joins are the operations of the profile computation and the incident join. Its final chapter asks questions that need several steps, which is where SQL fluency is tested.

**Stanford undergrad.** The book is short and stops before window functions; pair with DuckDB's documentation for those. Its emphasis on reading the data before querying is the right habit.

**Stanford grad.** Useful mostly as a teaching resource; its narrative approach is a good model for a project-specific SQL primer.

**PhD at Apple.** A small, well-made open resource; the kind of thing to fork and adapt to a team's own dataset for onboarding.

### Build

- Run stage 4 for a few days of files. Open one, confirm the column layout, and write it into the stage 5 README.
- Write `10_clean_pems.py`: chunked reading, the Fremont filter, DST-safe UTC conversion, bounds, `% Observed`, forward fill, the full grid, float32, monthly Parquet.
- Build the station registry with upstream and downstream neighbors from postmiles.
- Rewrite the profile computation and one lag feature as DuckDB SQL and check them against pandas.
- Plot one station for a day and a week; write `tests/test_clean_pems.py` on a 100-row fixture including the November repeated hour.

### Done when

- One month of Fremont Parquet exists with the plan's schema and the plot shows rush hours.
- The DST test passes, the script stays under 2 GB, and the SQL version agrees row for row.

### Pitfalls

- Loading a whole district-day at once.
- `object` dtypes and naive timestamps.
- Forward-filling across a station boundary, which invents data.

## Phase 4: Classical machine learning, the core

_weeks 14–22 · ≈70 h · builds stages 6, 7, 9, 10_

**Goal.** Labels, the incident join, baselines, logistic regression, gradient boosted trees, calibration, and honest row-level evaluation.

### What you need to learn

- The supervised setup: features, target, train, validation, test, and why three. Overfitting, bias and variance, regularization, early stopping.
- Metrics: confusion matrix, precision, recall, F1, the precision-recall curve and average precision, ROC-AUC, log loss, Brier score, reliability diagrams; why accuracy lies under imbalance.
- Leakage in its three forms: target, temporal, group. Class weighting.
- Linear and logistic regression with scikit-learn; decision trees; random forests; gradient boosting and what `num_leaves`, `learning_rate` and `min_data_in_leaf` do; categorical handling.
- Feature engineering as hypothesis-making; feature importance and SHAP values.
- Isotonic and Platt calibration; choosing a threshold from a curve.

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

#### [Machine Learning Specialization (Andrew Ng), courses 1 and 2](https://www.coursera.org/specializations/machine-learning-introduction)

_course · free (audit) · six weeks · the classic; audit for free_

**Age 5.** Video classes by a famous teacher who explains how computers learn, slowly and kindly, with drawings on a screen.

**Middle school.** A set of online courses on Coursera that you can watch for free. Course 1 covers linear and logistic regression and gradient descent, the same ideas as lesson 7. Course 2 covers neural networks and decision trees, plus practical advice on what to do when a model does badly. Do the labs; they are small Python notebooks where you fill in the missing pieces.

**High school.** Course 1 is lesson 7's math in video form with numpy labs, and its explanation of the cost function's shape and of feature scaling is the clearest available. Course 2's sections on evaluation, bias versus variance, and error analysis are the practical heart of phase 4, and its decision-tree week is a preview of lesson 8. Watch at normal speed and pause to write the equations yourself.

**Stanford undergrad.** Ng's strength is the practitioner's loop: diagnose whether the problem is bias or variance, then act (more data, more features, more regularization, a different model), and analyze the errors by hand. The treatment is light on theory and on modern tooling, so pair it with ISLP for depth and with the LightGBM documentation for the tools you will actually use.

**Stanford grad.** The specialization's error-analysis method, look at the examples the model gets wrong, categorize them, and prioritize the largest category, is underused in research and is exactly how to improve the traffic model's worst stations: pull the episodes it missed, group them by cause (incident, rain, detector fault), and fix the largest group first.

**PhD at Apple.** Ng's courses shaped how a generation frames ML projects around iterative diagnosis, and his later data-centric argument, that data quality dominates model choice on most real problems, is the thesis of this entire pipeline. For a lead, the course is also the common vocabulary a mixed team can be assumed to share.

#### [An Introduction to Statistical Learning with Python (ISLP), chapters 2–4 and 8](https://www.statlearning.com/)

_book · free · three weeks · free PDF; the clearest account of trees and boosting_

**Age 5.** A famous textbook, free, that explains learning machines with pictures and real examples, written by teachers who invented some of the methods.

**Middle school.** Written by statisticians, with Python labs. Chapter 2 explains what learning from data is and why a model can be too simple or too complicated. Chapter 3 is linear regression, chapter 4 is classification including logistic regression, and chapter 8 is trees, forests and boosting. Read a chapter, then do its lab in a notebook.

**High school.** The clearest explanation anywhere of bias versus variance (chapter 2, with figures worth redrawing), of logistic regression as a model of probability (chapter 4), and of why boosting works by fitting small trees to what is left over (chapter 8). Its lab for chapter 8 uses the same scikit-learn calls you will make in stage 9.

**Stanford undergrad.** ISLP is the practical face of The Elements of Statistical Learning. Add chapter 5 on cross-validation and the bootstrap, and chapter 6 on model selection and regularization; together with the core chapters they cover phase 4's theory. Its treatment of the bootstrap is the basis for lesson 6's block bootstrap, with the modification that blocks are days.

**Stanford grad.** After ISLP, go to ESL for proofs and depth. ISLP's boosting section gives the additive-model view; combine it with Friedman's 2001 paper for the gradient view that LightGBM implements. Note where ISLP's examples assume independent rows and how each method's assumptions change under temporal dependence.

**PhD at Apple.** As a citation for standard methods and as a teaching text it is the reference, and its evolution from R to Python editions tracks the field's tooling. For research, its framing of methods as estimators with bias and variance you can reason about is the habit that distinguishes a careful ML engineer from a library user.

#### [Google's Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course)

_course · free · about 15 hours · free; the practitioner's course, videos and exercises_

**Age 5.** A free class from Google with short videos and little exercises that teach how computers learn from examples.

**Middle school.** Google's free course covers linear and logistic regression, classification metrics, overfitting and regularization, neural networks, embeddings, and fairness, with short videos, interactive exercises and quizzes. It covers what a practitioner's textbook would, in about fifteen hours.

**High school.** The modules on linear regression, logistic regression, classification (precision, recall, ROC, and thresholds), data, and overfitting map onto phase 4; the fairness module previews phase 10. Its interactive exercises on learning rate and on thresholds are worth doing twice.

**Stanford undergrad.** The course is practical and current, and its treatment of data quality, of feature crosses and of embeddings connects the classical and neural halves of the course. It is light on theory; ISLP supplies that.

**Stanford grad.** Its production-oriented sections (data preparation, fairness, real-world ML) reflect Google's own practice; read them with the Rules of ML.

**PhD at Apple.** A free, maintained, vendor-flavored but sound curriculum; useful as a shared baseline for a mixed team.

#### [scikit-learn: Getting Started and the tutorials](https://scikit-learn.org/stable/getting_started.html)

_docs · free · an afternoon · free; the workflow end to end in a few lines_

**Age 5.** The first pages of the manual for the learning toolbox, showing how to teach a model and ask it questions in a few lines.

**Middle school.** scikit-learn's Getting Started page and tutorials show the whole workflow in a few lines: load data, fit a model, predict, evaluate, and build a pipeline that scales and encodes features before the model. Read them the first week of phase 4 and run every snippet.

**High school.** Getting Started covers estimators, transformers, pipelines, model evaluation with cross-validation, and hyperparameter search; the tutorial on statistical learning goes deeper on supervised and unsupervised methods. Together with the user guide they replace the practitioner's book.

**Stanford undergrad.** The API design (fit, transform, predict; pipelines; consistent parameters) is the pattern every internal library ends up copying; the tutorials show it end to end and the examples gallery shows every method on real data.

**Stanford grad.** The examples gallery is an underused resource: each example is a small, correct experiment with a figure, and many are directly reusable for the evaluation report.

**PhD at Apple.** scikit-learn's documentation is a reference for how to document an API; its tutorials are a model of teaching by runnable example.

#### [R2D3: A Visual Introduction to Machine Learning](http://www.r2d3.us/visual-intro-to-machine-learning-part-1/)

_interactive · free · half an hour · free; trees and overfitting as an animated story_

**Age 5.** A web page that tells a story with moving pictures: a computer learns to tell apart homes in two cities by asking questions about them, one question at a time.

**Middle school.** Two free interactive essays that show, with animated charts, how a decision tree learns to classify by splitting data on features, and then what overfitting is and how bias and variance trade off. Fifteen minutes each; the pictures make the ideas stick.

**High school.** Part one is lesson 8's decision tree, animated: watch splits partition the data and the tree grow. Part two shows a tree that memorizes its training data and fails on new data, which is the overfitting of lesson 6, with the bias-variance trade-off made visible.

**Stanford undergrad.** Use it as the picture behind the formal bias-variance decomposition; its scrolling design is also a model for explaining a method to a non-technical audience.

**Stanford grad.** A teaching resource; when you have to explain overfitting to a stakeholder in five minutes, this is the page to open.

**PhD at Apple.** An example of explanation as design, useful as a reference for how far interactive storytelling can carry a technical concept.

#### [MLU-Explain (Amazon)](https://mlu-explain.github.io/)

_interactive · free · an hour per essay · free; visual essays on precision, ROC, cross-validation and more_

**Age 5.** A website with pictures that move to explain ideas in learning: how a machine measures its mistakes, how trees vote, and how to tell if it is right.

**Middle school.** Amazon's Machine Learning University publishes free visual explanations: linear and logistic regression, decision trees and random forests, precision and recall, ROC and AUC, cross-validation, bias and variance, the neural network's basics, and more. Each is a short interactive page. Read the one on precision and recall before lesson 6.

**High school.** The pages on ROC and AUC, on precision and recall, on cross-validation, on random forests, and on double descent give the pictures behind phases 4 and 5; the one on train, test and validation sets explains the split with animation.

**Stanford undergrad.** Their visual treatment of double descent and of the bias-variance trade-off is a useful complement to the theory topic; each page links the papers it summarizes.

**Stanford grad.** Good for teaching and for checking your own intuition; the pages are correct and current, and their sources are cited.

**PhD at Apple.** A curated set of explanations from an industry group; their choice of topics is a fair map of what practitioners most often misunderstand.

#### [StatQuest with Josh Starmer (YouTube)](https://statquest.org/)

_video · free · hours, as needed · free; one idea per video, worked with numbers_

**Age 5.** Videos with silly songs and simple drawings that explain how learning machines work, one small idea at a time.

**Middle school.** A YouTube channel where each video explains one idea, such as what a decision tree is or how logistic regression works, with drawings and a short song. Watch the videos on decision trees, random forests, gradient boosting and logistic regression during phase 4; they are slow, clear and free.

**High school.** StatQuest's series on gradient boosting (four videos) walks through fitting trees to residuals step by step with numbers, which is exactly lesson 8's mechanism. The videos on ROC and AUC, on cross-validation and on bias and variance give the pictures behind lesson 6. The logistic regression series covers maximum likelihood and the meaning of the coefficients.

**Stanford undergrad.** The channel's strength is tracing a computation with small numbers until it is obvious; use it to check your understanding of a derivation rather than as a source of the derivation. Its XGBoost series covers the second-order split gain, which is the LightGBM math in lesson 8's higher levels.

**Stanford grad.** It stops at intuition and worked examples, which is the right place for a video; the formal accounts are in ISLP and ESL. When you teach, its pattern (a concrete numeric example before any symbol) is worth stealing.

**PhD at Apple.** A reminder that clarity is a research skill; the channel's popularity among practitioners reflects a gap textbooks leave, and it is the resource to hand to a stakeholder who wants to know what a boosted tree actually does.

#### [Kaggle Learn: Intro to ML, Intermediate ML, Feature Engineering](https://www.kaggle.com/learn/intro-to-machine-learning)

_course · free · weeks · free_

**Age 5.** Three short online lessons: build your first learning machine, make it better, and invent better clues for it.

**Middle school.** Intro to ML builds a decision tree and then a random forest on a table of house prices. Intermediate ML covers missing values, categorical columns, pipelines, cross-validation and XGBoost, a cousin of LightGBM. Feature Engineering covers finding useful columns, creating new ones, and encoding categories. Quick, practical, and graded as you go.

**High school.** They give you working code for the exact tools of phase 4 (imputation, categorical handling, gradient boosting with early stopping) in an afternoon. Their cross-validation is shuffled, which is fine for house prices and wrong for traffic; keep lesson 6 in mind and do not copy that part.

**Stanford undergrad.** The Feature Engineering course's target-encoding lesson is a leakage hazard for time series: encode out-of-fold in time or skip it. Its mutual-information lesson is a useful quick screen for which features carry signal. Intermediate ML's pipeline lesson is the pattern to adopt for the feature module.

**Stanford grad.** The courses reflect Kaggle's tabular culture: gradient-boosted trees, engineered features, and careful validation. The competition write-ups they link to are the richer resource; reading three winning solutions for a time-series competition teaches more about leakage and validation than any course.

**PhD at Apple.** Kaggle's tabular consensus (boosted ensembles, careful validation, feature engineering over architecture) has been stable for a decade, and understanding why, which Grinsztajn and colleagues studied systematically, is more valuable than any single course and directly justifies LightGBM as this project's production model.

#### [LightGBM documentation: Parameters, Parameters Tuning](https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html)

_docs · free · an hour · free; read before touching a parameter_

**Age 5.** The instruction sheet for the tree-building program, with all of its knobs and what each one does.

**Middle school.** Two pages of the LightGBM documentation. The first lists every parameter with a one-line meaning; the second is a guide to which knobs to turn for speed, for accuracy, or to stop the model memorizing. Read the tuning page before you change anything, and change one thing at a time.

**High school.** Know these entries: `objective`, `num_leaves`, `min_data_in_leaf`, `learning_rate`, `feature_fraction`, `bagging_fraction` with `bagging_freq`, `lambda_l2`, `max_bin`, `is_unbalance`, `categorical_feature`, and early stopping. The tuning page explains leaf-wise growth and why `num_leaves` and `max_depth` interact, which is the single most common source of overfitting in LightGBM.

**Stanford undergrad.** Read the Features page too: the histogram algorithm, gradient-based one-side sampling, exclusive feature bundling, and native categorical handling explain where the speed and the memory footprint come from. The Python API page's `Dataset` semantics (`free_raw_data`, `reference` for consistent binning between train and validation) prevent subtle bugs.

**Stanford grad.** The documentation is terse on interactions among regularizers; the LightGBM paper (Ke et al., 2017) and the XGBoost paper (Chen and Guestrin, 2016) supply the algorithms and the second-order split gain. Monotone constraints and linear trees are documented options directly relevant here, and the parameter aliases are a known source of silent misconfiguration.

**PhD at Apple.** Documentation quality is a production concern. Parameter aliases have caused silent misconfigurations in real systems; the fix is a team configuration schema that validates parameters before training. At scale, also read the distributed-training and GPU pages to understand what changes when the data no longer fits one machine.

#### [The boosting papers: Friedman (2001), XGBoost (2016), LightGBM (2017)](https://arxiv.org/abs/1603.02754)

_paper · free · an evening or two · free; the algorithm behind every parameter_

**Age 5.** Three papers, written years apart, about a way of learning that builds many small decision trees, each fixing the mistakes of the ones before. The later papers made it much faster.

**Middle school.** Friedman's 2001 paper invented gradient boosting: fit a small tree to what the current model gets wrong, add it with a small weight, repeat. XGBoost (Chen and Guestrin, 2016) made it fast and added a smarter way to choose splits. LightGBM (Ke and colleagues, 2017) made it faster still with histograms and by growing the most useful leaf first. Reading the abstracts and the pictures is enough at this stage.

**High school.** Friedman's paper defines boosting as gradient descent in the space of functions and derives the algorithm for several losses, including the logistic loss this project uses. XGBoost's paper introduces the second-order split gain and the regularized objective. LightGBM's paper explains gradient-based one-side sampling and exclusive feature bundling, the two tricks behind its speed. Read them in order, one per week, after lesson 8.

**Stanford undergrad.** Work through the derivations: the functional gradient view, the second-order Taylor expansion leading to the leaf-weight formula, and the histogram algorithm's memory footprint. Then map every LightGBM parameter to the equation it controls. This is the fastest route from tuning by folklore to tuning by understanding.

**Stanford grad.** Compare the three on regularization (shrinkage, subsampling, leaf constraints), on categorical handling, and on the assumptions their speed tricks make (GOSS biases the gradient sample). The papers are also a study in how systems contributions (histograms, cache-aware access) can matter as much as algorithmic ones.

**PhD at Apple.** Together they explain why boosted trees are the default for tabular problems in industry and why their implementations converge; the open research is in uncertainty (natural-gradient and probabilistic boosting), in monotone and shape constraints, and in hybrids with neural representations.

#### [Why do tree-based models still outperform deep learning on tabular data? (Grinsztajn, Oyallon, Varoquaux, 2022)](https://arxiv.org/abs/2207.08815)

_paper · free · an evening or two · free; the evidence for the production choice_

**Age 5.** A paper where scientists tested many learning machines on lots of tables of numbers and found that the tree-based ones usually won.

**Middle school.** A research paper that compared gradient-boosted trees, random forests and several neural networks on 45 table datasets. Trees won most of the time, and the paper tries to explain why: tables have columns with different meanings and scales, and trees handle that naturally. It is the evidence behind choosing LightGBM as this project's production model.

**High school.** Read the abstract, the benchmark figure and the section on why trees win: networks are biased toward smooth functions while real tables have sharp thresholds; uninformative columns hurt networks more; and rotating the feature space (which networks are indifferent to) destroys the meaning of columns. Each point maps to a design choice in stage 9.

**Stanford undergrad.** The paper is a careful benchmark with tuned baselines, which is rarer than it should be. Note its scope (medium-sized datasets, no time or graph structure) and what that leaves open for this project: the spatial structure is the one place a network has an argument, which is why the challenger is a graph model, not an MLP.

**Stanford grad.** Follow-up work on tabular deep learning (attention-based tabular models, tree-inspired networks) chips at the gap without closing it. The methodological lesson is about benchmark design: the choice of datasets and tuning budget decides the headline, so read the appendix before the abstract.

**PhD at Apple.** A rare example of a negative-result benchmark shaping practice. For a lead deciding on architecture, it is the citation that ends the argument for tabular problems, and its limitations define exactly where an argument remains.

#### [Interpretable Machine Learning (Molnar), the SHAP chapter](https://christophm.github.io/interpretable-ml-book/shap.html)

_book · free · an evening · free online_

**Age 5.** A free book about asking a learning machine why it said what it said, and getting an answer you can understand.

**Middle school.** Christoph Molnar's online book explains ways to understand models. The SHAP chapter shows how to split one prediction into a contribution from each clue, so you can see that 'downstream speed dropped' pushed the model toward slow and 'Sunday' pushed it away. Read the chapter, then make the plots on the project's own model.

**High school.** SHAP values come from game theory: each feature's fair share of the difference between this prediction and the average prediction. For tree models there is an exact and fast algorithm, TreeSHAP. The chapter explains the plots you will make in stage 9: the summary plot (which features matter overall), dependence plots (how a feature's effect changes with its value), and force plots (one prediction explained).

**Stanford undergrad.** Understand what SHAP does and does not say: it attributes the model's output, not the world's causes; correlated features share credit in ways that depend on the method; and global importance from mean absolute SHAP differs from split-gain importance. The chapter's section on caveats is the important part, and the neighboring chapters on partial dependence and permutation importance give alternatives.

**Stanford grad.** Read alongside the original papers (Lundberg and Lee, 2017; the TreeSHAP paper, 2020) and the critiques concerning feature dependence and off-manifold evaluation. Use SHAP for debugging features and for per-station failure analysis, where it excels, and resist presenting it as a causal story about traffic.

**PhD at Apple.** Interpretability in products is a communication and accountability tool as much as a technical one. SHAP's computational cost at scale and its misuse as causal explanation are both live issues, and Molnar's book is the best neutral survey to hand to a product partner or a regulator before a discussion about what the model 'thinks'.

#### [scikit-learn user guide: Model evaluation, Probability calibration](https://scikit-learn.org/stable/modules/model_evaluation.html)

_docs · free · an evening · free; the metrics you will report_

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

## Phase 5: Classical machine learning, the rest

_weeks 22–28 · ≈50 h · builds a regression twin, station clusters, a tuned model_

**Goal.** The rest of the classical toolbox: regression and its metrics, other classifiers and multiclass, unsupervised learning, tuning done honestly, and the information theory under it all.

### What you need to learn

- Regression: least squares, ridge and lasso, quantile regression; MAE, RMSE, MAPE, R², residual plots.
- k-nearest neighbors, naive Bayes, support vector machines and kernels; softmax and multiclass metrics; ordinal targets.
- Clustering (k-means, hierarchical, DBSCAN), principal components, anomaly detection; how to judge unsupervised results.
- Hyperparameter search: grid, random, Bayesian, early stopping of trials; selection bias and nested validation.
- Entropy, cross-entropy, KL divergence, mutual information; information gain in trees; feature selection.
- Feature scaling, encoding and pipelines in scikit-learn.

### Topic: Regression: predicting a number, and the metrics that judge it

**Goal.** Predict the speed itself, not only whether it will be slow, and know how to score a numeric forecast.

**Age 5.**

Sometimes the question is not 'will it be slow?' but 'how fast will the cars go?' The answer is a number, like 32 miles per hour.

Guessing a number is called regression. You are right when your number is close, and you can measure how far off you were.

_Try this._ Guess how tall your friend is in centimeters, then measure. How far off were you? That distance is your error.

**Middle school.**

A regression model predicts a number from clues. The simplest is a line: speed in 30 minutes ≈ a + b × speed now. Fitting the line means choosing a and b so the guesses are as close as possible to what happened.

How close is close? Mean absolute error is the average distance between guess and truth. Root mean squared error punishes big misses more. R² says how much better you did than just guessing the average every time.

_Try this._ Take ten (speed now, speed 30 minutes later) pairs from one station and plot them. Draw the line you think fits best. Compute the average distance from the line to the points.

**High school.**

Linear regression minimizes the sum of squared errors; with several clues it is y ≈ w·x + b, fit by least squares. Metrics: MAE (robust, in mph), RMSE (sensitive to outliers), MAPE (percent, undefined at zero), R² (fraction of variance explained). Residual plots show where the model fails.

Regularization: ridge (L2) shrinks weights, lasso (L1) zeroes some, both fight overfitting when clues are many or correlated. Quantile regression predicts the 10th or 90th percentile instead of the mean, which is what a 'worst plausible speed' needs.

Trees and networks do regression too: LightGBM with `objective='regression'` or `quantile`, the same features as the classifier, a different target (`speed_t30`).

_Try this._ Train LightGBM to predict `speed_t30` and report MAE against persistence (speed now) and against the profile. Which stations have the largest errors, and at what hour?

**Stanford undergrad.**

Least squares as maximum likelihood under Gaussian noise; the normal equations and their geometry (projection onto the column space); Gauss-Markov; heteroscedasticity and why MAE or quantile losses are preferable when noise scales with level. Bias-variance for regression; ridge as a Bayesian prior; lasso and sparsity; elastic net.

Evaluation subtleties: RMSE decomposition into bias and variance, MASE (scaled by a naive forecast) for comparability across series, and the pinball loss for quantiles. Multi-output regression (three horizons at once) versus three models.

_Try this._ Show that minimizing squared error yields the conditional mean and minimizing absolute error yields the conditional median, and explain which the traffic product should report.

**Stanford grad.**

Generalized linear models unify regression targets (Gaussian, Poisson for counts such as flow, Gamma for positive skewed durations). Heteroscedastic and distributional regression (predict mean and variance, or full quantile sets) are the bridge to uncertainty. Regression under dependence: errors autocorrelated in time, spatially correlated across stations, and what that does to standard errors and to cross-validation.

Trees regress piecewise-constantly and cannot extrapolate; linear models extrapolate linearly and can be wildly wrong; networks are in between. Monotone and shape constraints in gradient boosting encode physics.

_Try this._ Fit a Poisson GLM to flow counts with calendar features and compare its calibration of predicted variance with a squared-error model. Then fit quantile boosting at 0.1, 0.5 and 0.9 and check pinball loss and crossing.

**PhD at Apple.**

Production forecasting is regression at scale: hierarchical targets, reconciliation across levels (station, route, city), distributional outputs for downstream optimization, and metrics chosen by the decision they serve (ETA error weighted by trips, not by rows). Research threads: distribution-free intervals for regression, multi-horizon coherence, and losses aligned with downstream utility.

_Try this._ Propose the regression target and metric for a route-level travel-time product built on station speeds, and say how station-level errors compound along a route.

**Check yourself.**

<details>
<summary>Why is RMSE larger than MAE on the same data?</summary>

Squaring weights large errors more; RMSE is at least MAE and the gap grows with outliers.

</details>

<details>
<summary>What does R² = 0 mean?</summary>

The model does no better than predicting the mean every time. It can be negative when the model is worse than that.

</details>

### Topic: Other classifiers and multiclass: k-nearest neighbors, naive Bayes, support vector machines, softmax

**Goal.** Know the rest of the classical toolbox, when each wins, and how two-class methods extend to many classes.

**Age 5.**

There are many ways to sort things into groups. One way: look at the things nearest to you and copy their label. Another: count how often each clue shows up in each group. Another: draw the widest possible gap between the groups.

_Try this._ Sort a pile of leaves into trees by looking at each leaf's closest match among leaves you already know. That is nearest neighbors.

**Middle school.**

k-nearest neighbors predicts by finding the k most similar past rows and voting. Naive Bayes multiplies the chance of each clue given each class, assuming clues are independent, which is naive but often works. A support vector machine finds the boundary with the widest margin between classes.

When there are more than two classes (light, moderate, heavy congestion), softmax turns scores into probabilities that add to one, and the loss is cross-entropy over all classes.

_Try this._ With five past rows labelled slow or fine and their speeds, classify a new row using k = 3 nearest neighbors by speed. Change k to 1 and 5; does the answer change?

**High school.**

k-NN: no training, all the cost at prediction; needs scaled features and a distance; works when similar inputs have similar outputs and data are plentiful. Naive Bayes: fast, works with counts and text, calibrates badly. SVM: maximize margin, use kernels to draw curved boundaries, strong on small clean data, slow on millions of rows.

Multiclass: one-vs-rest, one-vs-one, or native softmax (logistic regression and networks). Metrics generalize: per-class precision and recall, macro versus micro averages, the confusion matrix as the primary picture. Ordinal classes (light < moderate < heavy) deserve ordinal losses or a regression on the ratio.

_Try this._ Turn the speed ratio into three classes (above 0.8, 0.6 to 0.8, below 0.6) and train a softmax logistic regression; print the 3 × 3 confusion matrix and macro F1.

**Stanford undergrad.**

Bayes-optimal classification and what each method approximates: k-NN estimates the posterior locally (consistent as k grows with n, cursed by dimension); naive Bayes is a generative model with a factorized likelihood; the SVM is a large-margin discriminative model whose dual reveals the kernel trick and support vectors. Hinge loss versus logistic loss; kernels as inner products in a feature space; the representer theorem.

Softmax regression as the multinomial logit; its gradient generalizes p − y to a vector. Calibration in multiclass (temperature scaling). Cost-sensitive learning and thresholding per class.

_Try this._ Derive the dual of the soft-margin SVM and identify the support vectors; then show that kernel k-NN and kernel SVM use the same similarity in different ways.

**Stanford grad.**

Where each method still matters: k-NN as a retrieval baseline and in nearest-neighbor search at scale (approximate methods, vector databases); naive Bayes as a fast text baseline; SVMs in small-sample scientific settings and as the origin of kernel methods leading to Gaussian processes. Structured prediction and sequence labelling as multiclass over structured outputs.

Metric choice under class imbalance in multiclass, hierarchical classification, and label noise. The relation between margin theory (SVMs, boosting) and generalization.

_Try this._ Implement a kernelized ridge classifier and a linear SVM on the same features; compare margins and calibration, and explain why the boosted trees still win on this data.

**PhD at Apple.**

Nearest-neighbor retrieval is now infrastructure (embeddings plus approximate search underlie retrieval-augmented systems and recommenders); kernel methods live on in theory (neural tangent kernels) and in Gaussian processes. The practical multiclass problems at scale are extreme classification (millions of labels) and hierarchical, cost-sensitive decisions.

_Try this._ Design an approximate nearest-neighbor index over per-station day embeddings to find 'days like today' as a forecasting baseline, and state its failure modes.

**Check yourself.**

<details>
<summary>Why does k-NN need scaled features?</summary>

Distance mixes units; an unscaled flow in vehicles swamps a speed in mph, so the neighbors are chosen by one feature only.

</details>

<details>
<summary>What does softmax guarantee?</summary>

A vector of positive numbers that sum to one, so the outputs can be read as class probabilities.

</details>

### Topic: Unsupervised learning: clustering, principal components and anomaly detection

**Goal.** Find structure with no labels: groups of similar stations or days, compact descriptions of many columns, and rows that look wrong.

**Age 5.**

Sometimes nobody tells you the answers, and you just sort things into piles that look alike. That is unsupervised learning: finding groups on your own.

It can also spot the odd one out, like the one sensor that says cars are going 200 miles per hour.

_Try this._ Dump out a box of buttons and sort them into piles however makes sense to you. Now find the button that fits no pile.

**Middle school.**

Clustering groups similar rows: k-means picks k centers and assigns each row to the nearest one, then moves the centers, and repeats. Days could cluster into 'normal weekday', 'holiday', 'crash day'.

Principal component analysis finds the few directions along which the data vary most, so sixty columns can be summarized by a handful. Anomaly detection flags rows far from everything else, which is how you catch a broken detector.

_Try this._ Take one station's daily speed profiles (288 numbers per day) for a month and run k-means with k = 3. Which days landed together? Look at the odd cluster.

**High school.**

k-means (choose k with the elbow or silhouette), hierarchical clustering (a tree of merges), DBSCAN (density, no k, finds noise). PCA: center, take the top eigenvectors of the covariance, project; explained variance tells how many to keep. Anomaly detection: distance from the profile, isolation forests, or reconstruction error.

Uses here: cluster stations by their weekly profile to find station types; PCA on the 288-slot daily profile to get a few 'shape' features; anomaly scores to flag detector faults before they poison training.

_Try this._ Run PCA on daily speed profiles across all stations and plot the first two components colored by weekday. What does component one measure?

**Stanford undergrad.**

Clustering as optimization (k-means as coordinate descent on within-cluster variance; Gaussian mixtures and EM as its probabilistic version), the meaning of a 'good' clustering (no ground truth; stability and downstream utility), and the curse of dimensionality for distances. PCA as the best low-rank approximation (Eckart-Young), its relation to the SVD, and its limits (linear, variance is not always signal). Nonlinear reduction (t-SNE, UMAP) for visualization only.

Anomaly detection formally: density estimation and its thresholds; one-class methods; the distinction between point, contextual and collective anomalies (a detector stuck at 65 mph is contextual).

_Try this._ Implement k-means and EM for a Gaussian mixture from scratch in numpy on the daily profiles, and show that k-means is the limit of EM as the variances shrink.

**Stanford grad.**

Representation learning as the modern form of unsupervised learning: autoencoders, contrastive methods, and the fact that most 'unsupervised' structure now comes from pretext tasks. Mixture models and latent-variable models for regime detection on the fundamental diagram; hidden Markov models for congestion states over time. Evaluation of unsupervised methods through downstream tasks and through stability under resampling.

Anomaly detection under distribution shift: what counts as anomalous changes with roadwork; contextual detection using the profile and neighbors.

_Try this._ Fit a hidden Markov model with three states to a station's speed and occupancy and compare the inferred states with the relative and absolute labels; where do they disagree?

**PhD at Apple.**

Unsupervised structure at scale underlies embeddings, retrieval, and pretraining; the research frontier is in self-supervised objectives, in identifiability of latent variables, and in detection of anomalies and drift without labels in streaming systems. For sensor networks, unsupervised detector-health models are a product in their own right.

_Try this._ Propose a self-supervised pretraining task on unlabeled PeMS history for all of California and argue what representation it would learn and how you would test transfer to Fremont.

**Check yourself.**

<details>
<summary>Why is there no accuracy for clustering?</summary>

There are no labels to compare with. You judge clusters by stability, by interpretability, and by whether they help a downstream task.

</details>

<details>
<summary>What does the first principal component of daily profiles usually capture?</summary>

Overall level, how slow or fast the whole day was. Shape differences come in the later components.

</details>

### Topic: Hyperparameter tuning and model selection

**Goal.** Choose settings and models with a budget and without fooling yourself.

**Age 5.**

A learning machine has dials, like how many questions a tree may ask. Turning the dials changes how well it learns. Finding good dial settings is tuning, and you check each setting on a quiz the machine has not seen.

_Try this._ Set a kitchen timer to different times for cooking rice. Which time gives the best rice? That is tuning one dial.

**Middle school.**

Hyperparameters are settings you choose, not things the model learns: number of leaves, learning rate, how many lags to use. Tuning tries settings and keeps the best by the validation score. Grid search tries every combination; random search tries random ones and is usually better for the same effort.

The trap: if you try a thousand settings and pick the best on validation, the validation score is now too optimistic. That is why the test split stays untouched until the end.

_Try this._ Train LightGBM with learning rates 0.01, 0.05, 0.2 and leaves 15, 63, 255. Make a 3 × 3 table of validation AP. Is the best corner where you expected?

**High school.**

Search strategies: grid, random, Bayesian optimization (build a model of score versus settings and pick the most promising next point), and early-stopping-based methods such as successive halving that kill bad trials early. Tools: Optuna. Rules: tune few things (learning rate, leaves, min data in leaf, feature fraction), on the chronological validation split, with a fixed budget of trials, and log every trial.

Model selection is the same problem one level up: which model family to ship. Use the same validation protocol, compare against baselines, and prefer the simpler model unless the gain is clear and reproducible across folds.

_Try this._ Run 30 Optuna trials over four LightGBM parameters with early stopping, plot validation AP against trial number, and check how much the best trial beats the defaults.

**Stanford undergrad.**

Why random beats grid (most dials matter little, so sampling explores the important ones more densely); Bayesian optimization with Gaussian processes or tree-structured estimators; multi-fidelity methods (Hyperband). The selection bias of picking the maximum of many noisy scores, and nested cross-validation as the honest fix. Learning curves to decide whether more data or more capacity helps.

For time series: tuning on rolling-origin folds, reporting the spread, and re-tuning cadence when retraining monthly.

_Try this._ Simulate selecting the best of N random models with pure-noise validation scores and plot the expected 'best' score against N; then explain what nested cross-validation changes.

**Stanford grad.**

AutoML as automated pipeline search; its wins on benchmarks and its costs in compute and interpretability. Hyperparameter importance analysis (which dials mattered) as a research tool. Transfer of hyperparameters across tasks and scales, and scaling laws as principled tuning for large models. The reproducibility problem: reported gains within tuning noise.

_Try this._ Fit a functional ANOVA on your tuning trials to estimate each parameter's importance, and use it to prune the search space for the monthly retrain.

**PhD at Apple.**

At scale, tuning is an engineering system: distributed trials, early stopping, warm starts from previous versions, and budgets set by cost. The research frontier includes learning-rate-free methods, tuning under distribution shift, and using scaling laws to extrapolate from small runs.

_Try this._ Design the tuning policy for the monthly retrain: what is re-tuned, what is inherited, how many trials, and the statistical test that decides whether the new configuration replaces the old.

**Check yourself.**

<details>
<summary>Why does random search often beat grid search?</summary>

Most parameters matter little; random search spends its trials exploring more distinct values of the ones that matter.

</details>

<details>
<summary>Why must the test split stay untouched during tuning?</summary>

Any choice made by looking at a score on a split makes that score optimistic; the test split is the one score you never chose on.

</details>

### Topic: Information theory in machine learning: entropy, cross-entropy, KL divergence, mutual information

**Goal.** See why cross-entropy is the loss, what KL divergence measures, and how mutual information screens features.

**Age 5.**

Information is surprise. If I tell you the sun rose today, that is no surprise, so little information. If I tell you it snowed in Fremont, that is a big surprise.

A learning machine is trying to be less surprised by what happens. Its score is how surprised it was.

_Try this._ Play twenty questions and count how many yes-or-no questions you need. Fewer questions means the answer carried less surprise.

**Middle school.**

Entropy measures how unpredictable something is: a fair coin has one bit of entropy, a coin that always lands heads has zero. Cross-entropy measures how surprised your model is by the truth: if you said 90 % slow and it was slow, small surprise; if you said 5 % and it was slow, big surprise.

Training minimizes the average surprise, which is exactly the cross-entropy loss of lesson 7. Mutual information measures how much knowing one thing (rain) reduces your uncertainty about another (slow).

_Try this._ Compute the surprise −log₂(p) for p = 0.9, 0.5 and 0.05. Notice how fast it grows as p shrinks.

**High school.**

Entropy H(p) = −Σ p log p. Cross-entropy H(p, q) = −Σ p log q is the expected surprise when the truth follows p and you predicted q. KL divergence KL(p‖q) = H(p, q) − H(p) is the extra surprise from using q instead of p; it is zero only when q = p and it is not symmetric. Minimizing cross-entropy over your model's q is minimizing KL to the truth.

Mutual information I(X; Y) = KL between the joint and the product of marginals; it is zero when independent and captures nonlinear dependence, which correlation misses. Decision trees choose splits by information gain, a drop in entropy.

_Try this._ Estimate the mutual information between each feature (binned) and `y_30` on one month; rank the features and compare the ranking with LightGBM's gain importance.

**Stanford undergrad.**

The source-coding theorem gives entropy its meaning (bits needed on average); cross-entropy is the cost of coding with the wrong distribution. Maximum likelihood equals cross-entropy minimization equals KL minimization. Proper scoring rules and the log score. Differential entropy and its caveats. Mutual information estimation from data is hard in high dimensions; binning and k-NN estimators.

Applications: the information bottleneck as a view of representation learning, entropy regularization, and the variational lower bound (ELBO) which is the KL machinery behind variational autoencoders.

_Try this._ Prove KL(p‖q) ≥ 0 with Jensen's inequality and derive the ELBO for a latent-variable model as a KL identity.

**Stanford grad.**

Information theory in learning theory: mutual-information generalization bounds, minimum description length and Occam, channel capacity as a limit on what noisy features can carry about a target. Rate-distortion as the theory of lossy compression, which is what representation learning does. Estimating mutual information with neural estimators and their pitfalls.

_Try this._ Bound the achievable average precision from the mutual information between the feature set and the target on a synthetic problem, and discuss why the bound is loose for real features.

**PhD at Apple.**

Information-theoretic ideas organize much of modern ML: cross-entropy pretraining of language models, compression as a view of intelligence, and channel-capacity arguments about sensor networks (how much a detector network can tell you about a road). The research frontier includes information-theoretic accounts of generalization in overparameterized models and of what pretraining learns.

_Try this._ Estimate, from data, how much information about a station's speed in 30 minutes is carried by its own history versus by its neighbors versus by the calendar, and what that implies about the horizon at which each source dominates.

**Check yourself.**

<details>
<summary>Why is cross-entropy the loss for classification?</summary>

It is the expected surprise of the truth under the model, which equals the negative log-likelihood; minimizing it moves the model's distribution toward the true one in KL divergence.

</details>

<details>
<summary>What does mutual information capture that correlation misses?</summary>

Any dependence, including nonlinear and non-monotone ones; correlation only measures linear association.

</details>

### Read and watch, explained at every level

#### [scikit-learn user guide: Supervised and unsupervised learning](https://scikit-learn.org/stable/supervised_learning.html)

_docs · free · an afternoon · free; one page per method with its pitfalls_

**Age 5.** The manual for a big toolbox of learning methods, with a page for each tool that explains what it is for.

**Middle school.** The scikit-learn user guide has one section per method: nearest neighbors, naive Bayes, support vector machines, linear models, clustering, principal components, anomaly detection. Each page explains the idea, shows the code, and links an example. Read the page for each method the week you meet it.

**High school.** For the classical-ML phase: the pages on nearest neighbors, naive Bayes, support vector machines, linear models (ridge, lasso, logistic), clustering (k-means, DBSCAN, hierarchical), decomposition (PCA), novelty and outlier detection, and model selection (cross-validation, grid and random search, learning curves). Their examples are runnable on the project's features.

**Stanford undergrad.** The guide is unusually good at stating each method's assumptions, complexity and pitfalls (for instance, k-means's sensitivity to scaling, or the meaning of the SVM's C). Read the 'tips on practical use' sections; they are the accumulated experience of the maintainers.

**Stanford grad.** It is also a reference for evaluation and preprocessing (scalers, encoders, pipelines, `TimeSeriesSplit`). Its consistent API (fit, predict, transform) is the pattern every internal ML library ends up imitating, and the reason to prototype in scikit-learn before writing anything custom.

**PhD at Apple.** scikit-learn's documentation is a public good and a standard; its design decisions (no GPU, no deep learning, stability over novelty) are a deliberate scope that has kept it reliable for fifteen years, which is a lesson in library governance.

#### [An Introduction to Statistical Learning with Python (ISLP), chapters 2–4 and 8](https://www.statlearning.com/)

_book · free · three weeks · now chapters 6 (regularization), 9 (SVMs) and 12 (unsupervised)_

**Age 5.** A famous textbook, free, that explains learning machines with pictures and real examples, written by teachers who invented some of the methods.

**Middle school.** Written by statisticians, with Python labs. Chapter 2 explains what learning from data is and why a model can be too simple or too complicated. Chapter 3 is linear regression, chapter 4 is classification including logistic regression, and chapter 8 is trees, forests and boosting. Read a chapter, then do its lab in a notebook.

**High school.** The clearest explanation anywhere of bias versus variance (chapter 2, with figures worth redrawing), of logistic regression as a model of probability (chapter 4), and of why boosting works by fitting small trees to what is left over (chapter 8). Its lab for chapter 8 uses the same scikit-learn calls you will make in stage 9.

**Stanford undergrad.** ISLP is the practical face of The Elements of Statistical Learning. Add chapter 5 on cross-validation and the bootstrap, and chapter 6 on model selection and regularization; together with the core chapters they cover phase 4's theory. Its treatment of the bootstrap is the basis for lesson 6's block bootstrap, with the modification that blocks are days.

**Stanford grad.** After ISLP, go to ESL for proofs and depth. ISLP's boosting section gives the additive-model view; combine it with Friedman's 2001 paper for the gradient view that LightGBM implements. Note where ISLP's examples assume independent rows and how each method's assumptions change under temporal dependence.

**PhD at Apple.** As a citation for standard methods and as a teaching text it is the reference, and its evolution from R to Python editions tracks the field's tooling. For research, its framing of methods as estimators with bias and variance you can reason about is the habit that distinguishes a careful ML engineer from a library user.

#### [Google's Clustering course](https://developers.google.com/machine-learning/clustering)

_course · free · a few hours · free; similarity, k-means and how to judge clusters_

**Age 5.** A free short class about sorting things into groups without being told the groups, like putting similar songs together.

**Middle school.** A short free course from Google about clustering: how to measure similarity, the k-means algorithm step by step, how to check whether clusters are any good, and where clustering is used. A few hours. Do it before clustering the stations.

**High school.** Its sections on similarity measures (why scaling matters), on k-means and its failure modes, and on interpreting and evaluating clusters are exactly the unsupervised topic's needs. The section on choosing k with the elbow and silhouette methods is practical.

**Stanford undergrad.** The course is introductory and k-means-centric; pair it with scikit-learn's clustering guide for density-based and hierarchical methods, and with the Gaussian-mixture view for the probabilistic version.

**Stanford grad.** Its emphasis on the similarity measure as the real modelling decision is correct and often skipped; for stations, the choice of profile distance decides the clusters more than the algorithm does.

**PhD at Apple.** A free, focused mini-course; useful as the standard reference when a team disagrees about what a clustering result means.

#### [Information Theory, Inference, and Learning Algorithms (MacKay)](https://www.inference.org.uk/mackay/itila/)

_book · free · a semester, selectively · free; entropy, coding and learning as one subject_

**Age 5.** A famous free book by a scientist who explained information, coding and learning as one big idea, with puzzles on every page.

**Middle school.** David MacKay's textbook, free online, covers information theory (entropy, coding, compression), probabilistic inference, and machine learning (clustering, neural networks, Gaussian processes) as one subject. It is long; read chapters 1, 2 and 4 for entropy and coding, and the exercises are famously good.

**High school.** Chapters 2 and 4 make entropy and the source-coding theorem concrete; chapter 22 on maximum likelihood and clustering connects to the unsupervised topic; chapter 36 previews decision theory. His writing style, with worked puzzles, rewards patience.

**Stanford undergrad.** The book unifies coding, inference and learning through probability; its chapters on Bayesian inference, on model comparison (Occam's razor as a consequence of marginal likelihood) and on Gaussian processes are a second, deeper take on the Bayesian topic.

**Stanford grad.** MacKay's treatment of the information-theoretic view of learning (chapters on inference and on neural networks as inference) is the intellectual background for variational methods and for the information bottleneck.

**PhD at Apple.** A classic, free, and still one of the best-written technical books in the field; its lectures (also free) are a model of teaching.

#### [StatQuest with Josh Starmer (YouTube)](https://statquest.org/)

_video · free · hours, as needed · the videos on k-means, PCA, SVMs and regularization_

**Age 5.** Videos with silly songs and simple drawings that explain how learning machines work, one small idea at a time.

**Middle school.** A YouTube channel where each video explains one idea, such as what a decision tree is or how logistic regression works, with drawings and a short song. Watch the videos on decision trees, random forests, gradient boosting and logistic regression during phase 4; they are slow, clear and free.

**High school.** StatQuest's series on gradient boosting (four videos) walks through fitting trees to residuals step by step with numbers, which is exactly lesson 8's mechanism. The videos on ROC and AUC, on cross-validation and on bias and variance give the pictures behind lesson 6. The logistic regression series covers maximum likelihood and the meaning of the coefficients.

**Stanford undergrad.** The channel's strength is tracing a computation with small numbers until it is obvious; use it to check your understanding of a derivation rather than as a source of the derivation. Its XGBoost series covers the second-order split gain, which is the LightGBM math in lesson 8's higher levels.

**Stanford grad.** It stops at intuition and worked examples, which is the right place for a video; the formal accounts are in ISLP and ESL. When you teach, its pattern (a concrete numeric example before any symbol) is worth stealing.

**PhD at Apple.** A reminder that clarity is a research skill; the channel's popularity among practitioners reflects a gap textbooks leave, and it is the resource to hand to a stakeholder who wants to know what a boosted tree actually does.

#### [Random Search for Hyper-Parameter Optimization (Bergstra, Bengio, 2012)](https://jmlr.org/papers/v13/bergstra12a.html)

_paper · free · an evening or two · free; why random beats grid_

**Age 5.** A paper that found that trying dial settings at random beats trying them in a neat grid, because only a few dials really matter.

**Middle school.** Bergstra and Bengio showed that random search finds good hyperparameters faster than grid search for the same number of trials. The reason is in one figure: if only one of two dials matters, a grid wastes trials repeating the same values of the important one, while random search tries many different values. Free to read in the Journal of Machine Learning Research.

**High school.** Read the introduction and the figure, then the experiments on neural networks. The lesson generalizes: budget your trials, sample them randomly (or with a smarter method), and look at which parameters actually moved the score.

**Stanford undergrad.** The argument rests on low effective dimensionality of the hyperparameter response surface; the paper also introduces the use of quasi-random sequences. It is the foundation for later Bayesian and multi-fidelity methods, which improve on random search by modelling the response surface.

**Stanford grad.** Compare with Hyperband and Bayesian optimization papers; random search remains the honest baseline that every tuning method must beat, and it parallelizes trivially.

**PhD at Apple.** A rare methodology paper that changed practice immediately; for a lead, its message is that a well-chosen search space and a fixed budget matter more than the search algorithm.

#### [Optuna documentation](https://optuna.readthedocs.io/)

_docs · free · an afternoon · free; tuning with pruning in an hour_

**Age 5.** The manual for a helper that tries dial settings for you, learns which ones look promising, and stops the bad tries early.

**Middle school.** Optuna is a free Python library for hyperparameter search. You write a function that trains a model with settings drawn from ranges you define and returns the validation score; Optuna runs trials, favors promising regions, and can stop bad trials early. The tutorial pages get you running in an hour.

**High school.** Read the tutorial (define an objective, create a study, optimize), the pruning page (early stopping of bad trials with LightGBM's callback), and the visualization page (parameter importance, optimization history). Use it to tune four LightGBM parameters with a budget of 30 trials on the chronological validation split.

**Stanford undergrad.** Its default sampler is a tree-structured Parzen estimator, a Bayesian optimization method; the docs explain samplers and pruners so you can choose. Understand how to make trials reproducible (seeds), how to store studies (SQLite) and how to run trials in parallel on the Mac's cores.

**Stanford grad.** Optuna's design (define-by-run search spaces, pluggable samplers) makes it usable for pipeline search, not only parameters; its parameter-importance analysis is a practical functional-ANOVA implementation for pruning the search space of a monthly retrain.

**PhD at Apple.** Tuning libraries are infrastructure; the questions at scale are distributed trials, warm starts across model versions, and integration with experiment tracking, all of which Optuna's docs address at a level worth reading before building your own.

#### [Visual Information Theory (Olah)](https://colah.github.io/posts/2015-09-Visual-Information/)

_essay · free · an hour · free; entropy and KL with pictures_

**Age 5.** A web page with pictures that explains what information is, using rectangles whose size shows how surprising something is.

**Middle school.** Chris Olah's essay explains entropy, cross-entropy and KL divergence with diagrams of code lengths: rare events need long codes, and using the wrong codebook wastes bits. It makes the cross-entropy loss feel like a real cost. Free, about an hour.

**High school.** Read it before the information-theory topic; its diagrams show why cross-entropy is at least entropy, why KL is not symmetric, and what mutual information measures. Then redo its examples with the traffic label's base rate as the distribution.

**Stanford undergrad.** The essay gives the coding interpretation that makes the theorems intuitive (the source-coding theorem as optimal code lengths); with it, the KL-based derivations in variational inference and the information bottleneck read naturally.

**Stanford grad.** Olah's other essays (on neural network representations, on attention) are of the same quality; his work on interpretability is where the visual method meets research.

**PhD at Apple.** An example of explanation as research communication at its best; when you write for a broad technical audience, this is the standard.

### Build

- Train LightGBM regression and quantile models for `speed_t30`; report MAE against persistence and the profile, per hour.
- Turn the speed ratio into three classes and train softmax regression, k-NN and a linear SVM; compare macro F1 and calibration.
- Cluster stations by weekly profile; run PCA on daily profiles; build an anomaly score for detector faults and inspect the top twenty days.
- Tune four LightGBM parameters with Optuna on the chronological validation split with a 30-trial budget; report the gain over defaults, honestly.
- Rank features by mutual information with `y_30` and compare with gain importance.

### Done when

- A regression report with MAE per horizon and a quantile band whose coverage you measured.
- A station-type map from the clustering and a written interpretation of the first two principal components.
- A tuning log with the best configuration and its validation gain, or a note that defaults were fine.

### Pitfalls

- Reporting R² without a baseline.
- Interpreting clusters as ground truth.
- Tuning a hundred trials and believing the best score.

## Phase 6: Time series and forecasting

_weeks 27–33 · ≈45 h · builds stage 8 targets, stage 11 episodes_

**Goal.** Make the model forecast, and evaluate it as a forecast, with the traffic physics behind why it works.

### What you need to learn

- Autocorrelation and the ACF; stationarity, trend, seasonality; lag and rolling features done right; the forecast horizon; direct versus recursive strategies.
- Persistence and seasonal-naive baselines; rolling-origin backtests; leakage through time.
- ARIMA and exponential smoothing at reading level; why trees on lag features replaced them here.
- Spatial correlation, the road graph, and kinematic waves: why a downstream slowdown predicts your future.
- Event-level metrics: episode onset, lead time, false alarms per station-day; calibration and thresholds for alerts.

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

#### [Forecasting: Principles and Practice, the Python edition (Hyndman, Athanasopoulos), chapters 1–5, 7, 9](https://otexts.com/fpppy/)

_book · free · three weeks · free online; the forecasting book_

**Age 5.** A free book about predicting the future from the past, by people who do it for a living, with lots of pictures of wiggly lines.

**Middle school.** The standard forecasting textbook, free online, now with Python. Chapters 1 to 5 cover what forecasting is, how to draw time series, how to split a series into trend and seasonal parts, simple forecasting methods, and how to check whether a forecast is any good. Chapter 7 is regression with time features, chapter 9 is ARIMA. Read 1 to 5 fully and do the exercises.

**High school.** Chapter 5's evaluation (training and test in time order, residual checks, accuracy measures) and its simple baselines (naive, which is persistence; seasonal naive, which is our profile rate) are the heart. Chapter 3's decomposition helps you see the daily and weekly cycles in a station's speed, and chapter 7's Fourier terms are the sine and cosine time features of lesson 5.

**Stanford undergrad.** The book supplies the statistical forecasting vocabulary that ML-first people lack: stationarity, autocorrelation, exponential smoothing, ARIMA, and its 'time series cross-validation', which is the rolling-origin protocol lesson 6 uses. Chapter 9 shows why ARIMA is a linear model on lags with a principled error structure, which clarifies what a gradient-boosted model on lag features adds and gives up.

**Stanford grad.** Compare its classical models with the ML approach: one ARIMA per series does not scale to a hundred stations with spatial dependence, and its hierarchical and global-model chapters bridge toward the global boosted and graph models used here. Its treatment of forecast intervals is the classical route to the uncertainty questions of phase 8.

**PhD at Apple.** Hyndman's wider work (the M4 and M5 forecasting competitions, hierarchical reconciliation, feature-based model selection) is the empirical forecasting literature. The M5 result, gradient-boosted models on lag features winning at scale, is direct evidence for this project's choice of LightGBM as the production model.

#### [Kaggle Learn: Time Series](https://www.kaggle.com/learn/time-series)

_course · free · half a day · free; lag features and hybrids in an afternoon_

**Age 5.** A short online lesson about guessing what comes next in a line of numbers.

**Middle school.** Six lessons on store sales data: the trend, seasonality using sine and cosine features, using past values as features (lags), combining two models, and forecasting several steps ahead with machine learning. About half a day, graded as you go.

**High school.** Its lag-feature and Fourier-feature lessons are exactly our lags and the sine and cosine encoding of time. Its hybrid lesson, a simple model for the predictable part plus a boosted model for what is left, is a pattern to try here: profile plus a residual model.

**Stanford undergrad.** Note how the course handles multi-step forecasting (direct versus recursive strategies) and where leakage hides in lag construction. Its examples are single series without spatial structure, so the neighbor features of lesson 5 and the graph models of lesson 10 are beyond it.

**Stanford grad.** The hybrid idea generalizes to 'physics plus residual', which lesson 10's advanced prompt asks about, and its Fourier treatment of seasonality is the linear analogue of what a recurrent network learns. Short courses like this are where practitioner defaults come from, so know what they teach and where they stop.

**PhD at Apple.** The design principle it plants, decompose into predictable structure and a learned residual, is robust across production forecasters and is how a classical model can remain as a fallback when a learned model goes stale.

#### [statsmodels documentation: one ARIMA fit on one station](https://www.statsmodels.org/stable/tsa.html)

_docs · free · an afternoon · free; do it once to know what it is_

**Age 5.** A tool that fits an old but famous kind of forecast to a line of numbers. You try it once to see what it does.

**Middle school.** statsmodels is a Python library for statistics. Fit an ARIMA model to one station's speed, look at its forecast for the next hour and at what it got wrong. The point is to know what ARIMA is and how it behaves, not to put it in the pipeline.

**High school.** ARIMA(p, d, q) predicts a value from p past values, d differences, and q past errors. Fit it with `ARIMA(series, order=(2, 0, 1)).fit()`, forecast twelve steps, and compare with persistence. You will see a linear model on lags that does about as well as persistence at short horizons and drifts toward the mean at long ones.

**Stanford undergrad.** Understanding ARIMA clarifies why lag features work and what a recurrent network adds: nonlinearity, exogenous inputs, and parameters shared across stations. SARIMAX adds seasonality and exogenous regressors, which is the classical way to include incidents and rain, and its diagnostics (residual autocorrelation) are useful for any model.

**Stanford grad.** Vector ARIMA and state-space models with the Kalman filter are the classical route to multi-station forecasting, and their scaling limits are what motivated the graph deep-learning literature. statsmodels' state-space module is a good place to see the Kalman filter concretely before reading about recurrent models as nonlinear state-space models.

**PhD at Apple.** Classical models remain the interpretable baseline and a source of principled uncertainty that deep models lack. A production system often keeps a simple statistical model as the fallback when the learned model is stale or its inputs are missing, and statsmodels is where that fallback would be built.

#### [traffic-simulation.de and its model documentation (Treiber, Kesting)](https://traffic-simulation.de/)

_interactive · free · an evening · free simulator and model docs; the physics of jams_

**Age 5.** A website where you can watch cartoon cars drive on a ring road, add a slow truck, and see a traffic jam appear from nowhere and travel backwards. The pages next to it explain why.

**Middle school.** A free browser simulation by the authors of a well-known traffic textbook. You change the number of cars, the road shape and driver behavior, and watch jams form and move upstream. The site's documentation pages explain the models it runs (car-following and lane-changing) without needing the book.

**High school.** Use the simulator to see lesson 10's ideas: the fundamental diagram, capacity, the backward-moving jam wave and its speed, and how an on-ramp creates a bottleneck. The documentation pages describe the Intelligent Driver Model and the kinematic-wave picture in enough detail to follow the equations.

**Stanford undergrad.** The model documentation covers the Intelligent Driver Model and lane-changing rules with parameters and references to the open literature; the authors' papers (free preprints) supply the theory of the textbook. Together they give you the physical priors that justify monotone constraints and neighbor direction.

**Stanford grad.** Use simulator-generated data for stress tests on regimes rare in the archive and as a baseline for physics-informed components; the source code of the simulation is open and readable.

**PhD at Apple.** Coupling learned forecasters with open simulators is where counterfactual evaluation starts; the site and its code are the practical entry, and the Intelligent Driver Model is simple enough to make differentiable.

#### [Kaggle: Store Sales, Time Series Forecasting competition](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)

_dataset · free · a week · free; forecasting practice off the traffic data_

**Age 5.** A practice contest with real store sales numbers where you guess future sales and get a score, for free.

**Middle school.** A permanent beginner competition on Kaggle: predict daily sales for many products and stores from history, holidays and oil prices. You submit a file and get a score. It is the exercise for the time-series topic on data that is not traffic, and the public notebooks show many approaches.

**High school.** Build lag and calendar features, a global gradient-boosted model, and a rolling-origin validation, then submit; read three public notebooks and note where they leak (features that use the future). The evaluation metric and the multi-series structure mirror the M5 findings.

**Stanford undergrad.** Use it to practice global models across series, hierarchical structure (store, family), exogenous regressors and honest validation on a leaderboard; then compare your validation score with your leaderboard score to see selection bias in action.

**Stanford grad.** Competitions are natural experiments about what works under a protocol; this one is small enough to iterate quickly and is a fair place to test a new idea before applying it to the traffic data.

**PhD at Apple.** A free, standing benchmark for time-series practice and for calibrating a newcomer's forecasting skill.

#### [The M5 forecasting competition and its findings (Makridakis, Spiliotis, Assimakopoulos)](https://www.kaggle.com/competitions/m5-forecasting-accuracy)

_dataset · free · days · free; the largest evidence for global boosted models_

**Age 5.** A big contest where thousands of people tried to predict how many of each thing a store would sell. The winners used the same kind of tree model we use for traffic.

**Middle school.** M5 was a large forecasting competition on Kaggle in 2020 with real Walmart sales data, tens of thousands of series. The organizers later published what worked: gradient-boosted trees on lag features won, simple statistical methods were strong baselines, and combining models helped. The competition page and the summary papers are free.

**High school.** Read the organizers' summary of the accuracy competition: the winning approaches, the role of cross-learning (one model across many series), and the evaluation metric. Then read one winner's write-up. The pattern, a global boosted model on lag and calendar features with careful validation, is this project's stage 9.

**Stanford undergrad.** M5 is the largest public evidence for global models over per-series models, for boosted trees over classical methods at scale, and for the importance of the validation protocol (many teams overfit the public leaderboard). Its uncertainty track (quantile forecasts) connects to the uncertainty topic in the final phase.

**Stanford grad.** The M-competition series (M1 through M6) is the empirical forecasting literature; read the M4 findings alongside M5 to see how the consensus moved from statistical ensembles to boosted global models, and what the organizers concluded about deep learning's role.

**PhD at Apple.** Competitions are natural experiments about what works under a fixed protocol; their limitations (leaderboard overfitting, no deployment costs) are as instructive as their results. For a forecasting team, M5's write-ups are the closest thing to a field manual.

### Build

- `13_feature_engineering.py` with forward-shifted targets and the rolling-origin split with a one-day gap.
- The episode-level half of `18_evaluate_lead_time.py`: onset, lead-time histogram, share caught, false alarms.
- Plot the ACF of speed for one station; estimate the shockwave speed from the fundamental diagram; compute skill over persistence per horizon.
- Train on the absolute label too; compare prevalence and skill and write the decision.

### Done when

- A lead-time histogram and one sentence per horizon on skill over persistence.
- A written decision on which label the product means, with numbers.

### Pitfalls

- `shift` in the wrong direction; a rolling window that looks ahead.
- Evaluating on rows whose target was imputed.

## Phase 7: Neural networks

_weeks 32–42 · ≈80 h · builds stage 9 networks_

**Goal.** The MLP in numpy with backprop by hand, then the spatio-temporal net in PyTorch on the M1's GPU.

### What you need to learn

- The perceptron and the multilayer perceptron; activations; loss functions; backpropagation for one hidden layer, by hand.
- Gradient descent variants (SGD, momentum, Adam); initialization; batch normalization, dropout, learning-rate schedules; early stopping.
- PyTorch: tensors, autograd, `nn.Module`, `DataLoader`, the `mps` device, saving and loading, `model.eval()`.
- Sequence models: RNN, GRU, LSTM, 1-D temporal convolutions; graph neural networks: adjacency, message passing; how DCRNN and STGCN model traffic.
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

#### [3Blue1Brown: Neural networks](https://www.3blue1brown.com/topics/neural-networks)

_video · free · two hours · free; four videos, watch before Karpathy_

**Age 5.** Four videos with moving pictures that show what a neural network is and how it learns, using handwritten numbers as the example.

**Middle school.** Watch them before Karpathy's course. They show a network as layers of numbers, what a weight and a bias do, what gradient descent means, and what backpropagation computes, all with animations of the digit-recognition example. Pause often and predict what the next picture will show.

**High school.** The third and fourth videos derive backpropagation with the chain rule visually. Pause and write the equations yourself as they appear; then lesson 9's numpy implementation will feel like transcribing the pictures. The first video's description of a neuron as a weighted sum plus a squashing function is lesson 7 in one image.

**Stanford undergrad.** The series builds intuition for the computation graph and for how gradients flow backward through it; after it, the formal Jacobian-chain view is easy. Its digit example is a plain multilayer perceptron, the same architecture as the numpy network in stage 9, and its remarks on what the hidden layers actually learn are a good caution against over-interpreting them.

**Stanford grad.** Good for teaching; note what it simplifies (no batching, no optimizers beyond plain gradient descent, no regularization, no normalization). The newer videos on transformers and attention extend the same visual method to the models of phase 8, and are the shortest route to understanding attention geometrically.

**PhD at Apple.** Its visual language has become the standard for explaining networks to non-specialists. Reuse the logic of its diagrams when presenting the spatio-temporal network to a product team, and its treatment of what a layer 'sees' as a caution when someone asks what the model has learned.

#### [TensorFlow Playground](https://playground.tensorflow.org/)

_interactive · free · fifteen minutes · free; poke a network in the browser_

**Age 5.** A web page where you build a tiny brain by clicking, then press play and watch it learn to separate blue dots from orange dots.

**Middle school.** A free browser toy: pick a dataset of dots, add layers and neurons, choose a learning rate, press play, and watch the network learn a boundary. You can see what each neuron detects. It makes the words in lesson 9 (layer, neuron, activation, learning rate) into something you can poke.

**High school.** Use it to run the experiments lesson 9 describes: too high a learning rate diverges; too few neurons underfit the spiral; adding features (x², sin x) lets a small network solve what a bigger one struggles with; regularization smooths the boundary. Fifteen minutes here saves an hour of confusion later.

**Stanford undergrad.** It visualizes the decision boundary and per-neuron activations for a two-input problem, which is the clearest demonstration of what hidden layers compute and why feature engineering and depth are partial substitutes for each other.

**Stanford grad.** Its role for you is demonstration when teaching; it cannot show the high-dimensional effects (initialization statistics, batch norm) that matter in practice.

**PhD at Apple.** A reminder that a well-designed toy can carry a concept further than a lecture; it is the demo to open when a stakeholder asks what a neural network 'is'.

#### [Neural Networks: Zero to Hero (Karpathy), micrograd and makemore](https://karpathy.ai/zero-to-hero.html)

_video · free · three weeks · free; backprop from scratch, the best there is_

**Age 5.** A programmer builds a tiny brain from scratch on video, typing every line, and explains each one as he goes.

**Middle school.** Andrej Karpathy's free YouTube course. micrograd builds automatic differentiation, the machinery that computes gradients, in about a hundred lines of Python. makemore builds models that generate names, starting from counting letter pairs and ending with a small transformer. Code along in a notebook; pause and predict what each line does before he runs it.

**High school.** micrograd is lesson 9's backward pass done generally: every operation records how to pass its gradient back, and the chain rule falls out. Build it, then use it to train lesson 7's logistic regression. makemore's early lessons teach the training loop, learning rates, initialization and batch normalization by showing failures happening on screen and fixing them.

**Stanford undergrad.** The course's method (make the bug, watch it, fix it) teaches debugging networks better than any textbook. Its lesson on the statistics of activations and gradients is a practical tutorial on initialization and normalization. After micrograd, PyTorch's autograd is obvious rather than magical, and you will know what `requires_grad` and `backward` actually do.

**Stanford grad.** makemore's progression to attention is the clearest path to understanding transformers, and the follow-up on building GPT completes it. For traffic the relevant carry-over is the sequence-modelling mindset and the training diagnostics: reading the loss curve, the gradient norms and the activation histograms before changing an architecture.

**PhD at Apple.** Karpathy's teaching sets the bar for from-scratch pedagogy, and his emphasis on reading gradients and activations is the diagnostic culture a research team should share. For a lead, the course is the fastest way to give a strong engineer without deep-learning background a working mental model in a week.

#### [PyTorch tutorials: Learn the Basics; the MPS backend notes](https://pytorch.org/tutorials/beginner/basics/intro.html)

_docs · free · an afternoon · free_

**Age 5.** The official lessons for the tool that runs neural networks, plus a note about making it use the Mac's graphics chip so it goes faster.

**Middle school.** Learn the Basics is eight short pages: tensors (grids of numbers), datasets, building a model, automatic gradients, the training loop, and saving. The MPS notes explain how to run on the Mac's GPU by setting the device to `mps`. Do the tutorial in a notebook and run each cell.

**High school.** After it you can write lesson 9's PyTorch network: `nn.Module`, `DataLoader`, the training loop, `torch.no_grad()` for evaluation, and saving a `state_dict`. The MPS page lists the limits that matter on the Mac mini: float32 and float16 only, no float64, and an environment variable that lets unsupported operations fall back to the CPU.

**Stanford undergrad.** Understand tensors' memory model (views versus copies, contiguity, in-place operations), the autograd graph and `requires_grad`, and why moving data to the device in large chunks matters more than any architectural choice for small models. Read the profiler tutorial to see where time actually goes on MPS; the answer is usually kernel launches, not arithmetic.

**Stanford grad.** The advanced tutorials (custom autograd functions, `torch.compile`, distributed training) are the next layer. For this project, test whether `torch.compile` helps on MPS rather than assuming; support is uneven. The tutorials on saving and loading, and on reproducibility, are the ones that prevent embarrassment at handover.

**PhD at Apple.** PyTorch's eager, Pythonic design is why research iterates fast; its deployment story (TorchScript's decline, ExecuTorch, ONNX and Core ML export) is where production effort goes. Knowing both ends, and the operator-support gaps between them, is what lets you promise a model can ship on a device.

#### [Dive into Deep Learning (d2l.ai), chapters 3–5 and 9–10](https://d2l.ai/)

_book · free · reference, chapters as needed · free, PyTorch throughout_

**Age 5.** A free online textbook where every idea comes with code you can run, about teaching computers using layers of simple math.

**Middle school.** Chapters 3 to 5 cover linear and softmax regression and multilayer perceptrons with PyTorch code you can execute. Chapters 9 and 10 cover recurrent networks (RNN, GRU, LSTM) and how to train them well. Run the notebooks; each section ends with exercises that are worth doing.

**High school.** Chapter 5's multilayer perceptron from scratch mirrors lesson 9's numpy network; chapter 9 explains why GRUs can remember and forget, which is the sequence half of the spatio-temporal network. The exercises about changing hidden size and learning rate are the experiments to actually run on the traffic data.

**Stanford undergrad.** d2l's coverage of optimization (chapter 12), attention (chapter 11) and computational performance (chapter 13) is deeper than most books and free. Its consistent code style makes it easy to lift components, and its treatment of truncated backpropagation through time explains what a recurrent model can and cannot learn from a 12-step window.

**Stanford grad.** Its graph-neural-network coverage is thin; use CS224W for that. Its attention chapters are the bridge to spatio-temporal transformers, and its chapter on computational performance (asynchrony, memory, multi-device) is the right background before profiling on unified memory.

**PhD at Apple.** A living, open textbook with code in several frameworks is itself a research-infrastructure achievement. Cite it for standard architectures, use its implementations as reference points when reproducing papers, and note its authors' choice to include the messy practical chapters most textbooks omit.

#### [Understanding Deep Learning (Prince)](https://udlbook.github.io/udlbook/)

_book · free · reference, chapters as needed · free PDF; chapters 1 to 9 now, the rest in phase 8_

**Age 5.** A newer big book about deep learning, free to read, with lots of clear pictures that show what each part of a network does.

**Middle school.** Simon Prince's textbook (2023), free as a PDF with notebooks. It covers supervised learning, networks, training, measuring performance, regularization, convolutional and recurrent and transformer networks, graph networks, and generative models. It is more modern and more visual than the older Deep Learning book. Read the early chapters after phase 6.

**High school.** Chapters 1 to 9 are the modern version of lesson 9: networks, loss functions, fitting, gradients and initialization, measuring performance, and regularization, each with figures that make the idea visible. Chapter 13 on graph neural networks covers the neighbor mixing of the spatio-temporal net. The notebooks let you run every figure.

**Stanford undergrad.** Prince's treatment of why deep networks generalize, of the loss landscape, and of the transformer architecture is the clearest current textbook account, and it includes the recent empirical findings (double descent, scaling) that Goodfellow predates. Chapter 12 on transformers is the reading before lesson 8's attention material.

**Stanford grad.** The chapters on generative models (VAEs, GANs, normalizing flows, diffusion) are a compact, correct introduction with the math intact, and the chapter on ethics is unusually careful for a technical text. Use it as the reference when reading current papers.

**PhD at Apple.** A free, modern, rigorous textbook is a public good; it is the one to recommend to a new graduate student and the one to cite for standard architectures in the current idiom. Its figures are also the model for how to draw a network in a paper.

#### [Practical Deep Learning for Coders (fast.ai)](https://course.fast.ai/)

_course · free · eight weeks · free; the practitioner's counterweight to the theory_

**Age 5.** A free class where you train a real neural network on the very first day and only later learn all the details, like learning to ride a bike before studying how it works.

**Middle school.** Jeremy Howard's course is top-down: you build working models first (on images, text and tables) and learn the theory as you need it. It uses the fastai library built on PyTorch. It is good for confidence and full of practical tricks that make training work.

**High school.** Its tabular lessons (random forests first, then neural networks on tables) and its training tricks (a learning-rate finder, the one-cycle schedule) are directly usable in stage 9. The lesson on embeddings for categorical variables shows how to give station identity to a network, which is how the spatio-temporal net could learn per-station behavior.

**Stanford undergrad.** The course's emphasis on iterating quickly and inspecting data is a counterweight to theory-first study, and its stated view that tree ensembles usually win on tabular data matches the evidence and this project's production choice. Its chapter on data ethics is short and worth the time for anyone whose model will issue alerts to drivers.

**Stanford grad.** The fastai library's abstractions can hide details you need for a custom model such as the spatio-temporal network. Learn the ideas from the course, then implement in plain PyTorch. The course's lessons on collaborative filtering and on tabular embeddings are the clearest introduction to learned representations for categorical data.

**PhD at Apple.** fast.ai's research output (training-efficiency results, transfer learning for text) shows that practitioner-driven groups produce research results. Its philosophy of accessible deep learning shaped how many teams onboard engineers, and its lesson structure is a reasonable template for an internal course on this pipeline.

#### [A Gentle Introduction to Graph Neural Networks (Distill, 2021)](https://distill.pub/2021/gnn-intro/)

_interactive · free · an evening · free, interactive; message passing before CS224W_

**Age 5.** An article with pictures you can play with about learning from things that are connected, like friends in a group or roads in a city.

**Middle school.** A free interactive article on Distill, a journal for clear explanations. It explains what a graph is, how information passes between connected nodes in a graph neural network, and what such networks can predict. Play with the interactive figures; they show messages moving along edges.

**High school.** Read it before lesson 10's higher levels and before CS224W: it explains message passing (each node updates from its neighbors), pooling, and the design choices (how many layers, what to aggregate) with live figures. The road graph of this project is a small instance of what it describes.

**Stanford undergrad.** The article's playground lets you change layers and aggregation and watch performance change, which is an intuition for over-smoothing and receptive field. Its taxonomy of graph tasks (node, edge, graph level) places station-level traffic prediction precisely.

**Stanford grad.** It is deliberately introductory; the formal treatment (spectral versus spatial convolutions, expressiveness) is in CS224W and the survey literature. Its value is fixing the vocabulary before those.

**PhD at Apple.** Distill's format, interactive and peer-reviewed, is the high bar for explanatory writing in ML; this article and its companion on graph convolutions are what to point a collaborator to before a graph-learning discussion.

#### [DCRNN (Li et al., ICLR 2018) and STGCN (Yu et al., IJCAI 2018)](https://arxiv.org/abs/1707.01926)

_paper · free · an evening or two · the two traffic papers; first pass now, second in phase 11_

**Age 5.** Two papers where scientists taught computers to predict traffic on real Californian roads, using the same kind of sensors we use.

**Middle school.** Two research papers that started modern traffic forecasting. DCRNN treats traffic as something that spreads along the road (diffusion) and uses a recurrent network to follow it in time. STGCN uses convolutions over time and over the road graph instead. Both are tested on PEMS-BAY, which is Bay Area PeMS data like ours, at 15, 30 and 60 minutes ahead.

**High school.** First pass: the abstract, figure 1, the results table (mean absolute error at 15, 30 and 60 minutes) and the datasets section. Note the baselines (historical average, ARIMA, vector autoregression, a plain LSTM) and how the gains grow with horizon. That table is your yardstick for the spatio-temporal network in stage 9.

**Stanford undergrad.** DCRNN: diffusion convolution with forward and backward random-walk matrices on a directed graph, GRU cells whose matrix multiplies are replaced by graph convolutions, and a sequence-to-sequence architecture with scheduled sampling. STGCN: gated temporal convolutions sandwiching Chebyshev graph convolutions, with no recurrence. Work through each equation; the plan's spatio-temporal net is a two-hop diffusion GRU without the decoder.

**Stanford grad.** Read critically: sensitivity to how the graph is built (a Gaussian kernel on road distance with a threshold), the evaluation protocol (12-step windows, masking missing values), and later critiques showing that simple baselines are competitive at short horizons. Graph WaveNet's adaptive adjacency is the natural next read.

**PhD at Apple.** These papers set the benchmark protocol the field still uses, and their limitations (fixed sensor sets, no incidents or weather, no transfer across networks) are the open problems. Reproducing them on the M1 with a reduced configuration and then adding incident and weather features to the same protocol is a publishable direction.

#### [Neural Networks and Deep Learning (Nielsen)](http://neuralnetworksanddeeplearning.com/)

_book · free · a week · free; the best written backpropagation chapter_

**Age 5.** A free online book that builds a tiny brain that reads handwritten numbers, and explains every piece as it goes.

**Middle school.** Michael Nielsen's classic free book teaches neural networks through one running example, recognizing handwritten digits, with code you can run. It explains neurons, gradient descent, backpropagation, and why deep networks are hard to train, with clear pictures. Read chapters 1 to 3 before lesson 9.

**High school.** Chapter 2 is the best written derivation of backpropagation; chapter 3 covers cross-entropy, regularization and initialization with experiments you can rerun; chapter 4 gives a visual proof that networks can approximate any function. The code is plain numpy, like lesson 9's.

**Stanford undergrad.** Nielsen's four backpropagation equations and his discussion of the vanishing-gradient problem are the background for the initialization and normalization material; his universality chapter is the intuition behind the approximation theorems.

**Stanford grad.** It predates modern architectures and optimizers, which is fine for its purpose: the foundations, explained better than most textbooks.

**PhD at Apple.** A free classic; still the recommendation for anyone who wants to understand backpropagation rather than call it.

#### [The Little Book of Deep Learning (Fleuret)](https://fleuret.org/francois/lbdl.html)

_book · free · a weekend · free; the whole field in 150 small pages_

**Age 5.** A short free book, made to fit on a phone screen, that explains deep learning from start to finish in plain language.

**Middle school.** François Fleuret's book is about 150 small pages, free, covering machine learning basics, efficient computation, training, the main model components, architectures, and applications. It is dense but readable; use it as a summary after the longer courses.

**High school.** Read the chapters on training (losses, autoregressive models, optimization), on model components (layers, attention, normalization) and on architectures after lesson 9 to consolidate; its figures are precise and its notation consistent.

**Stanford undergrad.** The book is a compact, rigorous map of the field; its chapter on efficient computation (GPUs, memory, batching) is the shortest correct account of why hardware shapes models.

**Stanford grad.** Useful as a reference for definitions and as a checklist of what a current practitioner should know; its brevity makes gaps in your own understanding visible.

**PhD at Apple.** A free, expert summary; the kind of document to hand a strong engineer who needs the whole picture in a weekend.

#### [A Recipe for Training Neural Networks (Karpathy)](https://karpathy.github.io/2019/04/25/recipe/)

_essay · free · an hour · free; the checklist that prevents silent failures_

**Age 5.** A checklist from an expert about how to train a learning machine without fooling yourself: start simple, look at the data, change one thing at a time.

**Middle school.** A blog post with a step-by-step recipe: inspect the data first, build a tiny end-to-end pipeline, get baselines right, overfit a small batch on purpose, then regularize and tune. It is short, free and practical, and it prevents most beginner mistakes.

**High school.** Apply it to the spatio-temporal net: verify the loss at initialization, overfit two batches, add complexity one piece at a time, and plot everything. Its list of things that silently go wrong (mismatched shapes, wrong learning rates, data leaks) is a debugging checklist.

**Stanford undergrad.** The post's principle, that neural network training fails silently and so must be built incrementally with verification at each step, is the engineering discipline behind every successful training run; its specific checks are cheap and catch real bugs.

**Stanford grad.** Read it with Karpathy's course on activations and gradients statistics; together they are the practical theory of training diagnostics.

**PhD at Apple.** A widely used piece of practitioner wisdom; for a lead, it is the shared protocol to require before anyone reports a result.

#### [UvA Deep Learning Tutorials](https://uvadlc-notebooks.readthedocs.io/)

_docs · free · an afternoon · free notebooks; initialization, optimization, transformers, graphs_

**Age 5.** A set of free lessons with code you can run that teach how to build learning machines with a popular tool called PyTorch.

**Middle school.** The University of Amsterdam's deep learning course notebooks, free online: PyTorch basics, activation functions, initialization, optimization, transformers, graph neural networks, autoencoders, and more. Each notebook runs and explains as it goes. They fill gaps between the theory and the code.

**High school.** The notebooks on initialization and optimization, on transformers and multi-head attention, and on graph neural networks are directly useful for lesson 9 and lesson 10's advanced levels; run them on the Mac (they are written for GPUs but work on CPU and MPS with smaller settings).

**Stanford undergrad.** Their careful experiments (comparing initializations, optimizers, normalization) are the empirical version of the training material, and their graph neural network notebook implements message passing from scratch.

**Stanford grad.** The tutorials on energy-based models, normalizing flows and autoregressive models are a bridge to the generative-model topic with working code.

**PhD at Apple.** A free, maintained set of reference implementations; useful when reproducing a paper and needing a correct baseline component.

#### [The training-recipe papers: Adam (Kingma, Ba, 2014), Batch Normalization (Ioffe, Szegedy, 2015), Dropout (Srivastava et al., 2014)](https://arxiv.org/abs/1412.6980)

_paper · free · an evening or two · free; the three tricks behind the standard recipe_

**Age 5.** Three papers that gave learning machines three tricks: a smarter way to take steps, a way to keep numbers in a comfortable range, and a way to make the machine not rely on any one part too much.

**Middle school.** Three research papers behind the standard training recipe. Adam adapts the step size for each weight; batch normalization keeps activations in a stable range so training is faster and more forgiving; dropout randomly switches off neurons during training so the network cannot depend on any one. All free; read the abstracts and the main figures.

**High school.** Read each after using the method: Adam's update rule and its bias correction, batch norm's normalization and its learned scale and shift, dropout's training and test-time behavior. The experiments show the effect sizes; note what each claims and what later work revised.

**Stanford undergrad.** Understand Adam as a combination of momentum and per-parameter scaling; batch norm's effect on optimization (smoothing the loss landscape, later analyses argue) rather than the internal-covariate-shift story; dropout as approximate model averaging. Each has known failure modes worth knowing.

**Stanford grad.** Together with later corrections (AdamW's decoupled weight decay, layer normalization for transformers, batch norm's issues at small batch size) they are the history of the training recipe; reading them teaches how empirical claims in ML get revised.

**PhD at Apple.** Canonical citations; for research, the interesting part is the gap between their stated explanations and the later understanding, which is a lesson in how the field learns.

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

## Phase 8: Deep learning across domains

_weeks 42–54 · ≈100 h · builds a headline classifier, a local LLM extractor, a profile generator_

**Goal.** Vision, language, large language models, generative models and transfer learning: the parts of deep learning the traffic data cannot carry, learned on small public data that fits the Mac mini.

### What you need to learn

- Convolutions, pooling, feature hierarchies, residual networks; classification, detection, segmentation; transfer from pretrained vision models.
- Tokens, embeddings, self-attention, transformers; encoder and decoder models; fine-tuning a text classifier.
- Large language models: pretraining, fine-tuning, preference tuning, prompting, retrieval-augmented generation, tool use; evaluation and its traps; running a quantized model locally.
- Autoencoders, variational autoencoders, GANs, diffusion models; what they are for beyond images.
- Transfer learning, self-supervised objectives, foundation models, parameter-efficient adaptation; when training from scratch still wins.

### Topic: Convolutional networks and computer vision

**Goal.** Understand how networks see: convolutions, pooling, feature hierarchies, and the tasks built on them.

**Age 5.**

A picture is a grid of tiny colored dots. A computer can slide a small window over the grid looking for edges, then corners, then eyes and wheels, and finally decide 'that is a car'.

_Try this._ Look at a photo through a paper tube. Move it around. You see edges and small parts before you can tell what the whole thing is.

**Middle school.**

A convolution slides a small filter (say 3 × 3 numbers) across the image and computes a weighted sum at each spot, so one filter finds one pattern everywhere. Stacking layers of filters finds bigger patterns; pooling shrinks the image so later layers see more of it at once.

Convolutional networks classify images, find objects with boxes, and label every pixel. They can also read a traffic camera and count cars, which is another way to measure a road.

_Try this._ Draw a 5 × 5 grid of zeros with a vertical line of ones, and slide a 3 × 3 filter [−1, 0, 1] across each row. Where does the output light up?

**High school.**

Layers: convolution (filters, stride, padding), nonlinearity, pooling, batch normalization, then fully connected layers or global pooling. Why convolutions: weight sharing and locality make them data-efficient and translation-equivariant. Classic architectures (LeNet, AlexNet, VGG, ResNet with skip connections) and what each added.

Tasks: classification, detection (boxes), segmentation (pixels); transfer learning from ImageNet-pretrained models; data augmentation. A 1-D convolution over time is the same idea and is what temporal convolutions in traffic models do.

_Try this._ Train a small convolutional network on a public digits dataset in PyTorch on MPS, then replace the classifier head and fine-tune a pretrained ResNet on a tiny dataset of your own photos.

**Stanford undergrad.**

Convolution as a linear operator with structure (Toeplitz), receptive fields, dilations, depthwise separable convolutions and their cost, normalization variants, and residual connections as a fix for optimization depth. Detection pipelines (region proposals, anchor-free methods), segmentation with encoder-decoders, and the metrics (mAP, IoU).

Vision transformers as an alternative inductive bias, and when convolutions still win (small data, efficiency). Self-supervised pretraining for vision.

_Try this._ Implement a 2-D convolution in numpy with padding and stride, verify against PyTorch, and count the multiplications for a 3 × 3 convolution on a 224 × 224 × 64 tensor.

**Stanford grad.**

Equivariance and invariance as design principles (group-equivariant networks); the interplay of architecture and data scale (ConvNets versus ViTs as data grows); efficient architectures for edge devices (MobileNet, quantization); robustness to distribution shift and adversarial examples; explaining vision models (saliency, concept methods). For transport: camera-based traffic counting and incident detection as complementary sensors.

_Try this._ Design a camera-based vehicle counter for one Fremont on-ramp: architecture, training data, evaluation, and how you would fuse it with loop-detector counts.

**PhD at Apple.**

Vision on device (phones, cameras, cars) is an engineering discipline of its own: latency and energy budgets, hardware-aware architecture search, quantization-aware training, and privacy (on-device processing of imagery). Research threads: multimodal models linking vision and language, 3-D and video understanding, and foundation models for perception.

_Try this._ Sketch a privacy-preserving on-device pipeline that turns roadside camera video into aggregate counts without storing images, and list the failure modes you would monitor.

**Check yourself.**

<details>
<summary>Why share the same filter across the whole image?</summary>

A pattern is a pattern wherever it appears; sharing makes the network data-efficient and lets it recognize a car in any corner.

</details>

<details>
<summary>What does pooling buy?</summary>

A larger receptive field for later layers and some tolerance to small shifts, at the cost of resolution.

</details>

### Topic: Language: tokens, embeddings, attention and transformers

**Goal.** Understand how text becomes numbers and how attention lets a model relate every word to every other.

**Age 5.**

A computer cannot read letters, so it turns words into lists of numbers. Words that mean similar things get similar numbers.

Then, to understand a sentence, it lets each word look at the other words and decide which ones matter for it. That looking is called attention.

_Try this._ In the sentence 'the road near the school was slow because of a crash', which words tell you why it was slow? Point at them. That is attention.

**Middle school.**

Tokens are pieces of text (words or word parts); each token gets an embedding, a vector learned so that related tokens sit near each other. Attention computes, for each token, how much to weigh every other token, then mixes their vectors accordingly. A transformer stacks attention and small networks many times.

The same machinery reads 511 incident headlines: 'Injury collision on I-880 N' can be turned into features, or classified by severity, by a small language model.

_Try this._ Take three incident headlines and list which words in each would help decide whether it is serious. Then guess which words a model would attend to.

**High school.**

Tokenization (byte-pair encoding), embeddings (word2vec's idea: predict neighbors), positional information, self-attention as softmax(QKᵀ/√d)V, multi-head attention, feed-forward blocks, residuals and layer normalization. Encoder models (BERT) for understanding, decoder models (GPT) for generation.

Older sequence models (RNNs, LSTMs) processed words in order and struggled with long range; attention sees everything at once, which is why transformers won. Evaluation: perplexity for language modelling, accuracy or F1 for classification.

_Try this._ Implement single-head self-attention in numpy for a sentence of six tokens with random embeddings, print the attention matrix, and explain what a row means.

**Stanford undergrad.**

The transformer in full: scaled dot-product attention and its quadratic cost, causal masking, positional encodings (sinusoidal, learned, rotary), training with teacher forcing, and the compute and memory of a layer. Pretraining objectives (masked language modelling, next-token prediction), fine-tuning for classification, and embeddings from encoders for retrieval.

Practical NLP: text preprocessing pitfalls, class imbalance in text classification, and using a small pretrained encoder on the 511 headlines for severity when the field is missing.

_Try this._ Fine-tune a small pretrained encoder to predict incident severity from headlines and compare with a bag-of-words logistic regression; report macro F1 and calibration.

**Stanford grad.**

Efficient attention (linear, sparse, flash), long-context methods, mixture of experts, and the interpretability of attention (attention is not explanation). Multilingual and multimodal models. Sequence models for non-text data: transformers over time steps and over stations in spatio-temporal forecasting, and their inductive-bias trade-offs against graph convolutions.

_Try this._ Replace the GRU in the spatio-temporal net with a small temporal transformer and the neighbor mixing with spatial attention; compare validation AP and training time on the M1.

**PhD at Apple.**

Language models are now general-purpose components: embeddings for retrieval, classifiers with no labels, and interfaces to systems. The research frontier spans scaling, alignment, efficiency, and evaluation. For a transport product the relevant questions are extracting structured facts from free-text reports reliably and cheaply, and doing it on device where privacy demands.

_Try this._ Design an evaluation for a model that extracts route, direction and severity from free-text incident reports, including how you would build a labelled set and measure calibration of its confidence.

**Check yourself.**

<details>
<summary>What is an embedding?</summary>

A learned vector for a token, arranged so that tokens used in similar ways end up near each other.

</details>

<details>
<summary>Why did attention replace recurrence?</summary>

It relates every position to every other in one step, trains in parallel, and handles long-range dependencies that recurrent models forget.

</details>

### Topic: Large language models: pretraining, fine-tuning, prompting, retrieval and evaluation

**Goal.** Know what a large language model is, how it is trained and adapted, how to use one responsibly in a system, and how to evaluate it.

**Age 5.**

A large language model has read a huge amount of writing and learned to guess the next word so well that it can answer questions and write stories.

It is not always right, so people check its answers, and they give it the right documents to read when they want facts.

_Try this._ Ask an adult to ask a chatbot what time it is. It probably cannot know. Why not? Because it has no clock unless someone gives it one.

**Middle school.**

Pretraining: the model reads billions of words and learns to predict the next token. Fine-tuning: it is trained a little more on examples of the behavior you want. Prompting: you write instructions and examples in the input. Retrieval-augmented generation: you search your documents and put the relevant ones in the prompt so the model answers from them.

Models can be confidently wrong (hallucinate), so evaluation matters: test on questions with known answers and measure how often the model is right.

_Try this._ Write a prompt asking a model to turn 'Injury collision on I-880 N at Mission Blvd' into route, direction and location fields. Try three headlines. Where does it guess?

**High school.**

The training pipeline: pretraining on web-scale text, supervised fine-tuning on demonstrations, preference tuning (reinforcement learning from human feedback or direct preference optimization). Scaling laws: loss falls predictably with model size, data and compute. Inference: sampling, temperature, context windows, cost per token.

Adaptation: prompting (instructions, few-shot examples, structured output), parameter-efficient fine-tuning (LoRA), retrieval (embeddings, a vector index, chunking), and tool use. Evaluation: held-out benchmarks, task-specific test sets, and human review. On a 16 GB Mac mini, a quantized 3 to 8 billion-parameter model runs locally but competes with training for memory.

_Try this._ Run a small quantized open model locally, ask it to extract structured fields from 50 incident headlines, and measure exact-match accuracy against fields you labelled by hand.

**Stanford undergrad.**

Architecture and training at scale: decoder-only transformers, tokenizer design, data curation and deduplication, mixed precision, distributed training. Alignment methods and their failure modes; hallucination as a calibration problem; retrieval as grounding. Evaluation science: contamination, benchmark saturation, LLM-as-judge and its biases, and why task-specific evaluation sets beat leaderboards.

Systems: latency and cost, caching, batching, structured decoding, guardrails, and observability for model calls. For this project: an LLM could parse free-text reports or draft the weekly monitoring summary, never decide alerts.

_Try this._ Build a small retrieval-augmented question answerer over the repository's documentation with a local embedding model, and write a 30-question evaluation set with exact answers to score it.

**Stanford grad.**

Research threads: scaling and data efficiency, reasoning and tool use, long context, interpretability (probing, circuits), safety and alignment, efficient inference (quantization, speculative decoding, distillation), and evaluation methodology. The open scientific question of what capabilities emerge from next-token prediction and why.

_Try this._ Design an experiment measuring whether an LLM's extracted incident severity improves the traffic model's average precision over the raw feed's field, controlling for the cases where the feed's field is missing.

**PhD at Apple.**

Deploying LLMs in products is a discipline of evaluation, cost and risk: task-specific evals with regression tests, human review loops, privacy of prompts, on-device versus server inference, and clear boundaries on what the model may decide. The research frontier includes reliable structured extraction, calibrated abstention, and agents that act through tools with verifiable behavior.

_Try this._ Write the evaluation and safety plan for using an on-device language model to summarize the weekly monitoring report for a traffic operations team, including what it must never assert.

**Check yourself.**

<details>
<summary>What is retrieval-augmented generation for?</summary>

Grounding answers in your documents so the model reads facts instead of recalling them, which cuts hallucination and keeps knowledge current.

</details>

<details>
<summary>Why evaluate on your own test set rather than a public benchmark?</summary>

Public benchmarks may be in the training data and rarely match your task; a held-out set of your own questions measures what you will ship.

</details>

### Topic: Generative models: autoencoders, variational autoencoders, GANs and diffusion

**Goal.** Understand models that learn to produce data, the ideas behind each family, and what they are for beyond pictures.

**Age 5.**

Some learning machines do not sort things; they make new things, like drawing a car that has never existed, by learning what cars usually look like.

_Try this._ Draw a car from memory. You are using what you learned about cars to make a new one. A generative model does the same with numbers.

**Middle school.**

An autoencoder squeezes data through a narrow middle and learns to rebuild it; the middle becomes a compact summary. A variational autoencoder makes that middle a probability so you can sample new data. A GAN trains a forger against a detective until the forgeries look real. A diffusion model learns to remove noise step by step, so starting from pure noise it can produce a realistic sample.

For traffic, generative models can produce realistic fake days for testing or fill in missing sensor data.

_Try this._ Take a daily speed profile, add random noise to it, and try to guess the original. That guessing, repeated many times, is what a diffusion model learns.

**High school.**

Autoencoders and reconstruction error as an anomaly score; VAEs with a latent Gaussian and a loss of reconstruction plus a KL term; GANs with a generator and discriminator and their training instability; diffusion models as learned denoising over many steps with a simple noise-prediction loss. Evaluation is hard: likelihood is not quality; sample-quality scores exist for images.

Uses: synthetic data, imputation, compression, and learning representations that transfer.

_Try this._ Train a small autoencoder on daily speed profiles in PyTorch; use reconstruction error to rank days by strangeness and inspect the top five.

**Stanford undergrad.**

The ELBO and amortized variational inference behind VAEs; the reparameterization trick; posterior collapse. GANs as a minimax game, mode collapse, Wasserstein formulations. Diffusion: forward noising process, reverse process, score matching, the connection to stochastic differential equations, classifier-free guidance. Normalizing flows as exact-likelihood models.

Time-series generation and imputation with these families; conditional generation (a day given weather and calendar) as a simulator for evaluation.

_Try this._ Derive the VAE loss from the ELBO and implement a conditional VAE that generates daily profiles given weekday and rain; compare generated and real distributions of the 5 pm speed.

**Stanford grad.**

Diffusion models' dominance in images and their extension to audio, video, molecules and time series; latent diffusion for efficiency; evaluation via distribution metrics and downstream utility. Generative models for scientific simulation and for data augmentation under distribution shift. Privacy risks (memorization) and the synthetic-data trade-offs.

_Try this._ Train a small diffusion model on daily station profiles and test whether augmenting training with its samples improves the classifier's average precision on rare rainy-day episodes.

**PhD at Apple.**

Generative modelling is now infrastructure for media, science and simulation. The research frontier includes fast sampling, controllability, evaluation, and world models that couple generation with dynamics. For transport, generative simulators of traffic conditioned on interventions are a path to counterfactual evaluation of alerts and routing.

_Try this._ Propose a generative traffic simulator conditioned on incidents and weather, trained on PeMS history, and the validation that would let you trust its counterfactuals.

**Check yourself.**

<details>
<summary>What does the KL term in a VAE do?</summary>

It keeps the learned latent distribution close to a simple prior so that sampling from the prior produces sensible outputs.

</details>

<details>
<summary>Why is likelihood a poor measure of sample quality?</summary>

A model can assign high likelihood to data while producing blurry or degenerate samples; likelihood and perceptual quality measure different things.

</details>

### Topic: Transfer learning, self-supervision and foundation models

**Goal.** Reuse what a model learned elsewhere, learn from unlabeled data, and know when a pretrained model beats training from scratch.

**Age 5.**

If you already know how to ride a bicycle, learning a scooter is faster. Learning machines can also start from something they already learned instead of starting from zero.

_Try this._ Name one thing you learned that made a new thing easier. That is transfer.

**Middle school.**

Transfer learning takes a model trained on a big dataset and adapts it to a small one: keep the early layers, retrain the last part. Self-supervised learning makes its own labels from unlabeled data, like predicting a hidden word or a missing patch, so it can learn from everything. Foundation models are big models pretrained this way that many tasks reuse.

For traffic, a model pretrained on all of California's PeMS could be fine-tuned for Fremont with less data.

_Try this._ Cover part of a sentence with your hand and guess the missing word. You just made a label with no teacher.

**High school.**

Feature extraction versus fine-tuning; learning rates for pretrained layers; catastrophic forgetting. Self-supervised objectives: masked prediction, contrastive learning (pull similar views together), next-step prediction. Foundation models in vision, language and now time series; zero-shot and few-shot use.

Practical rule: with little labelled data, start pretrained; with lots of in-domain data and a narrow task, training from scratch on tabular features still wins, which is why LightGBM is this project's production model.

_Try this._ Pretrain a small network on all Fremont stations with a masked-step objective, then fine-tune it on one station's classification task and compare with training that station's model from scratch on the same data.

**Stanford undergrad.**

Why transfer works (shared low-level features, representations as inductive bias), negative transfer, domain adaptation and domain shift; contrastive learning theory (InfoNCE as a mutual-information bound); masked autoencoders; scaling behavior of pretraining. Time-series foundation models and their mixed evidence against tuned task-specific models.

Evaluating transfer honestly: matched compute and data budgets, and the leakage risk when pretraining data overlaps the test period.

_Try this._ Implement a contrastive pretraining on station-day windows (two augmented views of the same window as positives), then a linear probe for the slow-episode label; report linear-probe AP against raw features.

**Stanford grad.**

Representation quality and its measurement (probes, transfer suites), the role of data curation in pretraining, parameter-efficient adaptation (adapters, LoRA), continual pretraining, and model merging. Foundation models for spatio-temporal data and the open question of whether traffic dynamics transfer across cities.

_Try this._ Design an experiment testing whether pretraining on PeMS District 4 improves 60-minute skill in Fremont, with a leakage-safe split in time and space.

**PhD at Apple.**

The economics of pretraining (one large cost amortized over many tasks) reshaped ML organizations; the research frontier includes what pretraining learns, how to adapt safely, and how to evaluate foundation models on tasks they were not built for. For transport, a state-wide pretrained traffic model served to many cities is a plausible product and a research program.

_Try this._ Write the research plan for a California-wide traffic foundation model: data, objective, evaluation across cities, and the failure modes you would expect.

**Check yourself.**

<details>
<summary>When does training from scratch beat a pretrained model?</summary>

With plenty of in-domain data and a narrow task, especially tabular problems where boosted trees on engineered features remain strongest.

</details>

<details>
<summary>What makes self-supervision possible?</summary>

The data contain their own supervision: hide part of an input and predict it from the rest.

</details>

### Read and watch, explained at every level

#### [Understanding Deep Learning (Prince)](https://udlbook.github.io/udlbook/)

_book · free · reference, chapters as needed · now chapters 10 to 18: convolutions, transformers, graphs, generative models_

**Age 5.** A newer big book about deep learning, free to read, with lots of clear pictures that show what each part of a network does.

**Middle school.** Simon Prince's textbook (2023), free as a PDF with notebooks. It covers supervised learning, networks, training, measuring performance, regularization, convolutional and recurrent and transformer networks, graph networks, and generative models. It is more modern and more visual than the older Deep Learning book. Read the early chapters after phase 6.

**High school.** Chapters 1 to 9 are the modern version of lesson 9: networks, loss functions, fitting, gradients and initialization, measuring performance, and regularization, each with figures that make the idea visible. Chapter 13 on graph neural networks covers the neighbor mixing of the spatio-temporal net. The notebooks let you run every figure.

**Stanford undergrad.** Prince's treatment of why deep networks generalize, of the loss landscape, and of the transformer architecture is the clearest current textbook account, and it includes the recent empirical findings (double descent, scaling) that Goodfellow predates. Chapter 12 on transformers is the reading before lesson 8's attention material.

**Stanford grad.** The chapters on generative models (VAEs, GANs, normalizing flows, diffusion) are a compact, correct introduction with the math intact, and the chapter on ethics is unusually careful for a technical text. Use it as the reference when reading current papers.

**PhD at Apple.** A free, modern, rigorous textbook is a public good; it is the one to recommend to a new graduate student and the one to cite for standard architectures in the current idiom. Its figures are also the model for how to draw a network in a paper.

#### [Dive into Deep Learning (d2l.ai), chapters 3–5 and 9–10](https://d2l.ai/)

_book · free · reference, chapters as needed · now chapters 7, 8 and 11: convolutions and attention_

**Age 5.** A free online textbook where every idea comes with code you can run, about teaching computers using layers of simple math.

**Middle school.** Chapters 3 to 5 cover linear and softmax regression and multilayer perceptrons with PyTorch code you can execute. Chapters 9 and 10 cover recurrent networks (RNN, GRU, LSTM) and how to train them well. Run the notebooks; each section ends with exercises that are worth doing.

**High school.** Chapter 5's multilayer perceptron from scratch mirrors lesson 9's numpy network; chapter 9 explains why GRUs can remember and forget, which is the sequence half of the spatio-temporal network. The exercises about changing hidden size and learning rate are the experiments to actually run on the traffic data.

**Stanford undergrad.** d2l's coverage of optimization (chapter 12), attention (chapter 11) and computational performance (chapter 13) is deeper than most books and free. Its consistent code style makes it easy to lift components, and its treatment of truncated backpropagation through time explains what a recurrent model can and cannot learn from a 12-step window.

**Stanford grad.** Its graph-neural-network coverage is thin; use CS224W for that. Its attention chapters are the bridge to spatio-temporal transformers, and its chapter on computational performance (asynchrony, memory, multi-device) is the right background before profiling on unified memory.

**PhD at Apple.** A living, open textbook with code in several frameworks is itself a research-infrastructure achievement. Cite it for standard architectures, use its implementations as reference points when reproducing papers, and note its authors' choice to include the messy practical chapters most textbooks omit.

#### [CS231n: Convolutional Neural Networks for Visual Recognition, course notes](https://cs231n.github.io/)

_course · free · weeks · free; the convolutional foundations with worked counts_

**Age 5.** The notes from a famous class that taught computers to see, with drawings of how a picture becomes numbers and then a label.

**Middle school.** Stanford's computer vision course notes are free online: image classification, linear classifiers, optimization, backpropagation, neural networks, convolutional networks, and how to train them. The notes are written for beginners with some Python, and the assignments build a network from scratch.

**High school.** Read the convolutional networks module: filters, stride, padding, pooling, and the famous architectures, with the parameter and computation counts worked out. The backpropagation and neural-network modules are a good second pass over the traffic course's lesson 9.

**Stanford undergrad.** The notes' treatment of initialization, batch normalization, dropout and learning-rate schedules is compact and correct, and the assignment code (numpy implementations of layers) is the vision analogue of the from-scratch rule. The lecture videos add transfer learning, detection and segmentation.

**Stanford grad.** The course predates vision transformers and self-supervised vision; read it for the convolutional foundations, then the recent literature for the rest. Its assignment design is a model for teaching by implementation.

**PhD at Apple.** CS231n shaped how a generation learned deep learning; knowing it tells you what practitioners take for granted. For a lead, its assignments are a fair calibration of what 'can implement a network from scratch' means.

#### [The Illustrated Transformer (Alammar)](https://jalammar.github.io/illustrated-transformer/)

_essay · free · an hour · free; read before the paper_

**Age 5.** A web page with drawings that shows how a computer reads a sentence by paying attention to the words that matter, a bit like how you look at the important parts of a picture.

**Middle school.** Jay Alammar's blog post explains the transformer, the model behind modern language models, with step-by-step pictures: how words become numbers, how attention lets each word look at the others, and how layers stack. Free, and readable without much math. Read it before any paper on transformers.

**High school.** Follow the pictures of self-attention: queries, keys and values, the dot-product scores, the softmax, and the weighted sum. Then multi-head attention and the encoder-decoder structure. When you can redraw the attention diagram from memory, read Attention Is All You Need with the post open beside it.

**Stanford undergrad.** The post is the intuition layer for the attention equation softmax(QKᵀ/√d)V; pair it with a from-scratch implementation (Karpathy's GPT lesson) so the pictures become code. Its companion posts on GPT-2 and on BERT extend the same method to decoder-only and encoder-only models.

**Stanford grad.** It omits positional encodings' subtleties, training dynamics and scaling; those come from the papers and from Prince's chapter 12. For spatio-temporal traffic models, the relevant transfer is attention over stations (spatial attention) and over time steps, which later traffic transformers use.

**PhD at Apple.** Alammar's posts are the standard first explanation across the industry; for briefing non-specialists on what a language model does, they remain the reference, and their visual grammar is worth adopting in talks.

#### [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)

_paper · free · an evening or two · free; the paper, with the post open beside it_

**Age 5.** The paper that introduced the transformer, the design inside today's language models, by showing that paying attention to all words at once works better than reading them one by one.

**Middle school.** A research paper from 2017 that proposed the transformer for translating languages. Its main idea is self-attention, where every word looks at every other word and decides what matters. It replaced the older recurrent networks and became the base of modern language models. Read the introduction and look at the architecture figure after the Illustrated Transformer.

**High school.** Read it with the Illustrated Transformer open: the attention equation, multi-head attention, positional encodings, the encoder-decoder layout and the training setup. The results table compares with the recurrent baselines; note the compute cost column.

**Stanford undergrad.** Work through the model dimensions and parameter counts, the reason for scaling by the square root of the key size, the role of residuals and layer normalization, and the label smoothing and learning-rate warmup in training. Then implement a single encoder layer from the paper in PyTorch.

**Stanford grad.** The paper's details (sinusoidal positions, dropout placement) have been revised by later work; understand what survived (attention, residual stacks) and what changed (pre-norm, rotary positions, decoder-only). Its influence is the case study in how an architecture spreads.

**PhD at Apple.** A canonical citation; for research, the interesting questions are the ones it opened (efficiency of attention, positional information, what attention learns) rather than what it settled.

#### [Hugging Face LLM course](https://huggingface.co/learn/llm-course/chapter1/1)

_course · free · two weeks · free; the practical stack for language models_

**Age 5.** A free online class that shows how to use ready-made language models, like borrowing a very well-read robot and teaching it your job.

**Middle school.** A free course from Hugging Face, the company behind the most-used library for language models. Chapters 1 to 4 explain how transformer models work and how to download, run and fine-tune one; later chapters cover datasets, tokenizers and common language tasks. Everything runs in notebooks.

**High school.** Do chapters 1 to 3 (transformers, using pretrained models, fine-tuning a classifier) and apply chapter 3 to the 511 headlines: fine-tune a small encoder to predict severity. Chapter 6 on tokenizers explains why 'I-880' may split oddly and how to handle it.

**Stanford undergrad.** The course teaches the practical stack (models, datasets, tokenizers, training loops with the accelerate library) that most applied NLP uses; understand what the abstractions hide (padding, attention masks, learning-rate schedules) by reading the code of one training step.

**Stanford grad.** Later chapters and the companion courses (agents, reinforcement learning from human feedback) track current practice; the course is a good index of what the ecosystem considers standard at any moment.

**PhD at Apple.** The Hugging Face ecosystem is where much of applied NLP lives; its course is also a reference for how open model sharing and evaluation are organized, which matters for reproducibility and licensing decisions in a product.

#### [Speech and Language Processing, 3rd edition draft (Jurafsky, Martin)](https://web.stanford.edu/~jurafsky/slp3/)

_book · free · reference · free; the textbook for the language chapters_

**Age 5.** A big free book about how computers handle language, from letters and words to whole conversations.

**Middle school.** The standard textbook on natural language processing, free online as a draft that the authors update. It covers text processing, n-gram models, embeddings, neural networks for text, transformers and large language models, and speech. Read chapters as references when a topic comes up.

**High school.** The chapters on tokenization and regular expressions, on vector semantics and embeddings, on transformers and on large language models are the textbook version of the language topic. Each ends with exercises; the embeddings chapter's exercises can be done on the incident headlines.

**Stanford undergrad.** Jurafsky and Martin give the linguistic and statistical background (n-gram language models, smoothing, evaluation with perplexity) that makes modern models intelligible as a continuation rather than a break. The chapters on fine-tuning and on alignment (including direct preference optimization) are current.

**Stanford grad.** Its breadth (parsing, semantics, dialogue, speech) is the map of the field beyond language modelling, useful when a product needs structured extraction or dialogue rather than generation.

**PhD at Apple.** A free, continuously revised textbook by leading researchers is the field's reference; its revision history is itself a record of how fast the field moved.

#### [LLMs-from-scratch repository (Raschka)](https://github.com/rasbt/LLMs-from-scratch)

_code · free · two weeks · free notebooks; a GPT you built yourself_

**Age 5.** A free set of notebooks that build a small language model step by step in code, so you can see every part instead of trusting a magic box.

**Middle school.** Sebastian Raschka's public repository walks through implementing a GPT-style model in PyTorch: the tokenizer, attention, transformer blocks, pretraining on text, and fine-tuning for classification and instruction following. The notebooks are free and complete; the companion book adds prose but is not required.

**High school.** Work through the notebooks in order and keep the model tiny so training takes minutes on the Mac mini. Type the code rather than copying it. When you finish, the Hugging Face abstractions will be transparent.

**Stanford undergrad.** The value is completeness: every tensor shape is shown, attention is built stepwise, and the training loop is explicit. Pair with the original papers and with Karpathy's GPT lesson for a second perspective on the same architecture.

**Stanford grad.** It stops at small scale; the scaling questions (data curation, distributed training, evaluation) are in the ultra-scale playbook and CS336. The instruction-tuning notebook is a compact account of supervised fine-tuning.

**PhD at Apple.** A from-scratch implementation is the fastest route to reading language-model papers critically; for a lead, assigning the repository to an engineer is a reliable way to build a real mental model before they touch a production model.

#### [Intro to Large Language Models (Karpathy, video)](https://www.youtube.com/watch?v=zjkBMFhNj_g)

_video · free · one hour · free; one hour, the whole picture_

**Age 5.** A one-hour talk that explains, with simple drawings, what a large language model is, how it is trained, and what it can and cannot do.

**Middle school.** Andrej Karpathy's free talk for a general audience: a model is two files (weights and code), pretraining compresses the internet into weights, fine-tuning teaches an assistant style, and scaling makes it better in predictable ways. It ends with what is coming and with security concerns. Watch it before the language-model topic.

**High school.** The talk's framing (base model, assistant model, the training stages, scaling laws, tool use, and the jailbreak and injection risks) is the outline of the language-model topic; watch it, then read the topic's high-school level and notice what the talk simplified.

**Stanford undergrad.** It is a survey, not a course; its value is the clear separation of pretraining, fine-tuning and preference tuning, and the honest treatment of security. Follow with the from-scratch repository for the mechanics.

**Stanford grad.** Karpathy's later talks and his from-scratch series are the technical continuation; this one is the shared vocabulary for a mixed audience.

**PhD at Apple.** The talk to send to a stakeholder before a discussion about using a language model in a product; its security section is the part most people skip and should not.

#### [Stanford CS224N: Natural Language Processing with Deep Learning](https://web.stanford.edu/class/cs224n/)

_course · free · a quarter · free lectures; embeddings to transformers, rigorously_

**Age 5.** A university class, free to watch, about teaching computers to read and write, from single words to whole documents.

**Middle school.** Stanford's course, with free lecture videos and notes: word vectors, neural networks for text, recurrent models, attention and transformers, pretraining, and the current state of language models. Watch the lectures on word vectors and on transformers during the language topic.

**High school.** Lectures 1 and 2 (word vectors) explain embeddings from first principles; the lectures on transformers and on pretraining are the theory behind the Hugging Face course's practice. The assignments (word2vec from scratch, a dependency parser, a translation model) are demanding and worth one.

**Stanford undergrad.** The course's treatment of word2vec's objective, of attention's derivation, and of pretraining objectives is rigorous; its later lectures on evaluation, on multilingual models and on ethics broaden the language topic.

**Stanford grad.** CS224N's evolution across years is a record of the field; its final lectures on current research are a fair map of open problems.

**PhD at Apple.** The standard graduate NLP course; knowing it lets you calibrate what a new hire in language modelling can be assumed to know.

#### [Hugging Face Computer Vision course](https://huggingface.co/learn/computer-vision-course/unit0/welcome/welcome)

_course · free · weeks · free; current vision practice with notebooks_

**Age 5.** A free online class about teaching computers to see, with code you can run and pictures to try it on.

**Middle school.** A community-written free course from Hugging Face covering image basics, convolutional networks, vision transformers, object detection, segmentation, video, and multimodal models, with notebooks. Do the units on convolutional networks and on transfer learning during the vision topic.

**High school.** Units on the fundamentals of computer vision, on convolutional networks, on vision transformers and on transfer learning give working code for the vision exercises; the unit on model optimization for deployment previews on-device constraints.

**Stanford undergrad.** It complements CS231n (foundations) with current practice (pretrained backbones, the transformers library for vision, evaluation on standard benchmarks); its multimodal units are a bridge to vision-language models.

**Stanford grad.** Community courses vary in depth; this one is strongest on practice and on the current model zoo, weakest on theory. Use it accordingly.

**PhD at Apple.** A free, maintained map of practical vision tooling; useful for calibrating what a modern vision baseline is before proposing anything custom.

#### [The generative-model papers: VAE (Kingma, Welling, 2013), GAN (Goodfellow et al., 2014), DDPM (Ho, Jain, Abbeel, 2020)](https://arxiv.org/abs/1312.6114)

_paper · free · an evening or two · free; the three originals_

**Age 5.** Three papers, years apart, each a new way for a computer to make new pictures: one squeezes and rebuilds, one plays forger against detective, one cleans up noise step by step.

**Middle school.** The three papers that defined generative modelling: the variational autoencoder, the generative adversarial network, and denoising diffusion. All free. Read the introductions and look at the sample figures; then read Lilian Weng's diffusion post for the connections.

**High school.** For each: the model, the training objective, and the samples. The VAE's loss has two parts (reconstruction and a KL term); the GAN is a two-player game; the diffusion model learns to predict noise. Compare their samples and note the years.

**Stanford undergrad.** Work through the derivations: the evidence lower bound and the reparameterization trick; the minimax objective and its optimum; the diffusion forward process and the simplified noise-prediction loss. Each paper's experiments show the evaluation problem of generative models.

**Stanford grad.** The later literature (Wasserstein GANs, hierarchical VAEs, score-based models, latent diffusion) builds directly on these; reading the originals shows what each solved and what it left.

**PhD at Apple.** Canonical citations; the interesting research questions are those they opened: evaluation, controllability, and the unification of the three families through score matching.

#### [Self-Supervised Representation Learning (Weng, Lil'Log)](https://lilianweng.github.io/posts/2019-11-10-self-supervised/)

_essay · free · an hour · free; the pretraining objectives, surveyed_

**Age 5.** A long web post about how a computer can learn from pictures and text without anyone labelling them, by hiding parts and guessing them.

**Middle school.** Lilian Weng's survey of self-supervised learning: pretext tasks in vision, contrastive learning, and language modelling as self-supervision. Free, long, with figures for each method. Read the introduction and the contrastive section before the transfer-learning topic.

**High school.** Its sections on contrastive learning (pull together two views of the same image, push apart others) and on masked prediction explain the objectives behind modern pretraining; apply the idea to station-day windows as the topic's exercise suggests.

**Stanford undergrad.** The post is a curated map of the literature with the objectives written out; use it to choose a pretext task for the traffic data and to understand the InfoNCE objective's connection to mutual information.

**Stanford grad.** Weng's companion posts on contrastive learning and on diffusion models complete the picture; her surveys are reliable starting bibliographies.

**PhD at Apple.** A well-maintained survey; the practical entry to the pretraining literature before the papers.

#### [Stanford CS336: Language Modeling from Scratch](https://cs336.stanford.edu/)

_course · free · a quarter · free lectures and assignments; how models are really built_

**Age 5.** A university class where students build a language model from the very bottom, including collecting and cleaning the text it learns from.

**Middle school.** A Stanford course whose lectures and assignments are public. Students implement everything: the tokenizer, the model, the training system, data cleaning and evaluation. It is demanding and current, and its assignment write-ups are worth reading even if you do not do them all.

**High school.** Watch the lectures on tokenization, on the transformer and its variants, and on data. The assignments show how much of a language model is engineering (systems, data pipelines) rather than architecture.

**Stanford undergrad.** The course's systems assignments (efficient attention kernels, distributed training, mixed precision) are the practical version of the scaling topic; its data assignment (filtering, deduplication) is the part most courses skip and most products get wrong.

**Stanford grad.** Compare its curriculum with the ultra-scale playbook and with the papers on data curation; together they are the current state of the art in how language models are built, at a level that lets you read the industrial reports.

**PhD at Apple.** The course is a snapshot of what a research lab considers foundational in 2026; for a lead, its assignments define what an engineer joining an LLM team should be able to do.

#### [What are Diffusion Models? (Weng, Lil'Log)](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)

_essay · free · an hour · free; the generative-model survey to read first_

**Age 5.** A long web post that explains how a computer can draw a picture by starting from noise and cleaning it up bit by bit.

**Middle school.** Lilian Weng's blog post explains diffusion models, the method behind modern image generators: add noise to data step by step, train a network to undo one step, then generate by starting from noise. It compares diffusion with GANs and VAEs. Free; it is long and mathematical in places, so read it in two sittings.

**High school.** Read the first sections (the forward and reverse processes and the training objective) and skim the later ones on speeding up sampling and on conditioning. Then implement a tiny diffusion model on one-dimensional data (daily speed profiles) to see the idea work.

**Stanford undergrad.** The post derives the variational objective, the simplification to noise prediction, and the connection to score matching; it is a faithful and compact version of the original papers. The later sections on classifier-free guidance and on fast samplers track the field.

**Stanford grad.** Weng's other posts (on attention, on self-supervised learning, on prompt engineering and on agents) are similarly careful surveys; her blog is a reliable index of a topic's literature.

**PhD at Apple.** A well-maintained survey post is often more useful than a review paper because it is revised; cite the papers, but read the post first.

#### [Practical Deep Learning for Coders (fast.ai)](https://course.fast.ai/)

_course · free · eight weeks · the vision and text lessons, with transfer learning throughout_

**Age 5.** A free class where you train a real neural network on the very first day and only later learn all the details, like learning to ride a bike before studying how it works.

**Middle school.** Jeremy Howard's course is top-down: you build working models first (on images, text and tables) and learn the theory as you need it. It uses the fastai library built on PyTorch. It is good for confidence and full of practical tricks that make training work.

**High school.** Its tabular lessons (random forests first, then neural networks on tables) and its training tricks (a learning-rate finder, the one-cycle schedule) are directly usable in stage 9. The lesson on embeddings for categorical variables shows how to give station identity to a network, which is how the spatio-temporal net could learn per-station behavior.

**Stanford undergrad.** The course's emphasis on iterating quickly and inspecting data is a counterweight to theory-first study, and its stated view that tree ensembles usually win on tabular data matches the evidence and this project's production choice. Its chapter on data ethics is short and worth the time for anyone whose model will issue alerts to drivers.

**Stanford grad.** The fastai library's abstractions can hide details you need for a custom model such as the spatio-temporal network. Learn the ideas from the course, then implement in plain PyTorch. The course's lessons on collaborative filtering and on tabular embeddings are the clearest introduction to learned representations for categorical data.

**PhD at Apple.** fast.ai's research output (training-efficiency results, transfer learning for text) shows that practitioner-driven groups produce research results. Its philosophy of accessible deep learning shaped how many teams onboard engineers, and its lesson structure is a reasonable template for an internal course on this pipeline.

### Build

- Train a small convolutional network on a public digits dataset on MPS, then fine-tune a pretrained ResNet on a tiny set of your own photos.
- Fine-tune a small pretrained encoder to predict 511 incident severity from headlines; compare with a bag-of-words logistic regression.
- Run a quantized open language model locally, extract route, direction and severity from 50 headlines, and score exact match against hand labels; build a small retrieval-augmented question answerer over the repository's docs with a 30-question eval set.
- Train an autoencoder on daily speed profiles and rank days by reconstruction error; train a conditional VAE that generates profiles given weekday and rain.
- Pretrain a masked-step model on all Fremont stations and compare a linear probe with raw features on the slow-episode label.

### Done when

- A headline severity classifier with macro F1 and calibration next to the bag-of-words baseline.
- An extraction accuracy number for the local LLM with an error analysis of its worst ten cases.
- A generated-profile plot next to real ones, with a distribution comparison of the 5 pm speed.

### Pitfalls

- Fine-tuning with a learning rate meant for training from scratch.
- Trusting an LLM's output without an eval set.
- Judging generative samples by eye alone.

## Phase 9: ML engineering and scale

_weeks 52–62 · ≈70 h · builds stages 12 and 13, a containerized service_

**Goal.** The pipeline runs itself on the Mac mini, ships as a service that could run anywhere, and you know what changes at scale.

### What you need to learn

- Reproducibility: seeds, pinned versions, a Makefile, one command from raw to report; the one-feature-module rule and train-serve skew.
- Testing ML code with fixtures and golden files; experiment tracking; a model registry as a folder with a manifest; rollback.
- Scheduling with launchd; logging and alerting; monitoring: drift with PSI, performance decay, retraining policy.
- APIs with FastAPI, containers with Docker, cloud basics; batch versus online serving; service contracts and runbooks.
- Why GPUs are fast; data, tensor and pipeline parallelism; mixed precision; profiling; scaling laws.

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

### Topic: APIs, containers and the cloud: serving a model beyond one Mac

**Goal.** Wrap a model in a service, package it so it runs anywhere, and know what changes when it leaves the Mac mini.

**Age 5.**

Right now the helper lives on one small computer. To let a phone app ask it questions, we give it a door with a doorbell: the app rings, asks 'is I-880 slow?', and the helper answers.

A container is a lunchbox with everything the helper needs packed inside, so it works on any computer.

_Try this._ Play doorbell: one person is the app, one is the helper. The app asks a question at the door and the helper answers. That is an API.

**Middle school.**

An API is an agreed way for programs to talk: the app sends a request (station, horizon) and gets a JSON answer (probability). A web framework such as FastAPI turns a Python function into that door. A container (Docker) packages the code, Python and libraries into one image that runs the same everywhere. The cloud is renting computers by the hour to run it.

_Try this._ Write a Python function `predict(station_id, horizon)` that returns the latest probability from `predictions_latest.json`. That function is the whole API before the door is added.

**High school.**

FastAPI: define request and response models, add an endpoint, run with a server, read the auto-generated docs. Docker: a Dockerfile lists the base image and install steps, `docker build` and `docker run` package and run it, `.dockerignore` keeps data out. Cloud basics: a virtual machine, object storage for files, a scheduler, secrets management, and logs. Batch versus online serving: this project's 5-minute batch writes a file; an online service answers on demand.

Security basics: never expose secrets, validate inputs, rate-limit, use HTTPS.

_Try this._ Wrap the predictor in a FastAPI endpoint, containerize it, run it on the Mac, and call it with `curl`. Then measure latency for 100 requests.

**Stanford undergrad.**

Service design: statelessness, health checks, versioned endpoints, timeouts and retries, idempotency, structured logging with request ids. Model serving patterns: model loaded once per process, warm-up, batching, shadow deployments, canary releases, rollback. Containers versus virtual machines; orchestration (Kubernetes) at a level of understanding; infrastructure as code. Cost and latency budgets as design inputs.

Data movement: object storage as the shared truth, caching, and the feature-consistency problem between batch and online paths.

_Try this._ Design the API contract (schemas, error codes, versioning) for the alert service and implement it with a golden test that replays a stored day through the endpoint.

**Stanford grad.**

Reliability engineering for ML services: service-level objectives, observability (metrics, traces, logs), load testing, graceful degradation (serve persistence when the model fails), and multi-region concerns. Edge deployment (on-device) versus cloud, and the privacy and latency trade-offs. Security threats specific to ML endpoints (model extraction, adversarial inputs).

_Try this._ Write the runbook for the alert service: what alerts fire, what the on-call does for each, and how to roll back a bad model in under five minutes.

**PhD at Apple.**

Serving at scale is a systems research area: inference optimization, hardware-aware scheduling, multi-tenant model serving, and the economics of inference. On-device serving flips the constraints to energy, memory and privacy, which the M1 sandbox mimics. The organizational lesson: the model is a small part of the service, and the service is what users experience.

_Try this._ Compare cloud batch, cloud online and on-device serving for a traffic alert product on cost, latency, privacy and failure modes, and recommend one with a migration path.

**Check yourself.**

<details>
<summary>What does a container solve?</summary>

Environment drift: the code, interpreter and libraries travel together, so the service runs the same on the Mac, a server or the cloud.

</details>

<details>
<summary>Why keep serving stateless?</summary>

Any instance can answer any request, which makes scaling, retries and rollbacks simple; state lives in storage, not in the process.

</details>

### Topic: Scaling: GPUs, distributed training and mixed precision

**Goal.** Know why GPUs are fast, how training spreads across many of them, and what breaks when models and data outgrow one machine.

**Age 5.**

A GPU is a chip with thousands of tiny calculators that all work at once, so it can do the same small sum on a huge pile of numbers very fast. Big learning machines use many GPUs at once, like many cooks in one kitchen.

_Try this._ Have four friends each add up a quarter of a long list of numbers, then add their four answers. That is parallel computing.

**Middle school.**

GPUs do many identical arithmetic steps in parallel, which is what matrix multiplication needs. Training a big model splits the work: data parallelism gives each GPU different batches and averages their gradient updates; model parallelism splits the model itself across GPUs. Mixed precision uses smaller numbers (16-bit) where it is safe, doubling speed and halving memory.

The Mac mini's GPU shares memory with the CPU; there is no copying, but there is also no way to add more.

_Try this._ Time a matrix multiplication of two 4096 × 4096 matrices on the CPU and on MPS. How many times faster is the GPU?

**High school.**

Throughput versus latency; memory bandwidth as the usual bottleneck; batch size and learning-rate scaling; data loading as the hidden bottleneck. Data parallelism with all-reduce of gradients; gradient accumulation to simulate large batches; mixed precision with loss scaling. Checkpointing and resuming. Profiling to find the actual bottleneck before optimizing.

_Try this._ Profile the spatio-temporal net's training step on the M1 with the PyTorch profiler; report the share of time in data loading, forward, backward and optimizer, and speed up the largest.

**Stanford undergrad.**

Hardware: GPU architecture (streaming multiprocessors, memory hierarchy, tensor cores), roofline models, arithmetic intensity. Distributed training: data, tensor and pipeline parallelism, ZeRO-style sharding of optimizer state, communication costs and overlap, collective operations. Numerics: bfloat16 versus float16, loss scaling, numerically stable implementations.

Scaling laws and compute-optimal training as the principled way to size a model for a budget.

_Try this._ Compute the arithmetic intensity of the net's matrix multiplications and place them on a roofline for the M1 GPU; explain why small batches are bandwidth-bound.

**Stanford grad.**

Systems for large-scale training: schedulers, fault tolerance, elastic training, and the engineering of thousand-GPU runs; compilers and kernels (fused attention, custom kernels); inference optimization (quantization, KV caching, speculative decoding). The environmental and financial cost of scale as a research constraint.

_Try this._ Estimate the compute, memory and time to train a city-scale spatio-temporal transformer on a year of statewide PeMS data on one GPU versus eight, and identify the parallelism strategy.

**PhD at Apple.**

Scale is the defining engineering fact of modern ML; the research frontier includes efficient architectures, training and inference at scale, and the science of scaling laws. The M1 sandbox teaches the discipline (profile, measure, budget) that transfers; the numbers do not.

_Try this._ Write a compute budget for a production traffic model at state scale (training, monthly retrains, inference), and argue where the cost actually lands.

**Check yourself.**

<details>
<summary>Why are GPUs fast for neural networks?</summary>

Networks are mostly matrix multiplications, which are thousands of identical, independent multiply-adds that a GPU runs in parallel.

</details>

<details>
<summary>What does mixed precision risk?</summary>

Underflow and overflow in 16-bit numbers; loss scaling and keeping master weights in 32-bit are the standard fixes.

</details>

### Read and watch, explained at every level

#### [CS 329S: Machine Learning Systems Design, lecture notes (Huyen, Stanford)](https://stanford-cs329s.github.io/)

_course · free · weeks · free; the systems view, as lecture notes_

**Age 5.** Free class notes about making a learning machine work in the real world day after day, written by someone who has done it.

**Middle school.** Chip Huyen's Stanford course, with free lecture notes and slides: what an ML system is, data engineering, training data, feature engineering, model development, deployment, monitoring, continual learning, and the human side. It is the free version of her book and covers phase 9's ideas.

**High school.** The notes on deployment (batch versus online), on monitoring and drift, and on continual learning map onto stages 12 and 13; the notes on data engineering and on training-serving skew explain the one-feature-module rule.

**Stanford undergrad.** The course notes give vocabulary and checklists (feature stores, model registries, data validation) with case studies; pair with Made With ML for implementations and with the technical-debt paper for the why.

**Stanford grad.** Its treatment of feedback loops and of testing in production is directly relevant to alerts that change traffic; the guest lectures (linked) cover industrial practice.

**PhD at Apple.** A free snapshot of the systems consensus; for a lead, the notes align a team's vocabulary before architecture discussions.

#### [MLflow documentation](https://mlflow.org/docs/latest/)

_docs · free · an afternoon · free; experiment tracking and a model registry_

**Age 5.** The manual for a tool that writes down every experiment you run: the settings, the score, and the model, so you can find them later.

**Middle school.** MLflow is a free, open-source tool for tracking experiments (parameters, metrics, artifacts), packaging models and keeping a registry of versions. Its quickstart runs locally in minutes. Use it to log every training run in phase 9 instead of a JSON file, once the JSON file gets unwieldy.

**High school.** Read the tracking quickstart and the model registry pages: log parameters and metrics from the LightGBM runs, store the model files as artifacts, and mark a version as production. Its local server shows runs in a browser.

**Stanford undergrad.** Understand what a tracking server and an artifact store are, how runs and experiments are organized, and how the registry supports staging and rollback; then decide whether the project needs it or whether a manifest and a folder suffice.

**Stanford grad.** MLflow is one of several tools for the same job; the concepts (runs, artifacts, registry, lineage) are what matter and transfer to any platform.

**PhD at Apple.** Experiment tracking is infrastructure for reproducibility; the research-relevant question is whether a result can be traced to code, data and parameters a year later, which the tool makes routine.

#### [Evidently, open-source ML monitoring documentation](https://docs.evidentlyai.com/)

_docs · free · an afternoon · free; drift and performance reports_

**Age 5.** The manual for a free tool that checks whether the data a learning machine sees today still looks like the data it learned from.

**Middle school.** Evidently is an open-source Python library for monitoring: it compares current data with reference data and reports drift per feature, data quality issues, and model performance over time, as reports or dashboards. Its tutorials run locally. Use it in stage 13 to check drift weekly.

**High school.** Read the tutorial on data drift and the one on classification performance: compute drift on the feature table against the training reference, and performance metrics on the scored predictions once labels arrive. Compare its drift tests with the PSI you implemented.

**Stanford undergrad.** The library implements several statistical tests for drift; understand what each assumes and why a single number per feature is a screen, not a verdict. Its report format is a reasonable template for the weekly monitoring output.

**Stanford grad.** Monitoring tools encode choices about what drift means; read the documentation's methodology pages critically and keep your own definition versioned.

**PhD at Apple.** Open-source monitoring made a piece of production practice inspectable; useful as a reference implementation when designing a team's own.

#### [The Turing Way](https://book.the-turing-way.org/)

_book · free · weeks · free; reproducible research as a handbook_

**Age 5.** A free handbook about doing science with computers in a way that others can check and repeat.

**Middle school.** A community-written handbook on reproducible, ethical and collaborative research with data: version control, environments, testing, documentation, data management, and project design. Free online. Read the reproducibility guide during phase 9.

**High school.** The chapters on reproducible environments, on version control, on testing and on research data management are the scientific version of the engineering rules; its templates (README, project structure) are worth adopting.

**Stanford undergrad.** Its treatment of the reproducibility spectrum (rerunnable, repeatable, reproducible, replicable) clarifies what the manifest and the golden test achieve, and its ethics chapters connect to phase 10.

**Stanford grad.** A useful reference for the norms of open research and for collaboration practices; its community process is itself a model.

**PhD at Apple.** For a lab or team, the handbook is a ready-made standard for research practice and a source of onboarding material.

#### [Site Reliability Engineering (Google), the free book](https://sre.google/sre-book/table-of-contents/)

_book · free · selected chapters, a week · free; SLOs, on-call and postmortems_

**Age 5.** A free book about keeping big computer services running all the time, and what to do when they break.

**Middle school.** Google's book on how they run services, free online: service level objectives, monitoring, being on call, handling incidents, postmortems, and avoiding overload. Read the chapters on monitoring, on service level objectives and on postmortems for the serving stage.

**High school.** The chapters on monitoring distributed systems (the four golden signals), on service level objectives, on being on call and on blameless postmortems give the operating discipline for the alert service; write the runbook and the postmortem template from them.

**Stanford undergrad.** Its treatment of error budgets, of toil, and of release engineering frames the retraining and rollback policies; the chapter on managing incidents is the procedure for a bad model in production.

**Stanford grad.** Written for large services, the principles scale down; the exercise is deciding which practices a one-machine system needs and which are ceremony.

**PhD at Apple.** A foundational operations text; ML systems inherit all of its concerns plus data and model drift.

#### [Hypothesis documentation (property-based testing)](https://hypothesis.readthedocs.io/)

_docs · free · an afternoon · free; tests that invent their own edge cases_

**Age 5.** The manual for a testing tool that makes up thousands of tricky examples for you and finds the one that breaks your code.

**Middle school.** Hypothesis is a free Python library: instead of writing one test input, you describe what inputs look like (a list of speeds between 0 and 100) and it generates many, including nasty edge cases, and shrinks any failure to the smallest example. Its quickstart is short.

**High school.** Use it on the parsers and feature code: properties such as 'cleaning never produces speeds outside 0 to 100', 'lag features of a shifted series equal shifted lag features', 'the profile of a constant series is that constant'. Read the pages on strategies and on shrinking.

**Stanford undergrad.** Property-based testing finds bugs example-based tests miss (empty inputs, boundaries, unusual timestamps); its stateful testing extends to pipelines. Understand how to write good properties: invariants, round trips, comparisons with a reference implementation.

**Stanford grad.** For data code, properties are often the only tractable specification; the technique is also a way to document assumptions executably, which the data-contract exercise needs.

**PhD at Apple.** Property-based testing is underused in ML engineering; adopting it for feature libraries is a cheap, high-yield practice.

#### [Made With ML (Mohandas)](https://madewithml.com/)

_course · free · weeks · free; testing, monitoring and pipelines with code_

**Age 5.** A free website that shows, with real code, how to build a learning machine the careful way, with tests and checks so it keeps working.

**Middle school.** Goku Mohandas's course walks through an ML project end to end: data, modelling, and then the operational parts, testing, monitoring, pipelines and serving, all with code you can read. It is free. Follow the operational sections in phase 7.

**High school.** Its testing lesson (data tests, model tests, behavioral tests) and its monitoring lesson (drift detection with statistical tests) are directly reusable in stage 13. Its pipeline lesson shows orchestration that you can scale down to launchd jobs on one Mac mini without losing the ideas.

**Stanford undergrad.** Compare its tooling with a minimal setup: one Mac mini does not need Airflow or a feature store, but it needs the ideas (idempotent tasks, lineage, validation, versioning). Extract the principles and implement them in a few hundred lines rather than adopting the stack.

**Stanford grad.** Its coverage of experiment tracking and model versioning gives the vocabulary to design a JSON-based registry for this project, and its lessons on scaling with Ray are for later, when the data outgrows one machine. The course's opinionated structure is itself a useful design for an internal curriculum.

**PhD at Apple.** A free, opinionated, code-first MLOps curriculum is rare. Use it to align a team on practices before choosing a platform, and note which of its practices survive the translation to a single-machine deployment: most of them, which says something about what MLOps actually is.

#### [Machine Learning Engineering (Burkov)](https://www.mlebook.com/)

_book · free · weeks · free to read online_

**Age 5.** A short, plain book about the whole job of building learning machines, from getting the data to keeping them running.

**Middle school.** Andriy Burkov's book, free to read online under a 'read first, buy later' model. It covers the project lifecycle: collecting data, building features, training, evaluating, deploying, and monitoring. It is concise, and each chapter reads like a checklist you can apply to a stage of this project.

**High school.** Its chapters on data leakage, on evaluation and on deployment are the fastest reads on those topics. The leakage chapter lists exactly the traps lessons 4 to 6 cover: target leakage, temporal leakage, and contamination between train and test. Read it before building the feature table.

**Stanford undergrad.** Burkov's earlier Hundred-Page Machine Learning Book is the companion for the math; together they are a compact reference set. Use the engineering book's checklists as review criteria for each stage: does stage 8 pass the leakage checklist, does stage 12 pass the deployment one.

**Stanford grad.** It is opinionated and short. Where it disagrees with Huyen (for instance, on how much automation a small team needs), both positions are worth thinking through, and the disagreement itself is a useful prompt for deciding what this project's phase 7 should actually build.

**PhD at Apple.** A concise practitioner reference is useful for writing internal guidelines, and its brevity forces prioritization, which senior engineers need more than additional content. Its structure is a reasonable skeleton for an engineering-standards document for an ML team.

#### [Full Stack Deep Learning](https://fullstackdeeplearning.com/)

_course · free · weeks · free; the production lifecycle with labs_

**Age 5.** A free class about everything around a learning machine: getting data, keeping it working, and putting it inside a real app.

**Middle school.** A free course (lectures and labs online) that covers the parts of a deep learning project that courses usually skip: data management, training infrastructure, testing, deployment, monitoring, and building a working product. Watch the deployment and monitoring lectures in phase 7.

**High school.** Its labs build an end-to-end application (a text recognizer) with experiment tracking, tests, a web deployment and monitoring; the lectures on troubleshooting training and on data management are directly useful even for a tabular model. The lecture on deployment patterns frames the replay-mode choice.

**Stanford undergrad.** The course's structure is the production ML lifecycle; its lectures on ethics, on project management and on team organization cover the non-technical failure modes. Compare its tooling choices with the minimal single-machine setup and extract principles.

**Stanford grad.** Together with Huyen's book and Made With ML it forms the practical MLOps curriculum; its treatment of testing and troubleshooting neural network training is the most concrete of the three.

**PhD at Apple.** Its evolution across years is a snapshot of how industry practice changed; for a lead, it is the course to point engineers to when they move from research code to a product.

#### [Rules of Machine Learning (Zinkevich, Google)](https://developers.google.com/machine-learning/guides/rules-of-ml)

_docs · free · an hour · free; 43 rules from hard experience_

**Age 5.** A list of rules from an engineer who has built many learning machines, about what to do first and what mistakes to avoid.

**Middle school.** A free document of 43 short rules from Google, such as 'do not be afraid to launch a product without machine learning', 'design and implement metrics first', and 'keep the first model simple'. Read it in an hour during phase 7; you will recognize this project in most of the rules.

**High school.** Its phases (before ML, the first pipeline, feature engineering, slowed growth) match the course's phases. Rules about training-serving skew, about logging features at serving time, and about launching a simple model first are exactly the decisions in stages 12 and 13. Note which rules you have already broken.

**Stanford undergrad.** The document encodes hard-won operational judgment: measure before modelling, prefer simple and debuggable, watch for silent failures, and treat feature logging as the foundation of everything later. Use it as a review checklist for the serving design.

**Stanford grad.** Some rules are specific to large consumer products (ranking, human-in-the-loop labels) and do not transfer to a one-machine forecaster; identifying which is a useful exercise in generalizing engineering advice.

**PhD at Apple.** A primary source on how a large organization thinks about ML systems, written before MLOps was a word; read with Sculley's technical-debt paper as a pair.

#### [The ML Test Score (Breck, Cai, Nielsen, Salib, Sculley, 2017)](https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/)

_paper · free · an hour · free; the production-readiness rubric_

**Age 5.** A checklist from engineers who ask: did you test your data, did you test your model, did you test the machine that runs it? You get points for each thing you checked.

**Middle school.** A short free paper that gives a scoring rubric for how production-ready an ML system is: tests for data (is the schema right, are features useful), tests for model development (is the baseline beaten, are hyperparameters tuned), tests for infrastructure (can you roll back), and monitoring (is the model stale). Score this project against it in phase 7.

**High school.** Each of the 28 items is a concrete test you could write for this pipeline: feature schema tests, a golden-day pipeline test, a check that serving features equal training features, a canary before model swap, alerts on data freshness. Turn the rubric into a `tests/` folder plan.

**Stanford undergrad.** The rubric operationalizes the technical-debt paper; its scoring (points for manual checks, more for automated ones) encodes the principle that unautomated checks decay. Use it to prioritize what stage 13 automates first.

**Stanford grad.** It predates data-validation tooling and drift libraries but defines what they should do; reading it explains why those tools exist and what they still miss (feedback loops, label delay).

**PhD at Apple.** A rare attempt to quantify production readiness; its limits (a checklist cannot weight items by risk) point toward risk-based test design, which is an open engineering problem in ML systems.

#### [Hidden Technical Debt in Machine Learning Systems (Sculley et al., 2015)](https://papers.nips.cc/paper_files/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html)

_paper · free · an hour, twice · free; ten pages, read twice_

**Age 5.** A short paper by people who build big systems, saying: the learning part is small, and the messy parts around it are big and bite you later if you are not careful.

**Middle school.** Ten pages, free. It lists the ways ML systems rot over time: tangled dependencies on data, glue code nobody owns, pipelines nobody understands, feedback loops where the model changes the data it learns from, and configuration mess. Read it once now and once after building phase 7, when every paragraph will sound familiar.

**High school.** Its famous figure, a tiny box labelled ML code surrounded by large boxes of infrastructure, is a picture of this pipeline. Its warnings about correction cascades, undeclared consumers and data dependencies map to concrete rules here: one feature module, a manifest with hashes, versioned labels, and no script that silently reads another's output.

**Stanford undergrad.** Treat each named debt as a checklist for the project: entanglement (changing one feature changes everything downstream), hidden feedback loops (alerts change traffic), pipeline jungles (stages 5 to 8 as separate scripts), dead experimental code paths, and configuration debt. Write one sentence per item about where it could appear here and what prevents it.

**Stanford grad.** Later work, on ML test scores and on data-validation systems for production pipelines, operationalizes the paper's warnings; read it together with those follow-ups and with Huyen's chapter 8. The paper's concept of the CACE principle, changing anything changes everything, is the reason retraining needs a canary.

**PhD at Apple.** A foundational engineering paper whose legacy is the field of ML systems and MLOps. At your level the exercise is to separate which debts are inherent to ML (data dependencies, feedback) from which are ordinary software engineering, and to identify where research, rather than discipline, could reduce the inherent ones.

#### [launchd.info and the pytest documentation](https://www.launchd.info/)

_docs · free · an afternoon · free_

**Age 5.** Instructions for two tools: one that makes the Mac run a program on a schedule by itself, and one that runs all your checks with a single command.

**Middle school.** launchd.info explains macOS's scheduler: you write a small file called a plist saying what to run and how often, and macOS runs it even after a restart. The pytest documentation explains how to write and run tests. Read the getting-started page of each and try the smallest example.

**High school.** launchd: `StartInterval` and `StartCalendarInterval`, `WorkingDirectory`, log paths, `launchctl load` and `unload`, and why it beats cron on a Mac (it handles sleep and wake, runs per user, and restarts jobs). pytest: fixtures, `parametrize`, `-k` to select tests, `-x` to stop early, `tmp_path` and `monkeypatch`, and `conftest.py` for shared fixtures.

**Stanford undergrad.** Understand launchd agents versus daemons and the permission model (Full Disk Access for reading files, the keychain for secrets), because a job that works in the terminal and fails under launchd is almost always a permissions or environment problem. In pytest, fixture scoping and the plugin ecosystem (`pytest-cov`, `hypothesis`) are the next layer.

**Stanford grad.** Scheduling is orchestration at small scale; know its failure modes (overlapping runs, missed intervals after sleep, environment differences) and their mitigations (lock files, idempotent jobs, explicit environment in the plist). For testing, property-based tests with hypothesis are the right tool for parsers and feature code, where the input space is large.

**PhD at Apple.** Tooling choices are context-dependent. The durable skill is specifying a job's contract, idempotence, timeout, alerting, and retry policy, independently of the scheduler, so that migrating from launchd to a cluster scheduler later is mechanical rather than a redesign.

#### [FastAPI documentation](https://fastapi.tiangolo.com/)

_docs · free · an afternoon · free; a door on the predictor in an afternoon_

**Age 5.** The manual for a tool that gives your program a door on the internet, so other programs can knock and ask questions.

**Middle school.** FastAPI is a free Python framework for building web APIs. Its tutorial shows how to turn a function into an endpoint, define what requests and responses look like, and run a server, with automatically generated interactive documentation. The first ten tutorial pages are enough to wrap the predictor.

**High school.** Read the tutorial through request bodies, path and query parameters, response models and error handling. Wrap the predictor: a request with station and horizon, a response with probability and alert. Try the automatic docs page and call the endpoint from the command line.

**Stanford undergrad.** FastAPI is built on type hints and a validation library; understand how validation, serialization and documentation come from the same models, and how dependency injection structures shared resources such as a loaded model. Read the pages on background tasks and on testing with the test client.

**Stanford grad.** Its async model and deployment guidance (workers, reverse proxies) are the practical side of serving; for batch-predict pipelines a web API is optional, for a dashboard it is the interface.

**PhD at Apple.** Framework choice matters less than the service contract; FastAPI's docs are also a good example of documentation that teaches the underlying concepts rather than only the library.

#### [Docker, Get started guide](https://docs.docker.com/get-started/)

_docs · free · an afternoon · free; the service in a box_

**Age 5.** Instructions for packing your program and everything it needs into one box that runs the same on any computer.

**Middle school.** Docker packages code, its interpreter and libraries into an image that runs anywhere Docker runs. The Get Started guide walks through building an image from a Dockerfile, running it, keeping data outside the container, and sharing images. An afternoon is enough for this project's service.

**High school.** Do the first parts of the guide (build, run, volumes), then write a Dockerfile for the prediction service: a slim Python base image, requirements installed, the code copied, the model files mounted from outside. Keep data and secrets out of the image.

**Stanford undergrad.** Understand layers and caching (order your Dockerfile so dependencies are cached), image size, multi-stage builds, and the difference between an image and a container. Know what a container does not isolate (the kernel) and what that means for security.

**Stanford grad.** Containers are the unit of deployment on orchestration platforms; the guide's later sections on Compose and on multi-container apps are the step toward a small stack (service, scheduler, monitoring). For Apple silicon, note multi-architecture images.

**PhD at Apple.** Containerization is the boundary that makes reproducibility and deployment mechanical; the research-relevant question is whether your experiment can be rerun from an image and a data hash a year later.

#### [PyTorch distributed overview](https://pytorch.org/tutorials/beginner/dist_overview.html)

_docs · free · an afternoon · free; what you would need past one machine_

**Age 5.** The manual page about using many computers or chips at once to train one learning machine faster.

**Middle school.** PyTorch's overview of distributed training explains the main options: data parallelism (every device gets different batches and they average their updates), and ways to split a model across devices when it is too big for one. It points to tutorials for each. Read it to understand what you would need if the model outgrew the Mac mini.

**High school.** Read the overview and the DistributedDataParallel tutorial to see the pattern: one process per device, a shared training script, gradients averaged with an all-reduce. You cannot run it meaningfully on one Mac, but reading it prepares you for any cloud GPU.

**Stanford undergrad.** Understand process groups, collective operations, the cost model of gradient synchronization, and the fully sharded variant that partitions optimizer state and parameters. The tutorials on checkpointing and on elastic training cover the operational side.

**Stanford grad.** Distributed training is where systems knowledge (communication, memory hierarchy) meets ML; the overview links the pieces you will meet in the ultra-scale playbook and in any large-model paper's training section.

**PhD at Apple.** The distributed stack is a research area in itself (compilers, sharding strategies, fault tolerance); for a lead, the practical question is which parallelism a model needs and what it costs in engineering time.

#### [The Ultra-Scale Playbook: Training LLMs on GPU Clusters (Hugging Face)](https://huggingface.co/spaces/nanotron/ultrascale-playbook)

_book · free · weeks · free; how the largest models are actually trained_

**Age 5.** A free online book about training gigantic language models on hundreds of chips at once, with pictures of how the work is split up.

**Middle school.** Hugging Face's playbook explains how large language models are trained across many GPUs: splitting data, splitting the model, splitting the layers, and keeping all the chips busy. It reports thousands of experiments. It is advanced; read the introduction to see how much engineering scale involves.

**High school.** Read the sections on data parallelism and on memory (activations, optimizer state) to understand why a model's memory is several times its parameter count, and how mixed precision and gradient accumulation help. The interactive figures make the trade-offs concrete.

**Stanford undergrad.** The playbook is the current practical reference on parallelism strategies (data, tensor, pipeline, context), on overlapping communication with computation, and on choosing a configuration for a given cluster. Its benchmarks are reproducible and its code is open.

**Stanford grad.** Compare with the academic papers on sharding and pipeline parallelism; the playbook's contribution is measurement across configurations. For anyone planning a large training run, it is the checklist.

**PhD at Apple.** Open, measured engineering knowledge of this kind is rare; the playbook is both a reference and a model of how to publish systems results.

#### [Scaling Laws for Neural Language Models (Kaplan et al., 2020)](https://arxiv.org/abs/2001.08361)

_paper · free · an evening or two · free; measure before assuming bigger helps_

**Age 5.** A paper that found that bigger models trained on more text get better in a very predictable way, like a line on a graph.

**Middle school.** Researchers trained language models of many sizes and found that the loss falls smoothly and predictably as model size, data and compute grow, following power laws. That predictability is why companies can plan enormous training runs. Read the introduction and the main figures.

**High school.** Look at the figures relating loss to parameters, tokens and compute; note that architecture details mattered less than scale. Later work (compute-optimal training) revised how to split a budget between model size and data. The lesson for any project: measure how your metric changes with data size before assuming a bigger model will help.

**Stanford undergrad.** The paper's methodology (fit power laws across orders of magnitude, hold other factors fixed) is a template for empirical scaling studies; its revision by compute-optimal analyses shows the importance of tuning learning rates per scale. Apply the method to the traffic model: skill against months of data.

**Stanford grad.** Scaling laws organize much of current research on large models; their limits (task-specific behavior, data quality, emergence claims) are active debates. For forecasting, the analogous question is whether more stations and years keep improving skill.

**PhD at Apple.** Understanding scaling is now part of research strategy; the open questions include what scales with what, and where data quality dominates quantity.

#### [Training Compute-Optimal Large Language Models (Hoffmann et al., 2022)](https://arxiv.org/abs/2203.15556)

_paper · free · an evening or two · free; the correction to the scaling laws_

**Age 5.** A paper that found that many big models had been trained on too little text, and that a smaller model trained on more text can be better.

**Middle school.** Researchers trained many models with different sizes and amounts of data and found the best balance for a fixed compute budget: model size and data should grow together. Their model, Chinchilla, beat bigger models trained on less data. Free; read it after the scaling-laws paper.

**High school.** Read the method (three approaches to fitting the trade-off) and the main result (tokens should scale roughly with parameters); compare with Kaplan's earlier conclusions and note how a methodological change (tuning learning-rate schedules per run) changed the answer.

**Stanford undergrad.** The paper is a lesson in experimental methodology: the earlier scaling result depended on a confound, and fixing it moved the optimum substantially. Apply the same skepticism to any scaling claim, including your own data-size curves.

**Stanford grad.** Later work refines the trade-off for inference cost and data quality; the paper's fitted laws remain the reference point for planning training runs.

**PhD at Apple.** Together with its predecessor it defines how the field reasons about compute allocation; for research strategy, it is the argument for data over parameters.

### Build

- `19_predict_live.py` in replay mode; `20_monitor.py`; four launchd plists; `make setup` and `make all`.
- A golden-file test that serving features equal training features for one stored day; a memory and time line in every stage's log.
- Wrap the predictor in FastAPI, containerize it, run the golden test through the endpoint, and measure latency.
- Profile the spatio-temporal net's training step on the M1 and speed up its largest cost; write a compute budget for a statewide model.
- Score the pipeline against the ML Test Score rubric and turn the gaps into a `tests/` plan.

### Done when

- From a fresh clone, `make setup && make all` produces the evaluation report, and the containerized service answers the golden test.
- The replay loop has run unattended for a week and the monitor has written four weekly files.

### Pitfalls

- A second copy of the feature code.
- cron instead of launchd on a Mac that sleeps.
- No way to roll back a bad model.
- Optimizing before profiling.

## Phase 10: Beyond supervised learning

_weeks 60–70 · ≈70 h · builds an alert-policy analysis, a fairness audit, a dashboard ranker_

**Goal.** What changes when a model's output is a decision: reinforcement learning, causal inference, fairness, privacy and ranking.

### What you need to learn

- Markov decision processes, value functions, Q-learning and policy gradients; bandits; off-policy evaluation from logged data; where RL fits in traffic control.
- Potential outcomes, confounding, causal diagrams, randomization; adjustment, instruments, difference-in-differences; interference; performative prediction.
- Fairness definitions and their conflicts; measuring disparity by slice; differential privacy; documentation (model cards, datasheets); governance.
- Recommenders and ranking: collaborative filtering, matrix factorization, learning to rank, ranking metrics, exposure bias and feedback loops.

### Topic: Reinforcement learning: agents, rewards, bandits and policies

**Goal.** Understand learning by acting: the setup, the core algorithms, and where it fits (and does not fit) in a traffic system.

**Age 5.**

Some learning happens by trying things and seeing what happens: touch the hot stove once and you learn. A computer can learn a game the same way, getting points for good moves and losing points for bad ones.

_Try this._ Teach a friend a simple game only by saying 'good' or 'bad' after each move, never explaining the rules. Do they figure it out?

**Middle school.**

In reinforcement learning an agent sees a state, picks an action, gets a reward, and the world moves to a new state. It learns a policy, a rule for choosing actions, that collects the most reward over time. The hard part is that rewards may come late, and that trying new actions (exploration) costs something.

A bandit is the simplest case: one decision, repeated, like which of three alert messages drivers respond to most. Traffic lights and ramp meters can be controlled by reinforcement learning.

_Try this._ Play with a bandit on paper: three slot machines with hidden payout rates; keep choosing, track averages, and decide when to stop exploring.

**High school.**

Markov decision processes: states, actions, transitions, rewards, discounting. Value functions and the Bellman equation; Q-learning and policy gradients as the two families; exploration versus exploitation; deep RL (a network for the value or policy). Bandits: epsilon-greedy, upper confidence bound, Thompson sampling. Simulators are usually required because trying actions in the real world is expensive or unsafe.

For traffic: ramp metering and signal timing are natural RL problems; alerting is closer to a bandit with delayed feedback.

_Try this._ Implement Q-learning on a small grid world in numpy, then Thompson sampling for a three-arm bandit; plot regret over time.

**Stanford undergrad.**

Dynamic programming (value and policy iteration), temporal-difference learning, function approximation and its instability (the deadly triad), policy gradient theorem and actor-critic methods, off-policy evaluation from logged data (importance sampling), and the sample-inefficiency problem. Contextual bandits as the practical middle ground for decisions with immediate feedback.

Offline RL and off-policy evaluation are what a system with logged alerts and outcomes could actually use; online RL on live traffic is rarely acceptable.

_Try this._ Formulate the alerting policy as a contextual bandit with delayed reward, propose an off-policy estimator for a new policy from logged data, and state its assumptions.

**Stanford grad.**

Model-based RL with learned or physical simulators (traffic microsimulation as an environment), multi-agent RL for networks of intersections, safe RL with constraints, and the evaluation problem (simulator-to-real gap). Connections to control theory and to causal inference (policies as interventions).

_Try this._ Design a ramp-metering controller for one Fremont on-ramp trained in a traffic simulator, and the validation that would justify a field trial.

**PhD at Apple.**

RL in products is mostly bandits and offline evaluation; full RL succeeds where a faithful simulator exists (games, some control). The research frontier includes sample efficiency, offline RL with guarantees, RL from human feedback for language models, and safe deployment. For transport, learned control of signals and meters is an active applied area with real deployments.

_Try this._ Write the evaluation protocol for an RL-controlled corridor: metrics, baselines (fixed-time, actuated), safety constraints, and how you would attribute changes to the controller rather than to demand shifts.

**Check yourself.**

<details>
<summary>What distinguishes RL from supervised learning?</summary>

There are no correct labels, only rewards, and the agent's actions change the data it sees.

</details>

<details>
<summary>Why are bandits the practical case?</summary>

Immediate feedback and no state transitions make evaluation and safety tractable; most product decisions look like that.

</details>

### Topic: Causal inference: from correlation to intervention

**Goal.** Ask what would happen if you acted, and know what data can and cannot tell you about it.

**Age 5.**

Ice cream sales and sunburns go up together, but ice cream does not cause sunburn; the sun causes both. Finding the real cause is harder than noticing things go together.

_Try this._ Think of two things that happen together in your life but where one does not cause the other. What causes both?

**Middle school.**

Correlation is when two things move together; causation is when changing one changes the other. Confounders are hidden causes of both. The cleanest way to find a cause is an experiment: randomly do the thing to some and not others, then compare. When you cannot experiment, you need assumptions and careful methods.

For traffic: incidents and slowdowns go together, but reporting an incident also depends on congestion; and an alert that changes driver behavior changes the outcome it predicted.

_Try this._ Design a fair test of whether an alert makes drivers reroute: who gets the alert, who does not, and how you decide.

**High school.**

Potential outcomes: each unit has an outcome with and without treatment; the effect is the difference; you only ever see one. Randomization makes the groups comparable. Without it: adjust for confounders (regression, matching, weighting), use natural experiments (instruments, discontinuities, differences-in-differences), and draw the causal diagram (DAG) to decide what to adjust for and what not to (colliders).

A/B tests are randomized experiments; interference (one driver's reroute affects others) breaks the usual analysis.

_Try this._ Draw the DAG for incident, congestion, reporting, rain and alert. Which variables should you adjust for to estimate the effect of an incident on speed, and which must you not?

**Stanford undergrad.**

Identification conditions (exchangeability, positivity, consistency), the back-door and front-door criteria, do-calculus at a reading level, estimators (outcome regression, inverse probability weighting, doubly robust), instrumental variables, regression discontinuity, difference-in-differences and synthetic controls. Causal machine learning: double machine learning and causal forests for heterogeneous effects.

Interference and spillovers, which are the rule in traffic; cluster-randomized and switchback designs.

_Try this._ Estimate the effect of rain on the probability of a slow episode with inverse probability weighting on hour and weekday, and compare with a naive difference; explain the gap.

**Stanford grad.**

Time-varying treatments and g-methods for sequential policies (alerting is one); mediation; sensitivity analysis to unmeasured confounding; causal discovery and its limits; causal representation learning. Performative prediction as a bridge between causal inference and ML deployment.

_Try this._ Design a switchback experiment for alerts across days and routes that respects interference, and the analysis that estimates the effect on travel time.

**PhD at Apple.**

Products that act on predictions are causal systems, and evaluating them requires experimental design or credible quasi-experiments; the research frontier includes experimentation under interference, off-policy evaluation, and causal effects of algorithmic systems on populations. Traffic is an unusually good testbed because interventions are visible and physical.

_Try this._ Write the causal question, identification strategy and analysis plan for 'do alerts reduce total delay on I-880 in the evening peak?'

**Check yourself.**

<details>
<summary>Why does adjusting for a collider cause bias?</summary>

Conditioning on a common effect of two variables induces an association between them that does not exist causally.

</details>

<details>
<summary>What does randomization buy?</summary>

Treated and untreated groups are comparable on everything, measured or not, so the difference in outcomes is the causal effect on average.

</details>

### Topic: Fairness, privacy and ethics: what a model owes people

**Goal.** See where a traffic model can treat people unequally or expose them, and know the tools and the limits of technical fixes.

**Age 5.**

A helper should be fair: it should not warn some neighborhoods and forget others. And it should not tell anyone where a particular person drove.

_Try this._ If a helper only worked for one street, who would be left out? How would they feel?

**Middle school.**

Fairness: a model's mistakes can fall more on some groups, for example if detectors are worse in some neighborhoods, alerts there are worse. Privacy: data about where people go is sensitive; even aggregated speeds can reveal a single car at 3 am. Ethics also covers being honest about what the model can do and who is accountable when it is wrong.

_Try this._ Look at the station map. Which parts of Fremont have few detectors? What does that mean for whom the alerts help?

**High school.**

Fairness definitions and why they conflict: equal error rates across groups, calibration within groups, demographic parity; you cannot have all at once. Measuring: slice every metric by area and time. Privacy: anonymization fails against linkage; differential privacy adds calibrated noise with a provable guarantee; aggregation thresholds and k-anonymity are weaker but common. Ethics in practice: documentation (datasheets, model cards), stating limits, and a person accountable for decisions.

For this project: detector coverage is uneven; probe data would raise privacy questions the loop data do not.

_Try this._ Compute recall and false-alarm rate per station and plot them on the map. Which areas get worse service, and why?

**Stanford undergrad.**

Formal fairness criteria (independence, separation, sufficiency) and impossibility results; sources of disparity (measurement, representation, label, deployment); mitigation methods (reweighting, constraints, post-processing) and their costs. Differential privacy: definition, composition, the privacy budget, DP-SGD; federated learning and its limits. Governance: impact assessments, audits, and regulation as constraints on design.

Prediction versus decision: fairness lives in the policy (who gets alerts) as much as in the model.

_Try this._ Implement a DP release of hourly average speeds per station with Laplace noise; plot utility against epsilon and decide the smallest epsilon the product could tolerate.

**Stanford grad.**

The limits of technical fairness (measurement of groups, contested definitions, distributional versus procedural justice); long-term dynamics of fair decisions; privacy attacks on models (membership inference, extraction) and defenses; the tension between transparency and gaming. Participatory design and the role of affected communities in defining harms.

_Try this._ Audit the alert policy for disparate impact across neighborhoods under two fairness definitions, show they conflict on this data, and write the memo that explains the choice to a city council.

**PhD at Apple.**

Responsible ML at scale is organizational: review processes, documentation, red-teaming, monitoring for disparate impact after deployment, and legal compliance. The research frontier includes fairness under distribution shift and feedback, privacy-preserving learning with useful guarantees, and evaluation of societal impact. Transport systems allocate a public good, which raises the stakes.

_Try this._ Design the governance for a public traffic-alert system: what is measured, who reviews it, what triggers a pause, and how the public can contest a decision.

**Check yourself.**

<details>
<summary>Why can a model be calibrated for every group and still have unequal error rates?</summary>

Calibration and equal error rates are different criteria, and when base rates differ across groups they cannot all hold at once.

</details>

<details>
<summary>What does differential privacy guarantee?</summary>

That any single person's data changes the released output's distribution by at most a bounded factor, whatever else an attacker knows.

</details>

### Topic: Recommender systems and ranking

**Goal.** Rank things for people from their behavior, evaluate a ranking, and see the feedback loops that ranking creates.

**Age 5.**

When a video app shows you what to watch next, it guessed from what you and people like you watched before. That guessing is a recommender.

Our helper ranks roads too: the ones most likely to jam go to the top of the list.

_Try this._ Make a list of your friend's favorite games ranked from most to least. What did you use to guess?

**Middle school.**

Recommenders learn from clicks, plays and ratings. Collaborative filtering says: people who liked what you liked will like this; matrix factorization learns a few hidden traits per person and per item. Content-based methods use item features. Ranking systems score candidates and order them; evaluation asks whether the good items are near the top.

Ranking the top-risk segments per route for a dashboard is a small ranking problem with the same metrics.

_Try this._ Rank ten stations by predicted probability of a slow episode. If only three actually became slow, how many were in your top three? That is precision at 3.

**High school.**

Matrix factorization with implicit feedback, nearest-neighbor methods, and modern two-tower models that embed users and items; candidate generation then ranking; ranking losses (pointwise, pairwise, listwise). Metrics: precision at k, recall at k, mean average precision, NDCG. Offline evaluation is biased by what was shown (exposure), which is the central difficulty.

Feedback loops: what is recommended gets clicked, which trains the next model; popularity bias and filter bubbles follow.

_Try this._ Implement matrix factorization by alternating least squares on a small public ratings dataset and report NDCG at 10 against a popularity baseline.

**Stanford undergrad.**

Learning to rank (LambdaMART, which is gradient boosting for ranking), embedding retrieval with approximate nearest neighbors, position bias and unbiased learning to rank, off-policy evaluation of rankers from logged data (inverse propensity scoring), and exploration in recommendation. Cold start and side information. Multi-objective ranking (relevance, diversity, freshness).

For the dashboard: rank segments by calibrated risk times exposure (traffic volume), and evaluate with NDCG against realized episodes.

_Try this._ Train a LightGBM ranker (lambdarank) to order stations by risk each 5 minutes using the classifier's features, and compare NDCG at 5 with sorting by calibrated probability.

**Stanford grad.**

Recommendation as a causal and sequential problem: exposure bias, interference between users, long-term effects, and the alignment of optimization targets with user welfare. Large-scale systems (retrieval, ranking, re-ranking stages) and their latency budgets. Evaluation through online experiments and interleaving.

_Try this._ Propose an off-policy evaluation of a new segment-ranking policy for the dashboard from logged views and outcomes, and the propensity model it needs.

**PhD at Apple.**

Recommenders are the most deployed ML systems and the most studied for societal effects; research spans sequential recommendation, causal recommendation, fairness of exposure, and generative retrieval. The methodological core, ranking under partial feedback, transfers to any system that shows people a prioritized list, including a traffic dashboard.

_Try this._ Design the ranking objective for a public traffic dashboard that balances predicted delay, number of affected drivers and fairness of coverage, and say how you would evaluate it.

**Check yourself.**

<details>
<summary>Why is offline evaluation of a recommender biased?</summary>

Logged feedback exists only for items that were shown; items never shown have no data, and the log reflects the old policy's choices.

</details>

<details>
<summary>What does NDCG reward?</summary>

Putting relevant items near the top, with a discount that makes the first positions count most.

</details>

### Read and watch, explained at every level

#### [Reinforcement Learning: An Introduction (Sutton, Barto)](http://incompleteideas.net/book/the-book-2nd.html)

_book · free · a semester · free; the standard first book_

**Age 5.** The classic free book about learning by trial and reward, by two of the people who invented the field.

**Middle school.** Sutton and Barto's textbook, free online, explains reinforcement learning from bandits through Markov decision processes, dynamic programming, Monte Carlo and temporal-difference methods, to function approximation and policy gradients. Part 1 is readable with basic probability. It is the standard first book.

**High school.** Read chapters 1 to 6 (bandits, MDPs, dynamic programming, Monte Carlo, temporal-difference learning) and implement the small examples (grid worlds, the cliff walk). Chapter 13 on policy gradients is the bridge to deep RL. The exercises are excellent and the answers are checkable by simulation.

**Stanford undergrad.** The book's careful treatment of the Bellman equations, of on-policy versus off-policy learning, and of the deadly triad (function approximation, bootstrapping, off-policy) is what you need to read deep RL papers critically. Chapter 11 on off-policy methods with approximation is the relevant one for learning from logged alert data.

**Stanford grad.** Its coverage of eligibility traces, of planning with learned models, and of the psychology and neuroscience connections gives the field's intellectual context; the second edition's case studies show what worked and why.

**PhD at Apple.** The canonical reference; its authors' later essays on the field's direction are worth reading alongside. For a lead, it is the book to require before anyone proposes RL for a control problem.

#### [David Silver's Reinforcement Learning course (UCL, DeepMind)](https://www.davidsilver.uk/teaching/)

_video · free · ten lectures, two weeks · free lectures alongside the book_

**Age 5.** Video lectures by a scientist who helped build a computer that beat the best human at the board game Go, explaining how machines learn from rewards.

**Middle school.** Ten free lectures that follow Sutton and Barto: Markov decision processes, planning, learning without a model, function approximation, policy gradients, and games. Watch them alongside the book; the lectures make the equations feel inevitable.

**High school.** Lectures 1 to 5 cover the foundations used in the RL topic; lecture 6 on value function approximation and lecture 7 on policy gradients are the bridge to deep RL. Do the book's exercises after each lecture.

**Stanford undergrad.** Silver's derivations (the Bellman equations, the policy gradient theorem) are clear and complete; his lecture on exploration and on integrating learning and planning is a compact treatment of the ideas behind model-based RL.

**Stanford grad.** The course predates much of deep RL; pair it with Spinning Up for algorithms and with recent surveys for offline RL. Its value is conceptual clarity.

**PhD at Apple.** A widely used lecture series that defines the shared vocabulary of the field; useful as a common reference when a team discusses control problems.

#### [Spinning Up in Deep RL (OpenAI)](https://spinningup.openai.com/)

_docs · free · a week · free; clean algorithms and a reading list_

**Age 5.** A free website that teaches modern trial-and-reward learning with clear notes and code you can run.

**Middle school.** Spinning Up is an educational site with an introduction to deep reinforcement learning, a list of key papers, and clean implementations of the main algorithms. It was written for people who know some deep learning and want to do RL properly. Read the introduction and run one algorithm on a simple simulated task.

**High school.** Read the three-part introduction (key concepts, kinds of algorithms, the policy-gradient derivation) and study the code for one algorithm (PPO or DQN). The exercises and the 'spinning up as a researcher' essay are practical advice.

**Stanford undergrad.** Its implementations are readable references for the standard algorithms and their tricks; the key-papers list is a curated syllabus for the field. Use it to reproduce a result on a simulated control task before attempting a traffic-control simulator.

**Stanford grad.** It reflects the state of deep RL around its writing; later methods (offline RL, RL from human feedback) are covered elsewhere. Its clarity about what makes RL hard (sample efficiency, instability, reward design) remains accurate.

**PhD at Apple.** A useful teaching resource and a reminder that reproducible baselines were a problem the field had to solve deliberately.

#### [Causal Inference: The Mixtape (Cunningham)](https://mixtape.scunning.com/)

_book · free · selected chapters, two weeks · free; applied causal methods with code_

**Age 5.** A free book about telling causes from coincidences, written in a friendly voice with lots of examples from real life.

**Middle school.** Scott Cunningham's book, free online, teaches the methods economists use to find causes in data without experiments: comparing groups over time, using cut-offs, using instruments, and matching. It explains directed graphs (pictures of cause and effect) early and uses them throughout. Code is included.

**High school.** Read the chapters on directed acyclic graphs, on potential outcomes, on matching, and on difference-in-differences. Then draw the DAG for incidents, congestion and reporting and decide what you can estimate from the archive.

**Stanford undergrad.** The Mixtape is the applied-economics complement to Hernán and Robins: more on quasi-experimental designs (instruments, regression discontinuity, synthetic control), less on time-varying treatments. Its worked examples with code are the fastest way to see each method fail and succeed.

**Stanford grad.** Its treatment of difference-in-differences includes the recent literature on staggered adoption, which is relevant to evaluating roadwork or policy changes across corridors. Read it alongside the causal machine-learning literature.

**PhD at Apple.** A free, current, applied causal-inference text; for a product team asking what an alert causes, it is the accessible primer before the formal one.

#### [Causal Inference: What If (Hernán, Robins)](https://miguelhernan.org/whatifbook)

_book · free · part 1, three weeks · free; the rigorous base, part 1 now_

**Age 5.** A free book about telling the difference between 'these two things happen together' and 'this one causes that one', which is harder than it sounds.

**Middle school.** A textbook by two epidemiologists, free as a PDF, about answering causal questions from data: what would have happened if we had done something else. Part 1 is readable with basic statistics and uses simple examples. Read it in the final phase when you ask what an alert actually causes.

**High school.** Part 1 introduces counterfactuals, randomized experiments, confounding, and the diagrams (DAGs) that make assumptions explicit. Apply it to the project: does an incident cause the slowdown, or does congestion cause the incident to be reported? The book gives the tools to state that question precisely.

**Stanford undergrad.** Its treatment of exchangeability, positivity and consistency as the conditions for causal identification, and of methods (standardization, inverse probability weighting, g-methods), is the rigorous base for causal questions in ML. Part 2 covers models; part 3 covers time-varying treatments, which is where traffic interventions live.

**Stanford grad.** Read alongside Pearl's structural approach (the Book of Why for the story, Causality for the math) and the modern causal ML literature (double machine learning, causal forests). The time-varying-treatment chapters are directly relevant to sequential alerting policies.

**PhD at Apple.** The epidemiological tradition of causal inference is the most careful one about assumptions; for a product that intervenes (alerts that change traffic), it is the framework for saying what you can and cannot learn from logged data.

#### [Introduction to Causal Inference (Brady Neal), course and book draft](https://www.bradyneal.com/causal-inference-course)

_course · free · weeks · free; causal inference for ML people, from zero_

**Age 5.** A free online class and book about telling causes from coincidences, taught from the very beginning with pictures of arrows.

**Middle school.** Brady Neal's free course (video lectures) and book draft introduce causal inference from a machine-learning perspective: potential outcomes, causal graphs, confounding, adjustment, and the main methods. It assumes only basic probability. It is the free entry point for the causal topic.

**High school.** Chapters 1 to 4 (motivation, potential outcomes, the flow of association and causation in graphs, causal models) give the vocabulary and the diagrams; chapters on identification and on instruments give the methods. Draw the traffic DAG after chapter 3.

**Stanford undergrad.** Neal presents Pearl's graphical approach and the potential-outcomes framework together and shows their correspondence, which the specialized books do less well; its later chapters connect to machine learning (causal discovery, counterfactuals).

**Stanford grad.** Read it as the bridge between the Mixtape's applied methods and Hernán and Robins's rigor; its treatment of the estimand and of identification is a clean template for writing your own causal question.

**PhD at Apple.** A free, ML-oriented causal course is the right shared reference for an engineering team beginning to ask what a system's outputs cause.

#### [Fairness and Machine Learning (Barocas, Hardt, Narayanan)](https://fairmlbook.org/)

_book · free · three weeks · free; the graduate text_

**Age 5.** A free book about making sure learning machines treat different groups of people fairly, and about how hard it is to even agree on what fair means.

**Middle school.** A textbook, free online, about fairness in machine learning: how models can treat groups differently, the different definitions of fairness and why they conflict, how discrimination is understood in law and philosophy, and what can be done. It is readable with basic statistics.

**High school.** Read the chapters on classification (the fairness criteria and the impossibility of satisfying them all), on causality, and on the legal background. Then audit the alert policy across neighborhoods using two criteria and see them disagree.

**Stanford undergrad.** The book's formal treatment (independence, separation, sufficiency; the trade-offs; measurement issues) is the rigorous base; its chapters on the limits of technical fixes and on broader questions are unusually honest for a technical text.

**Stanford grad.** It connects to causal inference (counterfactual fairness), to long-term dynamics of decisions, and to measurement of protected attributes; the bibliography is a map of the field.

**PhD at Apple.** The standard graduate text; for a lead deploying a public system, its framing of fairness as a socio-technical question, not a metric, is the one that survives contact with real decisions.

#### [The Algorithmic Foundations of Differential Privacy (Dwork, Roth)](https://www.cis.upenn.edu/~aaroth/privacybook.html)

_book · free · weeks · free from the authors; chapters 1 to 3_

**Age 5.** A book about adding a little randomness to shared numbers so that nobody can tell whether any one person's information was used.

**Middle school.** Cynthia Dwork and Aaron Roth's monograph, free from the authors, defines differential privacy: a way of releasing statistics with noise so that each person's data has a bounded effect on the result. It explains the basic mechanisms and how privacy loss adds up. The first chapters are readable; the rest is for specialists.

**High school.** Read chapters 1 to 3 (the promise, the definition, the Laplace and exponential mechanisms, composition) and implement a private release of hourly average speeds. Plot how accuracy changes with the privacy parameter.

**Stanford undergrad.** The monograph's treatment of composition, of sensitivity, and of the exponential mechanism is the foundation for private learning (DP-SGD) and for private data release. Understand what the guarantee does and does not cover (it protects individuals, not groups; it says nothing about the data being wrong).

**Stanford grad.** Later work covers private deep learning, local differential privacy (as used on devices), and privacy accounting; the monograph is the base for reading it. Its examples of failed anonymization are the argument for the definition.

**PhD at Apple.** Differential privacy is deployed at scale in operating systems and statistical agencies; the research frontier is utility at practical privacy budgets and privacy under continual release, which is exactly what a streaming traffic system does.

#### [Model Cards for Model Reporting (Mitchell et al., 2019) and Datasheets for Datasets (Gebru et al., 2018)](https://arxiv.org/abs/1810.03993)

_paper · free · an evening or two · free; documentation you will actually fill in_

**Age 5.** Two papers that say every learning machine and every pile of data should come with a label, like food has ingredients and warnings.

**Middle school.** Two short papers proposing standard documentation: a model card describes what a model is for, how it was evaluated, on whom, and where it fails; a datasheet describes how a dataset was collected, what it contains and how it should and should not be used. Both are free and have templates you can fill in.

**High school.** Fill in a datasheet for the Fremont PeMS and 511 data and a model card for the LightGBM alert model, using the papers' question lists. You will find questions you cannot answer; those are the gaps.

**Stanford undergrad.** The papers' value is the questions; they operationalize accountability without prescribing a metric. Compare with later documentation standards and with regulatory requirements for high-risk systems.

**Stanford grad.** Documentation is part of the technical-debt story (Sculley) and of governance (fairness book); the research question is whether documentation changes behavior, which later studies examine.

**PhD at Apple.** Model and data documentation are now expected in responsible deployment; for a lead, they are the artifacts a review board or regulator will ask for first.

#### [Google's Recommendation Systems course](https://developers.google.com/machine-learning/recommendation)

_course · free · weeks · free; the architecture of ranking systems_

**Age 5.** A free class from Google about how apps guess what you might like next.

**Middle school.** A short free course covering how recommendation works: candidate generation (finding possible items), scoring, and re-ranking; collaborative filtering and matrix factorization; embeddings; and the problems of cold start and bias. With small exercises in a notebook.

**High school.** Do the sections on collaborative filtering, matrix factorization and retrieval; then apply the ranking metrics (precision at k, NDCG) to the traffic dashboard's ordering of segments.

**Stanford undergrad.** The course is a compact tour of an industrial recommender's stages and their trade-offs; read it as architecture, then the papers for the methods (matrix factorization, two-tower retrieval, learning to rank).

**Stanford grad.** It is introductory and Google-flavored; the research literature on exposure bias, off-policy evaluation and fairness in ranking is where the hard questions live.

**PhD at Apple.** A useful shared reference for the structure of ranking systems; for a lead, its staged architecture is the design pattern for any system that shows people an ordered list.

#### [Recommender Systems chapter, Dive into Deep Learning](https://d2l.ai/chapter_recommender-systems/index.html)

_book · free · weeks · free; matrix factorization to ranking, with code_

**Age 5.** A free chapter, with code, about how apps guess what you might like next.

**Middle school.** The recommender-systems chapter of Dive into Deep Learning covers matrix factorization, autoencoders for ratings, implicit feedback, sequence-aware recommendation, feature-rich models and click-through prediction, each with runnable code. It replaces the paywalled matrix-factorization article with a free, complete treatment.

**High school.** Read the sections on matrix factorization and on implicit feedback (personalized ranking), and run their code on a small public ratings dataset; then compute precision at k and NDCG as the ranking topic asks.

**Stanford undergrad.** The chapter's coverage of Bayesian personalized ranking and of click-through models is the practical core of ranking with implicit feedback; its factorization machines section connects to the tabular models of phase 4.

**Stanford grad.** Its treatment is method-centric; pair with the exposure-bias and off-policy-evaluation literature for the evaluation problems, which the chapter does not solve.

**PhD at Apple.** A free, runnable reference for the classical recommender methods; useful as the baseline set when a team proposes something new.

#### [Hugging Face Deep RL course](https://huggingface.co/learn/deep-rl-course/unit0/introduction)

_course · free · four weeks · free; train agents hands-on_

**Age 5.** A free online class where you train little agents to play games and land a rocket, and see them get better.

**Middle school.** Hugging Face's free reinforcement-learning course: the foundations, Q-learning, deep Q-networks, policy gradients, actor-critic methods, and multi-agent settings, with hands-on units where you train agents in simulators and share them. It pairs practice with Sutton and Barto's theory.

**High school.** Do units 1 to 4 (introduction, Q-learning, deep Q-learning, policy gradient) alongside the RL topic; the hands-on notebooks run on a free cloud GPU or, with small settings, on the Mac. The unit on evaluation and on the reproducibility of RL results is short and important.

**Stanford undergrad.** The course uses standard libraries (Stable-Baselines3, Gymnasium) so you learn the practical stack; understand what the libraries hide (replay buffers, target networks, advantage estimation) by reading Spinning Up's implementations alongside.

**Stanford grad.** It is practice-first; its coverage of offline RL and of off-policy evaluation, which the alerting problem needs, is thin, so read those from the literature.

**PhD at Apple.** A free, maintained practical course; a good calibration of what a working RL baseline looks like before proposing a control experiment.

#### [Gymnasium documentation](https://gymnasium.farama.org/)

_docs · free · an afternoon · free; the environment interface, and how to build one_

**Age 5.** The manual for a set of practice worlds where a learning agent can try things safely, like a cart balancing a pole.

**Middle school.** Gymnasium is the maintained version of OpenAI Gym: a standard interface for reinforcement-learning environments (observe, act, reward, done) and a set of classic tasks. Its documentation explains the interface and how to build your own environment, which is what a traffic-control simulator would need.

**High school.** Read the basic usage page and the guide to making a custom environment; implement a tiny ramp-metering environment (state: queue and mainline occupancy; action: meter rate; reward: negative delay) and train a bandit or Q-learning agent on it.

**Stanford undergrad.** The environment API is the contract between algorithms and simulators; understand observation and action spaces, episode termination versus truncation, and vectorized environments for throughput.

**Stanford grad.** Wrapping a traffic simulator (SUMO) as a Gymnasium environment is the standard path to RL experiments in transport; the documentation's custom-environment guide is the template.

**PhD at Apple.** A shared interface made RL research comparable; the same lesson applies to any experimental platform you build.

#### [SUMO, Simulation of Urban MObility, documentation](https://sumo.dlr.de/docs/)

_docs · free · an afternoon · free; a city's traffic, car by car_

**Age 5.** The manual for a free program that simulates a whole city's traffic, car by car, so you can test ideas without touching real roads.

**Middle school.** SUMO is an open-source traffic simulator from the German Aerospace Center: road networks, vehicles with realistic car-following, traffic lights, ramp meters, and a Python interface to control and observe the simulation. Its tutorials build a small network and run it.

**High school.** Do the beginner tutorials (a simple network, adding vehicles, viewing the simulation), then import a piece of Fremont's road network from OpenStreetMap and simulate a rush hour with an incident. The Python interface (TraCI) lets you read speeds like PeMS would and change signals.

**Stanford undergrad.** SUMO is the environment for the traffic-control RL exercise and a source of synthetic data for stress-testing the forecaster; understand its car-following models (Krauss, IDM) and its calibration limits before trusting its numbers.

**Stanford grad.** Simulator-to-reality gaps are the central issue: calibrate demand and driver parameters against PeMS counts, and evaluate any learned controller under demand shifts. The documentation on calibration and on the TraCI interface is the practical reference.

**PhD at Apple.** Open simulators are where counterfactual questions about alerts and routing become answerable; SUMO with a learned forecaster and a control policy is a complete research platform on one machine.

#### [Fairlearn user guide](https://fairlearn.org/)

_docs · free · an afternoon · free; fairness assessment and mitigation in code_

**Age 5.** The manual for a free tool that checks whether a learning machine treats different groups of people fairly, and helps fix it.

**Middle school.** Fairlearn is an open-source Python library from Microsoft for assessing and improving fairness: metrics computed per group, dashboards, and algorithms that reduce disparities under constraints. Its user guide explains the fairness definitions it implements and their trade-offs.

**High school.** Read the user guide's sections on assessment (metric frames by group) and on mitigation, then compute recall and false-alarm rate per neighborhood for the alert policy and try one mitigation; see what it costs in overall precision.

**Stanford undergrad.** The guide's careful discussion of which fairness metric fits which harm, and of the limits of technical mitigation, is a practical companion to the fairness book; its implementations are correct references for the definitions.

**Stanford grad.** Fairness tooling encodes choices; use the library for measurement and treat its mitigations as experiments to be evaluated, not fixes to be applied.

**PhD at Apple.** A maintained open-source reference for fairness assessment; useful for standardizing audits across a team's models.

#### [Programming Differential Privacy (Near, Abuah)](https://programming-dp.com/)

_book · free · weeks · free; privacy mechanisms as runnable notebooks_

**Age 5.** A free online book that teaches how to add just enough randomness to shared numbers, with code you can run, so nobody can be picked out.

**Middle school.** A free book, written as runnable notebooks, that teaches differential privacy by doing: the Laplace mechanism, sensitivity, composition, private counts and averages, and private machine learning, all in Python on a real dataset. It is the practical companion to Dwork and Roth.

**High school.** Work through the chapters on the Laplace mechanism, on sensitivity and on composition, then release the hourly speed averages privately and plot accuracy against the privacy parameter as the topic asks. The chapter on differentially private gradient descent shows private model training.

**Stanford undergrad.** The book's worked examples make the definitions concrete and its exercises are checkable; its treatment of local differential privacy and of synthetic data connects to on-device settings and to the generative-model topic.

**Stanford grad.** Read it alongside the theory monograph: the book shows what the mechanisms do to real data, which is where the practical privacy-utility trade-off lives.

**PhD at Apple.** A free, executable textbook is the right onboarding for engineers who will implement private releases; its examples are a starting point for a team's own privacy library.

#### [Performative Prediction (Perdomo, Zrnic, Mendler-Dünner, Hardt, 2020)](https://arxiv.org/abs/2002.06673)

_paper · free · an evening or two · free; the theory of alerts that change traffic_

**Age 5.** A paper about what happens when a prediction changes the thing it predicted, like telling everyone a road will be slow so everyone avoids it and it is not slow after all.

**Middle school.** When a model's predictions influence the world, the data it learns from next time are shaped by its own past predictions. This paper names that situation and studies when retraining still settles down to a sensible model. Traffic alerts are a textbook case.

**High school.** Read the introduction and the examples; then the definition of a performatively stable model (one that stays optimal on the data it induces) and the conditions under which repeated retraining converges to one. Ask what it means for stage 13's monthly retrain.

**Stanford undergrad.** The framework formalizes feedback loops as a distribution map from model to data; the results relate convergence to how sensitive the distribution is to the model. It is the theory behind the practical warning in lessons 6 and 12.

**Stanford grad.** Subsequent work covers strategic behavior, optimization methods for performative risk, and connections to causal inference and reinforcement learning; for traffic, the open question is estimating the distribution map from logged alerts.

**PhD at Apple.** A foundational framing for any system that acts on its predictions; a research program on alerting policies would start here and with the causal-inference literature.

### Build

- Q-learning on a grid world and Thompson sampling on a three-arm bandit; formulate alerting as a contextual bandit and write the off-policy estimator.
- Draw the DAG for incidents, congestion, reporting, rain and alerts; estimate the effect of rain on slow episodes with inverse probability weighting; design a switchback experiment for alerts.
- Audit recall and false alarms per station on the map under two fairness definitions; release hourly speeds with differential privacy and plot utility against epsilon; fill in a model card and a datasheet.
- Train a LightGBM ranker to order segments by risk for the dashboard and evaluate NDCG against realized episodes.

### Done when

- A memo on what the logged data can and cannot say about the effect of alerts, with the experiment that would settle it.
- A fairness audit map and a filled model card.
- A ranked dashboard with NDCG next to sorting by probability.

### Pitfalls

- Adjusting for a collider.
- Treating feature importance as a causal claim.
- Believing one fairness metric settles the question.

## Phase 11: Depth

_weeks 70–76, then onward · open-ended · builds papers, PEMS-BAY, Bayesian models, write-ups_

**Goal.** From 'can build this' to 'can design the next one and say why it works'. Rotate the tracks; each is months, not weeks.

### What you need to learn

- Uncertainty: quantiles, Bayesian thinking, conformal prediction, adaptive conformal inference; risk control for alerts.
- Theory: generalization, bias-variance properly, convexity, margins, double descent; what breaks under shift.
- Bayesian machine learning: hierarchical models, MCMC and variational inference, Gaussian processes, Bayesian optimization.
- Reading and reproducing: DCRNN, STGCN, Graph WaveNet on PEMS-BAY at reduced scale; ablations; write-ups.
- Practice and communication: two competitions written up honestly, one open-source contribution, one public write-up of the traffic model, one talk.

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

### Topic: Bayesian machine learning and Gaussian processes

**Goal.** Treat parameters as uncertain, update beliefs with data, and use models that come with their own error bars.

**Age 5.**

Before you look, you have a guess. After you look, you have a better guess. Bayesian thinking is keeping track of how sure you are and changing it a little with every new clue.

_Try this._ Guess how many marbles are in a jar. Now someone tells you it is more than 50. Change your guess. Someone says fewer than 80. Change it again.

**Middle school.**

Bayes' rule updates a prior belief with data into a posterior. A Bayesian model does that for its parameters: instead of one best weight, it keeps a whole distribution of plausible weights, so its predictions come with uncertainty. A Gaussian process is a Bayesian model over functions: it predicts a curve and a band around it, wide where data are scarce.

_Try this._ Plot ten (time, speed) points for one station and sketch by hand a smooth curve with a band that is narrow near the points and wide between them.

**High school.**

Priors, likelihoods and posteriors; conjugate examples (Beta-Binomial for a station's slow rate); posterior predictive distributions; Bayesian linear and logistic regression; hierarchical models sharing strength across stations. Gaussian processes: a kernel defines similarity, predictions are closed-form with variance, cost grows as the cube of the data size, so they suit small data and hyperparameter tuning.

Computation: exact for conjugate cases, otherwise Markov chain Monte Carlo or variational inference (PyMC, Stan, or NumPyro).

_Try this._ Fit a Beta-Binomial model of each station's slow-episode rate with a shared prior, and compare the shrunken estimates with the raw rates for stations with few data.

**Stanford undergrad.**

Bayesian decision theory; model comparison and the marginal likelihood; MCMC (Metropolis, Hamiltonian Monte Carlo) and diagnostics; variational inference and the ELBO; Laplace approximation; Gaussian processes in depth (kernels, hyperparameter learning by marginal likelihood, sparse approximations). Bayesian optimization as the workhorse of hyperparameter tuning.

Bayesian neural networks and their approximations (dropout, ensembles, last-layer Laplace) for uncertainty in deep models.

_Try this._ Implement a Gaussian process regression for one station's daily profile with a periodic kernel and learn the hyperparameters by maximizing marginal likelihood; compare its predictive band with the bootstrap interval.

**Stanford grad.**

Scalable Bayesian inference (stochastic variational inference, inducing points), Bayesian nonparametrics, probabilistic programming, calibration of Bayesian predictive distributions under model misspecification, and the relation between Bayesian and conformal guarantees. Hierarchical spatio-temporal models as a classical alternative to graph networks.

_Try this._ Build a hierarchical Bayesian model of slow-episode rates with station and hour effects in a probabilistic programming language, check convergence, and compare its calibration with the boosted trees'.

**PhD at Apple.**

Bayesian methods underlie modern uncertainty quantification, experimental design, and Bayesian optimization at scale; the research frontier includes scalable inference for deep models, decision-making under uncertainty, and combining Bayesian and frequentist guarantees. For products, the practical value is honest uncertainty and principled pooling across sparse groups.

_Try this._ Argue, with a small experiment, whether a Bayesian hierarchical model or isotonic calibration on a boosted model gives better-calibrated per-station probabilities with three months of data.

**Check yourself.**

<details>
<summary>What does a prior do when data are scarce?</summary>

It dominates: the posterior stays near the prior until data accumulate, which is exactly the shrinkage that helps stations with few observations.

</details>

<details>
<summary>Why do Gaussian processes suit hyperparameter tuning?</summary>

They model a smooth unknown function with uncertainty from few evaluations, which is what Bayesian optimization needs to choose the next trial.

</details>

### Read and watch, explained at every level

#### [Probabilistic Machine Learning: An Introduction (Murphy)](https://probml.github.io/pml-book/book1.html)

_book · free · reference · free PDF; the modern reference_

**Age 5.** A very big free book that explains almost all of machine learning using the language of chance.

**Middle school.** Kevin Murphy's book, free as a PDF, about a thousand pages. It is a reference, not a read-through: look up a topic (logistic regression, decision trees, calibration, neural networks) and read that section. Its pictures are excellent and its examples are consistent from chapter to chapter.

**High school.** Use its chapters on probability and statistics (2 to 5), on linear models (9 to 11) and on trees and ensembles (18) as the rigorous version of ISLP. Its notation is consistent, and its habit of connecting every method to a likelihood shows that lessons 7, 8 and 9 are the same idea with different function classes.

**Stanford undergrad.** This is the graduate-level unification: every model as a probabilistic model, every loss as a negative log-likelihood, Bayesian and frequentist views side by side. Read the chapters on generalization, on optimization, and on calibration; its treatment of decision theory frames the alert threshold of lesson 11 properly.

**Stanford grad.** The second volume, Advanced Topics, covers inference, generative models and causality; between the two they are the modern reference set. Its treatment of Gaussian processes and Bayesian neural networks is the background for the uncertainty track of phase 8, and its chapters on time series connect state-space models to recurrent networks.

**PhD at Apple.** Murphy's books are the field's encyclopedia. Cite them for standard results and use them to find the original references quickly; for research, their breadth is what lets you place a new idea in the landscape in an afternoon rather than a week.

#### [The Elements of Statistical Learning (Hastie, Tibshirani, Friedman)](https://hastie.su.domains/ElemStatLearn/)

_book · free · reference · free PDF_

**Age 5.** A famous, hard, free book by three statisticians who invented many of the methods it describes.

**Middle school.** The graduate version of ISLP, free as a PDF, with proofs and depth. Read a chapter when you have met the topic in ISLP and want to know why it works: chapter 7 on judging models, chapter 10 on boosting, chapter 15 on random forests. Expect to read slowly and to reread.

**High school.** Chapter 10 is the origin story of gradient boosting written by its inventor, and it explains why fitting small trees to residuals works. Chapter 7 is the honest treatment of bias, variance and cross-validation, including why the test error estimate from cross-validation is itself uncertain. Chapter 15 explains why random forests work through variance reduction.

**Stanford undergrad.** ESL's derivations (boosting as additive modelling in function space, the bias-variance decomposition, effective degrees of freedom, shrinkage) are the background every ML PhD assumes. Work the exercises for chapters 7 and 10; they are the ones that make the later literature readable.

**Stanford grad.** Its coverage predates deep learning's rise and the modern boosting libraries, but its statistical framing (estimators, degrees of freedom, shrinkage, model selection) is what explains why those tools work. Pair chapter 10 with the XGBoost and LightGBM papers to see the second-order approximation that made boosting fast.

**PhD at Apple.** Still the canonical citation for many methods, and its authors' later work on sparsity and the lasso is the natural continuation. For a paper, its notation for loss functions and additive models is safe to adopt, and its chapter 7 remains the reference for arguing that a reported gain is within noise.

#### [Understanding Machine Learning: From Theory to Algorithms (Shalev-Shwartz, Ben-David)](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/)

_book · free · a semester, selectively · free PDF; learning theory with proofs_

**Age 5.** A free math book that asks: why can a machine that learned from old examples be right about new ones? And it proves the answers.

**Middle school.** A university textbook, free as a PDF, about the theory of learning: what it means to learn, when it is possible, and how much data you need. It is hard, and it is the right book when you want proofs rather than pictures. For the final phase.

**High school.** The first part (the PAC learning framework, VC dimension, bias-complexity trade-off) is the formal version of the theory ladder. Read chapters 2 to 6 slowly, doing the exercises; they make the words overfitting and generalization precise.

**Stanford undergrad.** Its coverage of uniform convergence, Rademacher complexity, regularization and stability, and convex learning gives the tools to read generalization bounds in papers. The chapters on boosting and on stochastic gradient descent connect the theory to the algorithms of phases 4 and 6.

**Stanford grad.** It is the standard reference for classical learning theory; modern questions (overparameterization, implicit bias) build on it and are covered in recent lecture notes and papers rather than textbooks. Read it with Boyd's convex optimization for the optimization side.

**PhD at Apple.** The base layer for theoretical work; its exercises are qualifying-exam material, and its treatment of stability is the entry point to the generalization theory of SGD.

#### [Convex Optimization (Boyd, Vandenberghe)](https://web.stanford.edu/~boyd/cvxbook/)

_book · free · a semester, selectively · free PDF and lectures_

**Age 5.** A free math book about finding the lowest point of a bowl-shaped hill, which is what a learning machine does when it learns.

**Middle school.** The standard textbook on convex optimization, free as a PDF, with lecture videos. It explains what makes a problem convex (a single lowest point), the methods that find it, and duality, a way of looking at the same problem from the other side. It is a graduate book; use it as a reference in the final phase.

**High school.** Chapters 2 to 5 (convex sets, convex functions, convex problems, duality) are the language behind why logistic regression has a global optimum and why regularization works. Chapter 9 on unconstrained minimization covers gradient descent and Newton's method with convergence proofs.

**Stanford undergrad.** Its treatment of gradient descent convergence, of conditioning, and of Newton and quasi-Newton methods gives the tools to reason about learning rates and about why second-order methods do not scale. The course's problem sets and the CVXPY library make it practical.

**Stanford grad.** Convexity is the regime where theory is complete; deep learning is the regime where it is not. Reading Boyd is how you know exactly what guarantees you give up, and its duality chapters are the background for support vector machines and for constrained learning (fairness constraints, monotonicity).

**PhD at Apple.** A canonical reference; its lecture videos are the fastest way for a strong engineer to acquire the optimization vocabulary that research papers assume.

#### [CS229 lecture notes (Stanford)](https://cs229.stanford.edu/main_notes.pdf)

_course · free · reference · free_

**Age 5.** The notes from a famous university class about learning machines, free for anyone to read.

**Middle school.** Stanford's machine learning course notes: linear regression, logistic regression, a family of related models called generalized linear models, models that describe how data is generated, support vector machines, and learning theory. Dense but clear. Read the first sections after phase 4, one section per sitting.

**High school.** The logistic regression and generalized-linear-model sections give lesson 7's math in the standard notation, including the derivation of the gradient that this course rebuilds by hand. The learning-theory notes on bias, variance and uniform convergence are the source material for the theory ladder in phase 8.

**Stanford undergrad.** The notes are the compact graduate treatment: exponential families unify linear, logistic and Poisson regression as one construction; the learning theory section introduces PAC-style bounds; the notes on regularization and model selection are practical. The public problem sets are worth doing for the theory sections.

**Stanford grad.** Compare CS229's derivations with ESL's, then read the newer notes on deep learning and on generalization. The course's treatment of the EM algorithm and of factor analysis is the background for latent-variable models of traffic regimes, which lesson 4's higher levels gesture at.

**PhD at Apple.** CS229 is what most ML researchers learned from, so knowing it lets you calibrate what a reviewer or a new hire assumes as background, and what they will not have seen (dependent data, forecasting protocols, calibration under shift) and will need explained.

#### [CS224W: Machine Learning with Graphs (Stanford)](https://web.stanford.edu/class/cs224w/)

_course · free · a quarter · free lectures_

**Age 5.** A university class, free to watch, about learning from things that are connected to each other, like roads that join.

**Middle school.** Jure Leskovec's course: how to represent a network of connected things, how to summarize each node with numbers, graph neural networks (where each node learns from its neighbors), and applications. The lectures are free on YouTube. Watch the graph-neural-network lectures after phase 6.

**High school.** The lectures on message passing (graph convolutional networks, GraphSAGE, graph attention) explain what neighbor mixing in the spatio-temporal net is doing and how to generalize it: attention over neighbors, more hops, learned edge weights. The lecture on spatio-temporal graphs connects directly to DCRNN.

**Stanford undergrad.** Understand graph convolutions as generalizations of image convolutions, the role of degree normalization, over-smoothing with depth, and the expressiveness limits given by the Weisfeiler-Leman test. A road is a directed graph, so the operators must be asymmetric, which is what DCRNN's forward and backward diffusion provides.

**Stanford grad.** The course's coverage of heterogeneous graphs, knowledge graphs and scalable training (neighbor sampling) is the toolkit for city-scale models, and its assignments in PyTorch Geometric are a practical route to implementing Graph WaveNet without writing every operator by hand.

**PhD at Apple.** CS224W and the PyTorch Geometric ecosystem are the standard entry to graph machine-learning research. Its open problems (expressiveness, scalability, dynamic graphs, transfer across graphs) overlap with traffic forecasting's own, and the course's reading list is a good map of the field.

#### [A Gentle Introduction to Conformal Prediction (Angelopoulos, Bates)](https://arxiv.org/abs/2107.07511)

_paper · free · an evening or two · free; intervals you can defend_

**Age 5.** A short guide about how to make a learning machine say 'I am 90 % sure the answer is in this range' and actually be right 90 % of the time.

**Middle school.** A tutorial paper, free, with code. It explains conformal prediction: measure how wrong your model is on data it has not seen, and use that to make ranges or sets of answers that hit the truth a guaranteed share of the time, no matter what model produced the predictions.

**High school.** Section 1 gives the recipe in a page: compute a score on calibration data, take a quantile of the scores, build the set. Later sections cover intervals for regression (conformalized quantile regression), sets for classification, and the assumptions. Apply the recipe to the 30-minute speed prediction and check coverage on the test split.

**Stanford undergrad.** The coverage guarantee follows from exchangeability in a few lines, which the tutorial proves. Understand why the guarantee is marginal rather than conditional, and what breaks under time-series dependence. The risk-control extension applies to controlling the false-alarm rate of a binary alert, which is lesson 11's problem.

**Stanford grad.** Follow with the authors' work on conformal risk control and with adaptive conformal inference for time series (Gibbs and Candès), and think about how to obtain guarantees for the episode-level false-alarm rate rather than the row-level one, which is what a driver experiences.

**PhD at Apple.** Conformal methods are the practical route to distribution-free guarantees in products. The research frontier, conditional coverage, dependence, and feedback between predictions and outcomes, is where the traffic alert problem sits, and the tutorial's clarity makes it the right document to hand to a product or legal partner.

#### [Adaptive Conformal Inference Under Distribution Shift (Gibbs, Candès, 2021)](https://arxiv.org/abs/2106.00170)

_paper · free · an evening or two · free; coverage that survives seasons_

**Age 5.** A paper about keeping the 'I am 90 % sure' promise even when the world keeps changing, by adjusting how wide the range is as you go.

**Middle school.** Conformal prediction promises a hit rate, but the promise assumes the future looks like the past. This paper adjusts the range online: if recent ranges missed too often, widen; if they hit too often, narrow. The result keeps the long-run hit rate even as the data shift, which is what traffic does across seasons.

**High school.** The method is a few lines: track whether each interval covered the truth, and update a single number that sets the next interval's width. Implement it on the rolling 30-minute speed predictions and plot the realized coverage week by week; compare with a fixed conformal interval.

**Stanford undergrad.** The guarantee is about long-run average coverage under arbitrary shift, achieved by an online update resembling gradient descent on the miscoverage rate; understand what it gives up (no per-period guarantee) and how the step size trades adaptivity for stability.

**Stanford grad.** Follow with later work on conformal methods for time series and on risk control; the connection to online learning (regret bounds) explains why the method works without distributional assumptions.

**PhD at Apple.** The practical entry point to valid uncertainty in non-stationary deployed systems; for an alerting product, this line of work is how you keep a false-alarm promise honest over months.

#### [Statistical Rethinking lectures and course materials (McElreath)](https://github.com/rmcelreath/stat_rethinking_2026)

_course · free · ten weeks · free lectures, slides and homework; the Bayesian track_

**Age 5.** Free video lessons that teach how to think carefully about chance and evidence, using stories, drawings and small models.

**Middle school.** Richard McElreath's Bayesian statistics course: the full lecture series is free on YouTube, and the course repository on GitHub has slides, homework and solutions, with code in several languages including Python ports. You build models by writing down what you believe, then update them with data.

**High school.** Watch the lectures on Bayesian inference, on causal diagrams and on multilevel models, and do the homework with the Python code. Then apply partial pooling to per-station slow rates, which is the topic's exercise.

**Stanford undergrad.** The lectures present hierarchical models with partial pooling (the answer to noisy per-station estimates) and causal thinking with DAGs in plain language; the homework, with solutions, is enough to learn the material without the book.

**Stanford grad.** The course's emphasis on generative models and posterior predictive checks is a different evaluation culture worth adopting; pair with Bayesian Data Analysis for depth.

**PhD at Apple.** McElreath's causal-first framing is influential in applied statistics; the free lectures make it accessible to a whole team.

#### [Probabilistic Programming and Bayesian Methods for Hackers (Davidson-Pilon)](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers)

_book · free · weeks · free; Bayesian models in code first_

**Age 5.** A free online book that teaches Bayesian thinking with code first and math second, so you can see beliefs update on a screen.

**Middle school.** Cameron Davidson-Pilon's book, free as notebooks, introduces Bayesian methods through probabilistic programming (PyMC): you describe a model in code, the library samples from the posterior, and you plot what you learned. Chapters cover Bayesian inference, sampling, priors, and practical examples.

**High school.** Chapters 1 to 3 (introduction, a little more on PyMC, opening the black box of MCMC) are enough to build the Beta-Binomial model of station slow rates and to see the posterior update as data arrive. The examples are small and fast.

**Stanford undergrad.** The book is intuition-first and light on theory; use it to become fluent with a probabilistic programming library before Bayesian Data Analysis makes the theory rigorous. Its chapter on priors and on loss functions connects to decision theory.

**Stanford grad.** Probabilistic programming is how applied Bayesian work is done now; the book is the fastest way to a working model, and its limits (no hierarchical depth, dated library versions in places) are easy to work around with the library's current docs.

**PhD at Apple.** A free, code-first introduction; useful for getting a team from Bayesian ideas to running models in a week.

#### [A Visual Exploration of Gaussian Processes (Distill, 2019)](https://distill.pub/2019/visual-exploration-gaussian-processes/)

_interactive · free · an hour · free, interactive; the kernel's effect, seen_

**Age 5.** A web page with pictures that move, showing a kind of model that draws a smooth curve through points and a band showing how unsure it is.

**Middle school.** A free interactive article on Distill: what a Gaussian process is, how the kernel controls the shape of the curves, how observing data narrows the band, and what the hyperparameters do. Play with the figures before reading the Gaussian-process textbook.

**High school.** Use the interactive figures to see how the kernel's length scale and the noise level change the fit, then reproduce the picture on one station's daily profile with a periodic kernel. The article's explanation of the prior over functions is the idea to keep.

**Stanford undergrad.** The article's treatment of conditioning a multivariate Gaussian is the whole computation behind a Gaussian process; with it, the textbook's chapter 2 reads as a formalization of what you saw.

**Stanford grad.** Distill's interactive format is the reference for explaining Bayesian nonparametrics; the article stops before sparse approximations and classification.

**PhD at Apple.** An exemplary explanation; the model for how to present a Bayesian method to an engineering audience.

#### [Foundations of Machine Learning (Mohri, Rostamizadeh, Talwalkar), 2nd edition](https://cs.nyu.edu/~mohri/mlbook/)

_book · free · reference · free PDF; learning theory with proofs, a second view_

**Age 5.** A free math book that proves why learning from examples works and how many examples you need.

**Middle school.** A graduate textbook on learning theory, free as a PDF from the authors: PAC learning, Rademacher complexity, support vector machines, kernels, boosting, online learning, and more, with proofs. Read it in the final phase alongside Shalev-Shwartz and Ben-David.

**High school.** Chapters 2 and 3 (the PAC framework, Rademacher complexity and VC dimension) are the theory topic's formal version; chapter 7 on boosting gives the margin theory behind lesson 8. Expect slow reading and do the exercises.

**Stanford undergrad.** Its treatment of online learning and of ranking (chapters 8 and 10) connects to the bandit and recommender topics; its proofs are complete and its notation consistent, which makes it a good reference for reading theory papers.

**Stanford grad.** Compare with Shalev-Shwartz and Ben-David: Mohri is stronger on kernels, boosting and online learning; the other on stability and convex learning. Together they cover classical learning theory.

**PhD at Apple.** A standard reference; its exercises are qualifying-exam material and its later chapters are entry points to research areas.

#### [PyTorch Geometric documentation](https://pytorch-geometric.readthedocs.io/)

_docs · free · an afternoon · free; graph networks without hand-written operators_

**Age 5.** The manual for a tool that lets a learning machine work on networks of connected things, like roads that join.

**Middle school.** PyTorch Geometric is a free library for graph neural networks: graph data structures, message-passing layers, standard datasets and models. Its introduction and tutorials show how to build a graph model in a few lines. Use it to implement the road-graph models after CS224W.

**High school.** Read the introduction by example and the tutorial on creating message-passing networks; then rebuild the two-hop neighbor mixing from lesson 10 as a PyG layer and compare with the hand-written version.

**Stanford undergrad.** Understand the library's data model (edge index, batching of graphs), its implementations of the standard layers, and its handling of directed graphs, which the road network needs; the temporal extension covers spatio-temporal models.

**Stanford grad.** PyG's model zoo includes the graph models from the traffic papers or their components, which makes reproduction faster and fairer; read the implementations to check them against the papers.

**PhD at Apple.** The standard toolkit for graph learning research; for a lead, it is the argument for not writing graph operators by hand past the learning stage.

#### [Bayesian Data Analysis, 3rd edition (Gelman, Carlin, Stern, Dunson, Vehtari, Rubin)](https://sites.stat.columbia.edu/gelman/book/)

_book · free · reference · free PDF; the applied Bayesian reference_

**Age 5.** A big free book about learning from data by keeping track of what you believed before and how the data changed it.

**Middle school.** The standard graduate textbook on Bayesian statistics, free as a PDF from the authors. Part 1 covers the basics, part 2 the computation (sampling methods), part 3 regression and hierarchical models. It is demanding; use it as the reference after Statistical Rethinking.

**High school.** Read the chapters on single-parameter models and on hierarchical models, and reproduce the classic eight-schools example in a probabilistic programming library. Then apply the same hierarchical model to per-station slow-episode rates.

**Stanford undergrad.** Its treatment of hierarchical models, of model checking (posterior predictive checks), and of Markov chain Monte Carlo diagnostics is the rigorous base for applied Bayesian work; the chapters on decision analysis and on model comparison connect to calibration and to model selection.

**Stanford grad.** Together with the authors' later work (Stan, workflow papers) it defines current Bayesian practice; its chapter on nonparametric models and on missing data are advanced references.

**PhD at Apple.** The canonical applied Bayesian text; for research, its emphasis on model checking and workflow is the counterweight to treating inference as a black box.

#### [Gaussian Processes for Machine Learning (Rasmussen, Williams)](https://gaussianprocess.org/gpml/)

_book · free · two chapters, a week · free; models with their own error bars_

**Age 5.** A free book about a kind of model that draws a smooth curve through data and also draws a band that shows how unsure it is.

**Middle school.** The standard textbook on Gaussian processes, free online. Chapters 2 and 3 explain regression and classification with Gaussian processes, chapter 4 the kernels that define what 'similar' means, and chapter 5 how to choose them. It is mathematical but self-contained.

**High school.** Read chapters 2 and 4 and implement Gaussian process regression on one station's daily profile with a periodic kernel. The predictive band is the picture to keep in mind whenever someone asks how sure the model is.

**Stanford undergrad.** The book's treatment of the marginal likelihood for hyperparameter selection, of the connection to kernel methods and to Bayesian linear regression, and of approximations for classification is the foundation for Bayesian optimization and for modern probabilistic models.

**Stanford grad.** Its coverage predates scalable approximations (inducing points, variational methods) and deep kernels; read those from the recent literature. The chapter on model selection and the appendix on Gaussian identities are lasting references.

**PhD at Apple.** The canonical reference; Gaussian processes remain the tool of choice for expensive-function optimization and for small-data science, and their theory underlies much of modern uncertainty quantification.

#### [Deep Learning (Goodfellow, Bengio, Courville)](https://www.deeplearningbook.org/)

_book · free · reference · free online_

**Age 5.** The big classic book about deep neural networks, free online, by three of the people who built the field.

**Middle school.** Three parts: the math background, practical deep networks (feedforward nets, regularization, optimization, convolutional and recurrent networks), and research topics. Read chapters from the second part as references after phase 6, when the words will already be familiar.

**High school.** Chapter 6 on feedforward networks and backpropagation and chapter 8 on optimization are the rigorous versions of lesson 9; chapter 10 on sequence modelling covers recurrent networks and gated units; chapter 7 on regularization explains early stopping, dropout and weight decay and why they work.

**Stanford undergrad.** It is the theory reference: backpropagation as a general algorithm on computation graphs, the difficulties of optimization (saddle points, ill-conditioning), and the regularization view of generalization. It predates transformers and modern training recipes, so supplement with Dive into Deep Learning and recent papers.

**Stanford grad.** Part three (representation learning, structured probabilistic models, generative models) frames later research. Some material is dated, but the conceptual chapters remain the shared vocabulary of the field, and its chapter on practical methodology is a good checklist before any training run.

**PhD at Apple.** A canonical citation and a snapshot of the field as of 2016. Knowing what it lacks (attention, scaling laws, modern generative models, the empirical understanding of generalization) is as important as knowing what it contains, and reading it against Murphy's second volume shows how the field's framing has shifted.

#### [Graph WaveNet (Wu et al., IJCAI 2019)](https://arxiv.org/abs/1906.00121)

_paper · free · an evening or two · free; the third traffic paper_

**Age 5.** A paper where scientists let the computer figure out for itself which roads affect which, instead of telling it.

**Middle school.** The third traffic paper. Instead of a fixed map of neighbors, Graph WaveNet learns which sensors influence which from the data (an adaptive adjacency), and it uses stacked convolutions over time instead of a recurrent network. It improved on DCRNN and STGCN on the same benchmarks.

**High school.** Read it after DCRNN and STGCN: what changes is the graph (learned from node embeddings) and the temporal model (dilated causal convolutions, cheaper to train than recurrence). Its ablation shows the learned graph helps even when a physical graph is available, which is a hint that the postmile line misses some dependencies.

**Stanford undergrad.** The adaptive adjacency, a softmax over the product of two learned embedding matrices, is a learned low-rank graph; combined with the diffusion operators it captures dependencies the road line misses, such as interchanges and parallel routes. Dilated convolutions give a long receptive field with few layers, which is why it trains fast.

**Stanford grad.** Later work (MTGNN, spatio-temporal transformers) generalizes the learned-graph idea. Evaluate whether learned graphs are interpretable and stable across retrains, which is a practical concern for monitoring: a graph that changes every month is hard to trust and hard to explain.

**PhD at Apple.** For Fremont the learned graph could reveal cross-freeway coupling (I-880 to I-680 through SR-84 and Mission), and testing whether that improves 60-minute skill is a concrete, publishable experiment that fits on the M1. The paper is also a good example of an ablation done properly.

#### [METR-LA and PEMS-BAY datasets (DCRNN repository)](https://github.com/liyaguang/DCRNN)

_dataset · free · days · free; the benchmark to reproduce on_

**Age 5.** Two ready-made piles of road sensor data that scientists all use to test their traffic guessers, so everyone can compare fairly.

**Middle school.** The DCRNN paper's authors published their code and two datasets on GitHub: METR-LA (207 sensors in Los Angeles) and PEMS-BAY (325 sensors in the Bay Area, from PeMS like our data). Everyone since has tested on them. Download them in the final phase to reproduce a paper on the Mac mini.

**High school.** The repository contains the data files, the adjacency matrices built from road distances, and scripts to create the 12-step windows and the train, validation and test splits. Study how the splits are made and how missing values are masked; that protocol is what every paper's table assumes.

**Stanford undergrad.** Use the datasets to reproduce DCRNN and STGCN at reduced scale and to test your own spatio-temporal net under the standard protocol, so your numbers are comparable with the literature. Note the differences from Fremont: no incidents, no weather, a fixed sensor set, and speed only.

**Stanford grad.** The benchmark's known issues (data leakage through normalization statistics in some codebases, sensitivity to the graph construction) are documented in later papers; reading those is part of using the benchmark responsibly.

**PhD at Apple.** Shared benchmarks made the field comparable and also narrowed it; extending the protocol with incidents and weather, and evaluating episode-level metrics, is a contribution in itself.

#### [ETA Prediction with Graph Neural Networks in Google Maps (Derrow-Pinion et al., 2021)](https://arxiv.org/abs/2108.11482)

_paper · free · an evening or two · free; the industrial version_

**Age 5.** A paper about how a maps app guesses how long a drive will take, by looking at the roads as a connected network and learning from millions of trips.

**Middle school.** A research paper describing the graph neural network behind travel-time estimates in Google Maps. Roads are grouped into segments and supersegments, a network learns how traffic on connected pieces relates, and the model runs at enormous scale. It is the industrial version of what the spatio-temporal net does for Fremont.

**High school.** Read the system description: the supersegment graph, the features (segment attributes, real-time speeds, time), the graph network with message passing, and the training tricks (MetaGradients, ensembling). Compare its evaluation (travel-time error across cities) with the row-level metrics of this project.

**Stanford undergrad.** The paper is a case study in making graph learning work in production: stability across retrains, latency, and the engineering around a model that is a small part of a larger system. Its ablations show which components mattered, which is the discipline to copy.

**Stanford grad.** Contrast the probe-data setting (trip-based, sparse, privacy-constrained) with fixed-detector data; the modelling choices differ accordingly. Its treatment of training stability is relevant to any graph model retrained monthly.

**PhD at Apple.** One of few public accounts of a deployed spatio-temporal graph model at scale; it frames the open problems (transfer across regions, non-stationarity, coupling with routing) that a research agenda in this area would address.

### Build

- A PEMS-BAY reproduction that runs on the mini, with one ablation.
- A conformal interval on the 30-minute prediction with realized coverage tracked weekly.
- A hierarchical Bayesian model of per-station slow rates compared with isotonic calibration.
- A public write-up of the Fremont model, limitations first.

### Done when

- You read a new traffic-forecasting paper in an evening and say what it would change in stage 9 and what it would not fix.
- The write-up is public and someone else has run your code.

### Pitfalls

- Collecting courses.
- Never finishing a write-up.
- Optimizing a leaderboard instead of understanding.

## How to study so it sticks

- **A weekly rhythm at ten hours.** Three hours of reading or lectures with notes. Five hours building. One hour rewriting the week's notes from memory. One hour explaining it to someone, or to a file if no one is around.
- **The three passes.** Run it. Change it and predict what will happen before you run it. Rebuild it from a blank file. Only the third pass counts as knowing.
- **A lab notebook.** `notebook/YYYY-WW.md`: what you tried, what broke, what you learned, the number. Numbers without the setup that produced them are noise a month later.
- **A baseline next to every result.** Persistence, the profile rate, the previous version. No exceptions, including for plots.
- **What would fool this metric?.** Ask it of every number. Accuracy under imbalance, AUC with persistence, F1 without a threshold rule, a shuffled split on a time series.
- **Read in this order.** Code, then docs, then papers, then blog posts. Blog posts last because they are the easiest to write badly.
- **The stuck rule.** Thirty minutes alone. Then write the question down precisely, which solves half of them. Then ask, with the exact error and what you already tried.
- **One project.** The traffic model is enough for the first year. Later phases add small side projects only where the traffic data cannot carry the idea.
- **Rest.** Two long sessions beat five short ones for building; five short ones beat two long ones for math. Sleep is when it consolidates.

### Honest notes

- Every link was checked in September 2026. Editions and chapter numbers drift; if a chapter does not match its description, find the topic in the index.
- The hours assume a beginner. If a phase takes twice as long, take it. Reorder nothing before phase 8; the later phases can be taken in any order.
- The math in phase 2 is the part people skip and the part that separates using ML from understanding it. Derive the gradient once by hand; everything in phases 7 and 8 is that derivation repeated.
- Being good at ML is mostly being good at evaluation. Phases 4 to 6 are where that is learned; do not shorten them to reach the neural networks sooner.
- The traffic project cannot carry vision, language or reinforcement learning; those topics use small public datasets and simulators, chosen to run on the Mac mini.

## The library

All free; a few need a free login or token.

| Title | Kind | Access | Time | Phase |
|---|---|---|---|---|
| [Software Carpentry: The Unix Shell, Version Control with Git](https://swcarpentry.github.io/shell-novice/) | course | free | a day | 0 |
| [Pro Git, chapters 1–3](https://git-scm.com/book/en/v2) | book | free | a weekend | 0 |
| The repo's own `00_CONCEPTS.md` and `05_PLAN_COMBINED_DATASETS.md` | docs | free | an hour, twice | 0 |
| [The Missing Semester of Your CS Education (MIT)](https://missing.csail.mit.edu/) | course | free | a week | 0 |
| [Elements of AI (University of Helsinki)](https://www.elementsofai.com/) | course | free | about 30 hours | 0 |
| [Automate the Boring Stuff with Python, chapters 1–13 and 16–18](https://automatetheboringstuff.com/) | book | free | weeks | 1 |
| [The official Python tutorial, sections 3–9](https://docs.python.org/3/tutorial/) | docs | free | an afternoon | 1 |
| [Kaggle Learn: Python](https://www.kaggle.com/learn/python) | course | free | a weekend | 1 |
| [Exercism, Python track](https://exercism.org/tracks/python) | interactive | free | ongoing, a few a week | 1 |
| [CS50's Introduction to Programming with Python (Harvard)](https://cs50.harvard.edu/python/) | course | free | ten weeks part-time | 1 |
| [The `zoneinfo` and `datetime` documentation](https://docs.python.org/3/library/zoneinfo.html) | docs | free | an hour | 1 |
| [Khan Academy: Linear algebra, Differential calculus, Statistics and probability](https://www.khanacademy.org/math) | course | free | weeks | 2 |
| [3Blue1Brown: Essence of linear algebra, Essence of calculus](https://www.3blue1brown.com/topics/linear-algebra) | video | free | an afternoon each | 2 |
| [Immersive Linear Algebra (Ström, Åström, Akenine-Möller)](https://immersivemath.com/ila/) | interactive | free | a few evenings | 2 |
| [Seeing Theory (Brown University)](https://seeing-theory.brown.edu/) | interactive | free | an evening | 2 |
| [Introduction to Probability (Blitzstein, Hwang) and Stat 110](https://probabilitybook.net/) | book | free | a semester | 2 |
| [Mathematics for Machine Learning (Deisenroth, Faisal, Ong), chapters 2–7](https://mml-book.github.io/) | book | free | reference | 2 |
| [Think Stats (Downey)](https://allendowney.github.io/ThinkStats/) | book | free | two weeks | 2 |
| [MIT 18.06 Linear Algebra (Strang), OpenCourseWare](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/) | course | free | ten lectures, two weeks | 2 |
| [OpenIntro Statistics](https://www.openintro.org/book/os/) | book | free | weeks | 2 |
| [Python for Data Analysis, 3rd edition (McKinney), chapters 4–8 and 10–11](https://wesmckinney.com/book/) | book | free | weeks | 3 |
| [pandas user guide: Time series and date functionality](https://pandas.pydata.org/docs/user_guide/timeseries.html) | docs | free | an afternoon | 3 |
| [Kaggle Learn: Pandas, Data Cleaning, Data Visualization](https://www.kaggle.com/learn/pandas) | course | free | an afternoon each | 3 |
| [The polars user guide, first three chapters](https://docs.pola.rs/) | docs | free | an afternoon | 3 |
| [SQLBolt, interactive SQL lessons](https://sqlbolt.com/) | interactive | free | a weekend | 3 |
| [DuckDB documentation, the Python API and SQL reference](https://duckdb.org/docs/) | docs | free | an afternoon | 3 |
| [PeMS documentation and the Clearinghouse (Caltrans)](https://pems.dot.ca.gov/) | docs | free (login) | an afternoon | 3 |
| [Open-Meteo, historical weather API documentation](https://open-meteo.com/en/docs/historical-weather-api) | docs | free | an hour | 3 |
| [511 SF Bay Open Data, traffic API documentation](https://511.org/open-data) | docs | free (token) | an hour | 3 |
| [Python Data Science Handbook (VanderPlas)](https://jakevdp.github.io/PythonDataScienceHandbook/) | book | free | weeks | 3 |
| [Fundamentals of Data Visualization (Wilke)](https://clauswilke.com/dataviz/) | book | free | weeks | 3 |
| [Kaggle Learn: Intro to SQL, Advanced SQL](https://www.kaggle.com/learn/intro-to-sql) | course | free | an afternoon each | 3 |
| [Select Star SQL](https://selectstarsql.com/) | interactive | free | two hours | 3 |
| [Machine Learning Specialization (Andrew Ng), courses 1 and 2](https://www.coursera.org/specializations/machine-learning-introduction) | course | free (audit) | six weeks | 4 |
| [An Introduction to Statistical Learning with Python (ISLP), chapters 2–4 and 8](https://www.statlearning.com/) | book | free | three weeks | 4, 5 |
| [Google's Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course) | course | free | about 15 hours | 4 |
| [scikit-learn: Getting Started and the tutorials](https://scikit-learn.org/stable/getting_started.html) | docs | free | an afternoon | 4 |
| [R2D3: A Visual Introduction to Machine Learning](http://www.r2d3.us/visual-intro-to-machine-learning-part-1/) | interactive | free | half an hour | 4 |
| [MLU-Explain (Amazon)](https://mlu-explain.github.io/) | interactive | free | an hour per essay | 4 |
| [StatQuest with Josh Starmer (YouTube)](https://statquest.org/) | video | free | hours, as needed | 4, 5 |
| [Kaggle Learn: Intro to ML, Intermediate ML, Feature Engineering](https://www.kaggle.com/learn/intro-to-machine-learning) | course | free | weeks | 4 |
| [LightGBM documentation: Parameters, Parameters Tuning](https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html) | docs | free | an hour | 4 |
| [The boosting papers: Friedman (2001), XGBoost (2016), LightGBM (2017)](https://arxiv.org/abs/1603.02754) | paper | free | an evening or two | 4 |
| [Why do tree-based models still outperform deep learning on tabular data? (Grinsztajn, Oyallon, Varoquaux, 2022)](https://arxiv.org/abs/2207.08815) | paper | free | an evening or two | 4 |
| [Interpretable Machine Learning (Molnar), the SHAP chapter](https://christophm.github.io/interpretable-ml-book/shap.html) | book | free | an evening | 4 |
| [scikit-learn user guide: Model evaluation, Probability calibration](https://scikit-learn.org/stable/modules/model_evaluation.html) | docs | free | an evening | 4 |
| [scikit-learn user guide: Supervised and unsupervised learning](https://scikit-learn.org/stable/supervised_learning.html) | docs | free | an afternoon | 5 |
| [Google's Clustering course](https://developers.google.com/machine-learning/clustering) | course | free | a few hours | 5 |
| [Information Theory, Inference, and Learning Algorithms (MacKay)](https://www.inference.org.uk/mackay/itila/) | book | free | a semester, selectively | 5 |
| [Random Search for Hyper-Parameter Optimization (Bergstra, Bengio, 2012)](https://jmlr.org/papers/v13/bergstra12a.html) | paper | free | an evening or two | 5 |
| [Optuna documentation](https://optuna.readthedocs.io/) | docs | free | an afternoon | 5 |
| [Visual Information Theory (Olah)](https://colah.github.io/posts/2015-09-Visual-Information/) | essay | free | an hour | 5 |
| [Forecasting: Principles and Practice, the Python edition (Hyndman, Athanasopoulos), chapters 1–5, 7, 9](https://otexts.com/fpppy/) | book | free | three weeks | 6 |
| [Kaggle Learn: Time Series](https://www.kaggle.com/learn/time-series) | course | free | half a day | 6 |
| [statsmodels documentation: one ARIMA fit on one station](https://www.statsmodels.org/stable/tsa.html) | docs | free | an afternoon | 6 |
| [traffic-simulation.de and its model documentation (Treiber, Kesting)](https://traffic-simulation.de/) | interactive | free | an evening | 6 |
| [Kaggle: Store Sales, Time Series Forecasting competition](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) | dataset | free | a week | 6 |
| [The M5 forecasting competition and its findings (Makridakis, Spiliotis, Assimakopoulos)](https://www.kaggle.com/competitions/m5-forecasting-accuracy) | dataset | free | days | 6 |
| [3Blue1Brown: Neural networks](https://www.3blue1brown.com/topics/neural-networks) | video | free | two hours | 7 |
| [TensorFlow Playground](https://playground.tensorflow.org/) | interactive | free | fifteen minutes | 7 |
| [Neural Networks: Zero to Hero (Karpathy), micrograd and makemore](https://karpathy.ai/zero-to-hero.html) | video | free | three weeks | 7 |
| [PyTorch tutorials: Learn the Basics; the MPS backend notes](https://pytorch.org/tutorials/beginner/basics/intro.html) | docs | free | an afternoon | 7 |
| [Dive into Deep Learning (d2l.ai), chapters 3–5 and 9–10](https://d2l.ai/) | book | free | reference, chapters as needed | 7, 8 |
| [Understanding Deep Learning (Prince)](https://udlbook.github.io/udlbook/) | book | free | reference, chapters as needed | 7, 8 |
| [Practical Deep Learning for Coders (fast.ai)](https://course.fast.ai/) | course | free | eight weeks | 7, 8 |
| [A Gentle Introduction to Graph Neural Networks (Distill, 2021)](https://distill.pub/2021/gnn-intro/) | interactive | free | an evening | 7 |
| [DCRNN (Li et al., ICLR 2018) and STGCN (Yu et al., IJCAI 2018)](https://arxiv.org/abs/1707.01926) | paper | free | an evening or two | 7 |
| [Neural Networks and Deep Learning (Nielsen)](http://neuralnetworksanddeeplearning.com/) | book | free | a week | 7 |
| [The Little Book of Deep Learning (Fleuret)](https://fleuret.org/francois/lbdl.html) | book | free | a weekend | 7 |
| [A Recipe for Training Neural Networks (Karpathy)](https://karpathy.github.io/2019/04/25/recipe/) | essay | free | an hour | 7 |
| [UvA Deep Learning Tutorials](https://uvadlc-notebooks.readthedocs.io/) | docs | free | an afternoon | 7 |
| [The training-recipe papers: Adam (Kingma, Ba, 2014), Batch Normalization (Ioffe, Szegedy, 2015), Dropout (Srivastava et al., 2014)](https://arxiv.org/abs/1412.6980) | paper | free | an evening or two | 7 |
| [CS231n: Convolutional Neural Networks for Visual Recognition, course notes](https://cs231n.github.io/) | course | free | weeks | 8 |
| [The Illustrated Transformer (Alammar)](https://jalammar.github.io/illustrated-transformer/) | essay | free | an hour | 8 |
| [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) | paper | free | an evening or two | 8 |
| [Hugging Face LLM course](https://huggingface.co/learn/llm-course/chapter1/1) | course | free | two weeks | 8 |
| [Speech and Language Processing, 3rd edition draft (Jurafsky, Martin)](https://web.stanford.edu/~jurafsky/slp3/) | book | free | reference | 8 |
| [LLMs-from-scratch repository (Raschka)](https://github.com/rasbt/LLMs-from-scratch) | code | free | two weeks | 8 |
| [Intro to Large Language Models (Karpathy, video)](https://www.youtube.com/watch?v=zjkBMFhNj_g) | video | free | one hour | 8 |
| [Stanford CS224N: Natural Language Processing with Deep Learning](https://web.stanford.edu/class/cs224n/) | course | free | a quarter | 8 |
| [Hugging Face Computer Vision course](https://huggingface.co/learn/computer-vision-course/unit0/welcome/welcome) | course | free | weeks | 8 |
| [The generative-model papers: VAE (Kingma, Welling, 2013), GAN (Goodfellow et al., 2014), DDPM (Ho, Jain, Abbeel, 2020)](https://arxiv.org/abs/1312.6114) | paper | free | an evening or two | 8 |
| [Self-Supervised Representation Learning (Weng, Lil'Log)](https://lilianweng.github.io/posts/2019-11-10-self-supervised/) | essay | free | an hour | 8 |
| [Stanford CS336: Language Modeling from Scratch](https://cs336.stanford.edu/) | course | free | a quarter | 8 |
| [What are Diffusion Models? (Weng, Lil'Log)](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) | essay | free | an hour | 8 |
| [CS 329S: Machine Learning Systems Design, lecture notes (Huyen, Stanford)](https://stanford-cs329s.github.io/) | course | free | weeks | 9 |
| [MLflow documentation](https://mlflow.org/docs/latest/) | docs | free | an afternoon | 9 |
| [Evidently, open-source ML monitoring documentation](https://docs.evidentlyai.com/) | docs | free | an afternoon | 9 |
| [The Turing Way](https://book.the-turing-way.org/) | book | free | weeks | 9 |
| [Site Reliability Engineering (Google), the free book](https://sre.google/sre-book/table-of-contents/) | book | free | selected chapters, a week | 9 |
| [Hypothesis documentation (property-based testing)](https://hypothesis.readthedocs.io/) | docs | free | an afternoon | 9 |
| [Made With ML (Mohandas)](https://madewithml.com/) | course | free | weeks | 9 |
| [Machine Learning Engineering (Burkov)](https://www.mlebook.com/) | book | free | weeks | 9 |
| [Full Stack Deep Learning](https://fullstackdeeplearning.com/) | course | free | weeks | 9 |
| [Rules of Machine Learning (Zinkevich, Google)](https://developers.google.com/machine-learning/guides/rules-of-ml) | docs | free | an hour | 9 |
| [The ML Test Score (Breck, Cai, Nielsen, Salib, Sculley, 2017)](https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/) | paper | free | an hour | 9 |
| [Hidden Technical Debt in Machine Learning Systems (Sculley et al., 2015)](https://papers.nips.cc/paper_files/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html) | paper | free | an hour, twice | 9 |
| [launchd.info and the pytest documentation](https://www.launchd.info/) | docs | free | an afternoon | 9 |
| [FastAPI documentation](https://fastapi.tiangolo.com/) | docs | free | an afternoon | 9 |
| [Docker, Get started guide](https://docs.docker.com/get-started/) | docs | free | an afternoon | 9 |
| [PyTorch distributed overview](https://pytorch.org/tutorials/beginner/dist_overview.html) | docs | free | an afternoon | 9 |
| [The Ultra-Scale Playbook: Training LLMs on GPU Clusters (Hugging Face)](https://huggingface.co/spaces/nanotron/ultrascale-playbook) | book | free | weeks | 9 |
| [Scaling Laws for Neural Language Models (Kaplan et al., 2020)](https://arxiv.org/abs/2001.08361) | paper | free | an evening or two | 9 |
| [Training Compute-Optimal Large Language Models (Hoffmann et al., 2022)](https://arxiv.org/abs/2203.15556) | paper | free | an evening or two | 9 |
| [Reinforcement Learning: An Introduction (Sutton, Barto)](http://incompleteideas.net/book/the-book-2nd.html) | book | free | a semester | 10 |
| [David Silver's Reinforcement Learning course (UCL, DeepMind)](https://www.davidsilver.uk/teaching/) | video | free | ten lectures, two weeks | 10 |
| [Spinning Up in Deep RL (OpenAI)](https://spinningup.openai.com/) | docs | free | a week | 10 |
| [Causal Inference: The Mixtape (Cunningham)](https://mixtape.scunning.com/) | book | free | selected chapters, two weeks | 10 |
| [Causal Inference: What If (Hernán, Robins)](https://miguelhernan.org/whatifbook) | book | free | part 1, three weeks | 10 |
| [Introduction to Causal Inference (Brady Neal), course and book draft](https://www.bradyneal.com/causal-inference-course) | course | free | weeks | 10 |
| [Fairness and Machine Learning (Barocas, Hardt, Narayanan)](https://fairmlbook.org/) | book | free | three weeks | 10 |
| [The Algorithmic Foundations of Differential Privacy (Dwork, Roth)](https://www.cis.upenn.edu/~aaroth/privacybook.html) | book | free | weeks | 10 |
| [Model Cards for Model Reporting (Mitchell et al., 2019) and Datasheets for Datasets (Gebru et al., 2018)](https://arxiv.org/abs/1810.03993) | paper | free | an evening or two | 10 |
| [Google's Recommendation Systems course](https://developers.google.com/machine-learning/recommendation) | course | free | weeks | 10 |
| [Recommender Systems chapter, Dive into Deep Learning](https://d2l.ai/chapter_recommender-systems/index.html) | book | free | weeks | 10 |
| [Hugging Face Deep RL course](https://huggingface.co/learn/deep-rl-course/unit0/introduction) | course | free | four weeks | 10 |
| [Gymnasium documentation](https://gymnasium.farama.org/) | docs | free | an afternoon | 10 |
| [SUMO, Simulation of Urban MObility, documentation](https://sumo.dlr.de/docs/) | docs | free | an afternoon | 10 |
| [Fairlearn user guide](https://fairlearn.org/) | docs | free | an afternoon | 10 |
| [Programming Differential Privacy (Near, Abuah)](https://programming-dp.com/) | book | free | weeks | 10 |
| [Performative Prediction (Perdomo, Zrnic, Mendler-Dünner, Hardt, 2020)](https://arxiv.org/abs/2002.06673) | paper | free | an evening or two | 10 |
| [Probabilistic Machine Learning: An Introduction (Murphy)](https://probml.github.io/pml-book/book1.html) | book | free | reference | 11 |
| [The Elements of Statistical Learning (Hastie, Tibshirani, Friedman)](https://hastie.su.domains/ElemStatLearn/) | book | free | reference | 11 |
| [Understanding Machine Learning: From Theory to Algorithms (Shalev-Shwartz, Ben-David)](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/) | book | free | a semester, selectively | 11 |
| [Convex Optimization (Boyd, Vandenberghe)](https://web.stanford.edu/~boyd/cvxbook/) | book | free | a semester, selectively | 11 |
| [CS229 lecture notes (Stanford)](https://cs229.stanford.edu/main_notes.pdf) | course | free | reference | 11 |
| [CS224W: Machine Learning with Graphs (Stanford)](https://web.stanford.edu/class/cs224w/) | course | free | a quarter | 11 |
| [A Gentle Introduction to Conformal Prediction (Angelopoulos, Bates)](https://arxiv.org/abs/2107.07511) | paper | free | an evening or two | 11 |
| [Adaptive Conformal Inference Under Distribution Shift (Gibbs, Candès, 2021)](https://arxiv.org/abs/2106.00170) | paper | free | an evening or two | 11 |
| [Statistical Rethinking lectures and course materials (McElreath)](https://github.com/rmcelreath/stat_rethinking_2026) | course | free | ten weeks | 11 |
| [Probabilistic Programming and Bayesian Methods for Hackers (Davidson-Pilon)](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers) | book | free | weeks | 11 |
| [A Visual Exploration of Gaussian Processes (Distill, 2019)](https://distill.pub/2019/visual-exploration-gaussian-processes/) | interactive | free | an hour | 11 |
| [Foundations of Machine Learning (Mohri, Rostamizadeh, Talwalkar), 2nd edition](https://cs.nyu.edu/~mohri/mlbook/) | book | free | reference | 11 |
| [PyTorch Geometric documentation](https://pytorch-geometric.readthedocs.io/) | docs | free | an afternoon | 11 |
| [Bayesian Data Analysis, 3rd edition (Gelman, Carlin, Stern, Dunson, Vehtari, Rubin)](https://sites.stat.columbia.edu/gelman/book/) | book | free | reference | 11 |
| [Gaussian Processes for Machine Learning (Rasmussen, Williams)](https://gaussianprocess.org/gpml/) | book | free | two chapters, a week | 11 |
| [Deep Learning (Goodfellow, Bengio, Courville)](https://www.deeplearningbook.org/) | book | free | reference | 11 |
| [Graph WaveNet (Wu et al., IJCAI 2019)](https://arxiv.org/abs/1906.00121) | paper | free | an evening or two | 11 |
| [METR-LA and PEMS-BAY datasets (DCRNN repository)](https://github.com/liyaguang/DCRNN) | dataset | free | days | 11 |
| [ETA Prediction with Graph Neural Networks in Google Maps (Derrow-Pinion et al., 2021)](https://arxiv.org/abs/2108.11482) | paper | free | an evening or two | 11 |
