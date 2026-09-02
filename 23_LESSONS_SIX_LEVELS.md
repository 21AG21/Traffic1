# Six Ladders: twelve lessons at six levels

_September 2026. Companion to `21_PLAN_M1_TRAFFIC_MODEL.md` and `22_LEARNING_PATH_ML.md`._

**Twelve lessons on the traffic model, each told six times: for a five-year-old, a middle schooler, a high schooler, a Stanford undergrad, a Stanford grad and a PhD at Apple.**

Every lesson covers one piece of the pipeline, from what a prediction is to running the model unattended on the Mac mini. Pick a level and read the whole course at that height, or read one lesson up the ladder and watch the same idea gain precision. Each lesson ends with a guided walkthrough you can run against the repository, questions with answers, and the mistakes people make.

## Contents

1. [The question: what are we predicting?](#lesson-1-the-question-what-are-we-predicting)
2. [The data: detectors, five-minute rows, incidents](#lesson-2-the-data-detectors-five-minute-rows-incidents)
3. [Cleaning: from a giant text file to a tidy table](#lesson-3-cleaning-from-a-giant-text-file-to-a-tidy-table)
4. [Labels: what does 'slow' mean?](#lesson-4-labels-what-does-slow-mean)
5. [Features: turning the past into columns](#lesson-5-features-turning-the-past-into-columns)
6. [Baselines and honest evaluation](#lesson-6-baselines-and-honest-evaluation)
7. [Logistic regression from scratch: the math the model runs on](#lesson-7-logistic-regression-from-scratch-the-math-the-model-runs-on)
8. [Gradient boosted trees](#lesson-8-gradient-boosted-trees)
9. [Neural networks and backpropagation, on the M1](#lesson-9-neural-networks-and-backpropagation-on-the-m1)
10. [Jams move: sequences, neighbors and the road graph](#lesson-10-jams-move-sequences-neighbors-and-the-road-graph)
11. [From scores to alerts: calibration, thresholds, lead time](#lesson-11-from-scores-to-alerts-calibration-thresholds-lead-time)
12. [Shipping it on one Mac mini: memory, scheduling, monitoring](#lesson-12-shipping-it-on-one-mac-mini-memory-scheduling-monitoring)

## Lesson 1: The question: what are we predicting?

_Lives in the plan's 'What is being predicted'; stages 6, 8 and 10._

**Goal.** Turn 'will traffic be bad?' into a unit, a target, a horizon and a metric, and see why a probability is the right kind of answer.

### Age 5

Roads get slow sometimes, like when everyone leaves the park at the same time and the path gets crowded.

We are building a helper that looks at the road right now and guesses whether it will be slow in a little while, so people can pick a different road before they get stuck.

It is like looking at a dark sky and guessing it will rain soon. Sometimes the guess is wrong, and that is okay. A good helper is right more often than a coin.

**Try this.** Look out the window and guess whether it will rain in one hour. Check later. Were you right? What clue did you use?

### Middle school

Every half mile or so on the freeway there is a sensor in the pavement that counts cars and measures their speed, and it writes a row of numbers every five minutes.

Our program reads the last hour of those rows and answers, for every sensor, one question three times: will this spot be slow 15, 30 and 60 minutes from now?

The answer is not yes or no. It is a number between 0 and 1. A 0.8 means 'very likely slow'; a 0.1 means 'probably fine'. Later we decide how big the number must be before we warn anyone.

To find out whether the program is any good, we wait and see what actually happened. At 4:00 pm sensor 12 shows 65 mph and the program says 0.7 for 4:30. At 4:30 the sensor shows 22 mph. That guess was right.

**Try this.** Write down what 'slow' should mean. Under 30 mph? Slower than it usually is at this time? Both are reasonable, and they are different questions. Keep your answer; lesson 4 comes back to it.

### High school

A prediction problem has four parts. The unit: one station at one 5-minute timestamp. The inputs: everything known at that moment, and nothing after it. The target: a yes-or-no fact about the future, here 'the station is inside a slow episode h minutes later', with h in {15, 30, 60}. The metric: how guesses are scored.

A slow episode is three consecutive 5-minute steps in which speed is below 60 % of what is typical for that station at that time of day. Three steps, so a single noisy reading does not count.

The model outputs a probability rather than a verdict because a 0.55 and a 0.95 carry different information, and because the decision of when to warn depends on how costly a false alarm is compared with a missed jam. That threshold is chosen afterwards, not baked into the model.

One model per horizon, so the 15-minute question and the 60-minute question get their own answers.

**Try this.** Pick a station in your head. Write one row on paper: a few input columns (speed now, speed 5 minutes ago, speed at the next station downstream) and one target column (slow in 30 minutes?). Notice that at the moment you would use this row, the target column is blank. That blank is the whole point.

### Stanford undergrad

Formally this is supervised binary classification on a panel indexed by (station s, time t). The feature vector x_{s,t} is a function of the information available at t, written F_t; the target is y^{(h)}_{s,t} = 1[label(s, t+h) = 1]. We fit f_h(x) ≈ P(y^{(h)} = 1 | x) for each horizon, a direct multi-horizon strategy rather than a recursive one.

The distinction between nowcasting (h = 0) and forecasting (h > 0) is not cosmetic. With `speed_lag_1` in the features and a three-step label, h = 0 is almost solved by copying the last value, which is exactly what the reference plan would have measured.

Calibrated probabilities matter because the alert is a decision. With costs C_FA for a false alarm and C_miss for a missed episode, the Bayes decision rule warns when p > C_FA / (C_FA + C_miss). That is only valid if p is an honest probability.

The metric follows from prevalence: positives are a few percent of rows, so the natural summary is average precision, the area under the precision-recall curve, because the operational question is 'how good is the ranking near the top'.

**Try this.** Derive the threshold rule: write expected cost as a function of the decision for a row with probability p and show that warning is cheaper exactly when p exceeds C_FA / (C_FA + C_miss). Then compute it for a false alarm costing 1 and a miss costing 4.

### Stanford grad

The label is a function of a future path, three consecutive slow steps ending at t + h, not of a single future point, so the targets across horizons and across adjacent timestamps are strongly dependent. The panel is neither independent across t (autocorrelation) nor across s (propagation), so the effective sample size is orders of magnitude below the ten million rows, and every confidence interval must respect that.

The target is defined relative to a profile estimated from training data. Label noise therefore inherits the profile estimator's error, and the label's meaning shifts with the split. Two competing framings avoid part of this: regress the speed ratio with a quantile loss and threshold afterwards, keeping ordinal information; or model episode onset as a hazard and predict time-to-onset.

Direct multi-horizon classifiers ignore coherence: nothing forces the 60-minute probability to relate sensibly to the 30-minute one. Joint sequence outputs or monotone post-processing are the usual fixes.

Prevalence drifts across seasons and roadwork, so a single test window is one draw from a moving distribution. Report rolling-origin folds, not one number.

**Try this.** Show that if the profile were computed on the test period too, the positive rate on test would be pinned near a fixed quantile by construction, and explain why that hides real change.

### PhD at Apple

At the scale of a phone maps product the unit is a road segment with probe data from devices, not a fixed detector, and the problem becomes travel-time and speed prediction over a graph of millions of segments under privacy constraints: on-device aggregation, differential privacy on aggregate speeds, and sparse coverage that varies by segment and hour. The binary 'slow episode' is a product simplification that trades information for interpretability; production systems keep a continuous target and threshold at the edge.

The literature that matters here is graph-based ETA prediction on supersegments (Derrow-Pinion et al., 2021) and the spatio-temporal forecasting benchmarks on PeMS data (DCRNN, STGCN, Graph WaveNet). Open questions worth a thesis chapter: multi-horizon coherence with monotone or joint outputs; non-stationarity handled online versus by periodic retraining; and performative prediction, since drivers who act on an alert change the very label the alert predicted.

The 16 GB Mac mini is a useful proxy for on-device budgets: model size, latency and energy are the constraints that decide what ships, and a model that only runs in a data center is a different product.

**Try this.** Define the label on probe data with heterogeneous sampling rates. State the estimator bias it introduces for low-coverage segments and propose a correction.

### Guided walkthrough

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

### Check yourself

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

### Misconceptions

- 'The model predicts crashes.' It predicts slow episodes; most are propagation of a jam that already exists somewhere downstream, or the daily build-up.
- 'Accuracy is the score.' With 5 % positives, always saying 'not slow' is 95 % accurate. Average precision and recall at a threshold are the scores.
- 'More horizons are free.' Each horizon is a separate, harder problem; skill at 60 minutes is modest and must be shown against persistence.

## Lesson 2: The data: detectors, five-minute rows, incidents

_Lives in stages 1, 2 and 4 · 02_collect_events.py, 09_collect_pems_selenium.py._

**Goal.** Know what a PeMS row and a 511 event actually measure, how much of them there are, and where the numbers lie.

### Age 5

Under the road there are little wires that feel cars driving over them, like a doorbell mat that rings when you step on it.

Every five minutes the wires write down how many cars went by and how fast they were going. Another helper, called 511, writes down when there is a crash and where.

We collect both into two big lists. The lists are the food our helper learns from.

**Try this.** Count the cars that pass your window for one minute and write the number down. Do it again a few minutes later. The numbers change; that is data.

### Middle school

PeMS is California's collection of those road sensors. Each sensor is a 'station'. Every 5 minutes a station makes one row: the time, the station number, how many cars passed (flow), how full the road was (occupancy, the share of time a car was sitting over the sensor), and the average speed.

One day of the Bay Area district is about 1.3 million rows in a text file with no column names. We only care about the roughly 100 stations in Fremont on I-880, I-680 and SR-84.

511.org lists incidents: crashes, stalls, roadwork, each with a time, a road name and a map point. Our collector asks 511 every 15 minutes 'what is happening?' and saves anything near Fremont into a spreadsheet-like file.

The two lists look different: one is a steady drumbeat of numbers, the other is a scattered log of events.

**Try this.** Open `data/fremont_events_log.csv` in a spreadsheet. Sort by `created`. Which hour of the day has the most incidents? Which road?

### High school

Three quantities per station per 5 minutes. Flow: vehicles counted. Occupancy: the fraction of time the loop was covered, which stands in for how dense the traffic is. Speed: miles per hour, sometimes measured directly with a pair of loops and sometimes estimated from flow and occupancy with an assumed vehicle length.

The fundamental diagram links them: as density rises, flow rises until the road reaches capacity, then flow falls as the road jams. That is why occupancy matters and not just speed; a road at 60 mph with heavy occupancy is about to break down.

Timestamps in the file are California local time with no zone attached. We convert them to UTC and handle the two daylight-saving switches, where one hour repeats in November and one is missing in March.

Incidents arrive as JSON with `created` and `updated` times, but our poller only looks every 15 minutes, so a short stall can be missed entirely and every start time is fuzzy by up to 15 minutes.

**Try this.** Sketch the fundamental diagram: flow on the vertical axis, density on the horizontal. Mark free flow, capacity and jam. Where on the curve is 'slow'?

### Stanford undergrad

Detector physics decides the error model. Single-loop stations infer speed from flow and occupancy through a per-station 'g-factor' (effective vehicle length), so speed error grows at low flow and when truck mix changes. PeMS also imputes missing samples from neighboring stations and history and reports the share of real samples as `% Observed`; aggregate columns are flow-weighted across lanes.

Volume arithmetic: about 100 stations × 288 steps × 365 days is 10.5 million rows a year; at 5 float32 columns that is 210 MB in memory, and with 60 feature columns about 2.5 GB. Memory is rows × columns × bytes, and dtype choices decide whether two years fit a 16 GB machine.

Time semantics: `tz_localize('America/Los_Angeles')` must be told what to do with the ambiguous hour (fall back) and the nonexistent hour (spring forward); the plan drops the ambiguous hour rather than guess.

The 511 feed is an event stream sampled every 15 minutes, so incident start and end times are interval-censored. `created` is when CHP logged the report, which lags the physical event by an unknown, skewed delay.

**Try this.** Compute the memory of a year of features as rows × columns × 4 bytes. Then compute the same for float64 and for a naive pandas pipeline that holds three copies during a groupby-shift.

### Stanford grad

Write the measurement model: observed speed = true speed + detector noise + imputation error, where imputation is itself a model of neighbors and history. Imputed rows are therefore correlated with neighboring stations' values, which leaks spatial information into what looks like an observation and flatters any spatial model evaluated on them.

Spatial sampling sets what is predictable. Station spacing of 0.3 to 0.6 miles and backward-forming shock speeds around 10 to 15 mph (kinematic wave theory) mean a jam crosses one station every two to four minutes, so a 30-minute horizon corresponds to roughly two to six stations of upstream propagation.

Temporal aggregation at 5 minutes filters the high-frequency dynamics; the data cannot resolve a 90-second stall. The incident covariate has measurement error of its own: reporting delay, missing geography, unknown severity, and a selection effect, since incidents are reported more readily when congestion is already visible.

**Try this.** Estimate how many stations upstream a queue's tail passes in 30 minutes at a 12 mph shockwave with half-mile spacing, and what that implies for the value of two-hop versus five-hop neighbor features.

### PhD at Apple

Sensor networks in production mix fixed loops, radar and probe data, and the fixed loops come with known pathologies: g-factor drift, chattering detectors, lane-assignment errors. In a district it is normal for a fifth to a third of detectors to be unhealthy at any time, so a serious pipeline carries a detector-health model as a component in its own right, and the imputation mask becomes a feature rather than an embarrassment.

Probe data introduces the privacy layer: on-device aggregation, differential privacy on segment speeds, and coverage that is a function of device share. The statistical problem becomes estimating a segment speed from a biased, sparse sample and propagating that uncertainty.

Here the binding constraint is not modelling but data availability: a 305-second rate limit and a one-day publication lag. Recognizing that a project is data-bound rather than model-bound is most of senior judgment.

Research threads: learned spatio-temporal imputation (graph recurrent imputation networks) against the risk that the downstream model learns the imputer; representing incidents as a marked point process joined to a regular grid; and asynchronous multi-rate sensor fusion.

**Try this.** Design a detector-health score from `% Observed`, flow-occupancy consistency against the station's own fundamental diagram, and agreement with neighbors. Say how you would validate it without ground truth.

### Guided walkthrough

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

### Check yourself

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

### Misconceptions

- 'The file has headers.' It does not; column order must be verified against the documentation on a real file.
- 'Speed is measured.' Often it is estimated from flow and occupancy with an assumed vehicle length.
- 'Incidents are the labels.' They are inputs. The label comes from the detectors.

## Lesson 3: Cleaning: from a giant text file to a tidy table

_Lives in stage 5 · 10_clean_pems.py._

**Goal.** Turn 200 MB of headerless text a day into a small, correct, gap-free table without ever loading the whole file.

### Age 5

The road sensors send a huge messy pile of numbers, like a toy box with everything dumped in at once.

Cleaning means putting each toy in its bin, throwing out the broken ones, and noticing when a piece is missing so we do not pretend it is there.

When it is done, the pile is a neat shelf where we can find anything fast.

**Try this.** Sort a pile of mixed building blocks by color. Count each color. Set aside any broken piece. Which color had the most?

### Middle school

One day's PeMS file is about 200 MB of text. We read it a chunk at a time so it never fills the memory, and we keep only the Fremont stations, which throws away about 97 % of the rows straight away.

Then we convert the times to a common clock (UTC), throw out impossible numbers such as a speed of 250 mph, mark rows that PeMS mostly made up as missing, and fill short gaps of up to 15 minutes by repeating the last value.

We also make sure every station has a row for every 5-minute slot, even if it is blank, so that 'six rows ago' always means 'thirty minutes ago'.

Finally we save the result in Parquet, a format that stores each column separately and loads in a fraction of a second.

**Try this.** In a spreadsheet, write 24 hourly temperatures. Delete two of them. What is a fair way to fill the holes? Now delete ten in a row. Is filling still fair?

### High school

Nine steps. Build the station registry from the metadata file: freeway, direction, postmile, coordinates, lanes, and from the postmile order, each station's upstream and downstream neighbor. Read each gzip in chunks with only the 12 columns you need. Filter to registry station ids inside each chunk. Convert timestamps to UTC with explicit daylight-saving handling. Apply bounds (speed 0 to 100, occupancy 0 to 1, flow up to 3000 per lane). Blank rows with `% Observed` under 50. Forward-fill gaps of at most three steps, within a station only. Reindex every station onto the full 5-minute grid. Write monthly Parquet files in float32.

Why float32: it halves memory and disk, and speed to a hundredth of a mile per hour is more precision than the detector has.

Why the full grid: later, a lag of six rows must mean exactly thirty minutes. If a row is missing, positional shifts silently become the wrong time.

**Try this.** Compute the bytes for 10 million rows × 5 columns at float64 and at float32. Then look up how much memory your machine has and see how many years of each would fit.

### Stanford undergrad

The principles are those of any data pipeline: idempotence (re-running yields identical output), streaming over in-memory, columnar storage with predicate pushdown, dtype economy, and validation at the boundary (assert the schema, the value ranges, and 288 rows per station-day).

Daylight saving is the classic correctness trap. `America/Los_Angeles` has a nonexistent hour in March and a repeated hour in November. `ambiguous='NaT'` drops the repeated hour rather than mislabelling half of it; `nonexistent='shift_forward'` handles the spring gap deterministically.

Forward fill must be grouped: `df.groupby('station_id').ffill(limit=3)` so that one station's last value never bleeds into the next station's first missing row. The reindex to the full grid must happen before any positional operation.

Testing: a fixture of a hundred rows that includes a DST boundary, a station boundary and a `% Observed` failure. If the fixture passes, the whole year is likely right; if it fails, you saved a day.

**Try this.** Write a pytest that constructs two stations with a gap at the boundary and asserts that forward fill does not cross it. Make it fail first by removing the groupby.

### Stanford grad

Imputation is inference. A zero-order hold assumes the process is locally constant; linear interpolation uses the next observed value and therefore looks ahead, which is inadmissible for features at serving time but harmless for cleaning history if labels never use imputed rows. Decide per column which rule applies and write it down.

Missingness mechanism matters: a power outage is missing completely at random; a loop that fails under heavy, stop-and-go traffic is missing not at random, correlated with exactly the state you want to predict. Dropping such rows biases the positive rate downward; filling them biases speed upward.

The registry's neighbor chain assumes a line. Interchanges (I-880 with SR-84, I-680 with Mission) break it; those need explicit edges or the graph model inherits a wrong prior.

**Try this.** Classify each failure cause (power outage, loop failure under congestion, PeMS server outage, roadwork removing a station) by missingness mechanism and state the direction of bias each induces in label prevalence.

### PhD at Apple

At scale this stage is a streaming system with schema evolution, late and corrected data, backfills and lineage. The design that survives is layered: raw is immutable and content-hashed (the manifest in this plan), cleaned is a pure function of raw plus code version, features are a pure function of cleaned. Any month can then be rebuilt bit for bit, which is what makes an evaluation trustworthy a year later.

Detector health deserves a model of its own, with the imputation mask exposed downstream as a feature. On unified-memory hardware the cleaning stage is memory-bandwidth-bound long before it is compute-bound; profile with real files rather than reasoning about big-O.

Research threads: learned spatio-temporal imputation versus the downstream model learning the imputer; treating the observation mask as data; and reproducible backfills when the upstream provider re-publishes a corrected day.

**Try this.** Design the backfill procedure for a corrected PeMS day: what is recomputed, what is invalidated (features, models, logged metrics), and how the manifest records it.

### Guided walkthrough

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

### Check yourself

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

### Misconceptions

- 'Parquet is just a faster CSV.' It is columnar and typed; the dtypes you write are the dtypes you get back, and only the columns you ask for are read.
- 'Filling gaps is harmless.' Every fill is a model; a fill that looks ahead leaks the future into features.
- 'Cleaning is a one-off.' It is the stage that runs on every new day forever, and the one whose silent errors are costliest.

## Lesson 4: Labels: what does 'slow' mean?

_Lives in stage 6 · 11_build_labels.py._

**Goal.** Define the target precisely, compute it without leaking, and understand that the definition decides what the model can mean.

### Age 5

How do we know a road is slow? We compare it with how it usually is.

If cars usually zoom at 65 and today they crawl at 30, that is slow. If they always crawl at five o'clock, that is just five o'clock.

So the helper first learns what 'usual' looks like for every road and every time of day, and then notices when today is different.

**Try this.** Is it cold today? Compared with what: yesterday, summer, or winter? Notice that the answer changes with what you compare to.

### Middle school

For every station and every 5-minute slot of the day, separately for weekdays and weekends, we find the typical speed by taking the median of all the days we have. The median is the middle value when you line the numbers up, so a couple of crazy days do not drag it around the way an average would.

Then 'slow' means today's speed is below 60 % of that typical speed. A slow episode is three slots in a row that are slow, fifteen minutes, so one odd reading does not count.

That definition has a surprise inside it: the evening rush hour, which is slow every day, is not 'slow' by this rule, because slow is normal then. The rule finds unusual slowness.

**Try this.** Line up these speeds: 62, 64, 65, 12, 66, 63, 65. Find the median and the mean. Which one is a better 'typical'? Why?

### High school

Three formulas. profile(s, slot, daytype) = median of speed over training days. ratio = speed / profile. is_slow = ratio < 0.6, and label = 1 when is_slow holds for three consecutive steps, the window ending at the row's own time.

Two design choices with consequences. Relative to profile: the label flags unusual slowness, so recurrent congestion is invisible to it. Absolute (speed below 35 mph for three steps): the label flags congestion itself, rush hour included. The plan computes both and makes you pick.

Details that matter: 5-minute slots instead of hours, because a 4 pm median mixes free flow at 4:00 with a jam at 4:55; and free-flow speed is better measured as the station's 85th percentile than as a median, which is why the plan adds `ratio_ff` too.

**Try this.** Make up one day of speeds for a station with a 5 pm rush hour. Compute both labels by hand. Where do they disagree, and which one would a driver want?

### Stanford undergrad

The label is a deterministic function of the series and an estimated profile, so the profile is part of the model. Estimating it on all data lets the test period influence training labels: target leakage through the label definition. The profile must be fit on training timestamps only and stored as an artifact.

Quantile estimation: the median has a 50 % breakdown point and, for n days per cell, a standard error near 1.25 σ/√n; with eight weeks of weekdays that is n ≈ 40, so smoothing across neighboring slots trades a little bias for a useful variance reduction, and cells below 20 samples fall back to the hourly median.

Consecutive-run labels: label_t = Π_{k=0..2} is_slow_{t−k}. For horizon h = 3 steps, the label at t + 3 depends on speeds at t + 1, t + 2, t + 3 and does not touch the features at t. Prevalence is a steep function of the 0.6 cutoff and the run length; sweep both once and freeze them in the filename.

**Try this.** Show that for a horizon of two steps the label window at t + 2 would include the row at t itself, overlapping the feature `speed_lag_0`, and explain what that does to a naive evaluation.

### Stanford grad

Label design is a modelling decision. The relative label approximates conditional anomaly detection; the absolute label approximates a level target. They induce different class balance, different predictability (recurrent congestion is calendar-predictable, non-recurrent is not), and different product semantics, and a model good at one can be useless at the other.

Alternatives keep more information: regress the speed ratio with a pinball loss and derive probabilities from quantiles; or model episode onset as a point process with covariates and predict hazard. Both avoid the brittle 3-step run and make horizon coherence natural.

Errors-in-variables: detector noise near the 0.6 boundary is heteroscedastic (single-loop speed error grows at low flow), so label noise is concentrated exactly where the decision is made. And because the profile pins the training positive rate near a quantile by construction, real seasonal shifts in congestion are masked in the training window.

**Try this.** Derive how a 5 % multiplicative bias in speed estimation shifts the positive rate near the threshold, given a local density of the ratio around 0.6, and propose a check using stations with double loops.

### PhD at Apple

Production traffic systems keep the modelled quantity continuous (speed, travel time) and push any binary definition into a presentation policy. One model then serves several products, and changing what 'slow' means does not trigger retraining or invalidate history.

Research angles: learning when the target itself depends on estimated quantities; semi-supervised regime detection on the fundamental diagram (hidden Markov or change-point models that find breakdown without a hand threshold); and measuring label quality through agreement across definitions and across detector types.

Operationally, version the label definition as you would a schema. A label change invalidates every logged metric and every comparison across model generations.

**Try this.** Propose an A/B design that compares two label definitions on a downstream user metric such as reroute acceptance, and state what confounds the comparison.

### Guided walkthrough

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

### Check yourself

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

### Misconceptions

- 'The median profile is the free-flow speed.' It is the typical speed, which at rush hour is far below free flow. Free flow is the 85th percentile.
- 'Threshold 0.6 is a fact.' It is a choice that sets prevalence and difficulty; version it.
- 'A label is ground truth.' It is a derived quantity with its own estimation error, concentrated at the boundary.

## Lesson 5: Features: turning the past into columns

_Lives in stage 8 · 13_feature_engineering.py._

**Goal.** Build about sixty columns that a model can use, every one of them known at prediction time, and the targets shifted into the future.

### Age 5

To guess what the road will do, we look at clues: how fast it was a minute ago, how fast the road just ahead is, whether there was a crash, whether it is a school day, whether it is raining.

Each clue is a number we write in a row. Lots of rows make a table. The helper learns which clues matter by reading the table.

**Try this.** What clues tell you it is almost dinner time? A smell, the clock, your tummy. Which clue is the most reliable?

### Middle school

A feature is one column of clues. The row for a station at a time has about sixty of them: the speed now and 5, 10, 15, 30, 60 minutes ago (lags); the average of the last 30 minutes (a rolling mean); the speed at the next station downstream and upstream; whether an incident is active nearby and how far away; the time of day, the day of the week, holidays, and rain.

The one rule: every clue must be something you could actually know at that moment. Speed ten minutes from now is not a clue, it is the answer.

The target columns are the answers, copied from the future: the label 15, 30 and 60 minutes later.

**Try this.** Which of these is a legal clue at 4:00 pm for predicting 4:30? Speed at 3:55. Speed at 4:10. Tomorrow's weather. Today's date. Yesterday's speed at 4:30. Say why for each.

### High school

Lags come from a shift within each station on the full grid: `df.groupby('station_id')['speed'].shift(k)`. Rolling statistics from `.rolling(6).mean()` where 6 steps is 30 minutes. The slope is `speed_lag_0 − speed_lag_3`.

Time is a circle: 23:55 is next to 00:00, but the number 1435 is far from 0. So minute-of-day becomes two columns, sin(2π m / 1440) and cos(2π m / 1440), which put midnight next to itself. Weekday gets the same treatment.

Neighbors come from the registry: look up the upstream and downstream station ids and merge their speed at the same timestamp. Incidents come from lesson 7's join. Targets are `label.shift(−h)` per station, where a negative shift looks forward.

Before training, one assert: no feature column may use any row after t. If you cannot say how a column is computed from the past alone, delete it.

**Try this.** Draw the sine and cosine of the hour on a circle for hours 0, 6, 12, 18, 23. Why do you need both columns rather than just one?

### Stanford undergrad

Every feature must be F_t-measurable, a function of information available at t. That single principle rules out centered rolling windows, target encodings computed on the same rows, and profiles fit on the future.

Mechanics: `groupby` before every shift and rolling to prevent cross-station contamination; `min_periods` to control early-window NaNs; the full grid so a positional shift is a time shift. Trees consume NaN natively and learn a default direction; linear models and networks need imputation plus a missing indicator, and standardization with training-set statistics only.

Categorical station identity: native categorical for LightGBM, one-hot or embedding elsewhere. Interactions (rain × rush hour) can be hand-built for linear models; trees discover them.

Expect the correlation between `speed_lag_0` and the target to fall with the horizon. That decay is the shape of the problem: momentum dominates at 15 minutes, neighbors and calendar at 60.

**Try this.** Compute the Pearson correlation of `speed_lag_0` with `y_15`, `y_30`, `y_60` on one month and plot it. Then do the same for `down1_speed`. Which decays slower, and why?

### Stanford grad

Feature sets carry bias-variance trade-offs that depend on the learner: a dozen correlated lags inflate variance for a linear model through multicollinearity and cost nothing for trees. Rolling summaries plus a few lags are the robust compromise.

Neighbor features encode a hand-specified graph prior (one or two hops along the postmile line); the spatio-temporal network learns the graph instead. Both are legitimate; the hand prior is cheaper to monitor.

Leakage is subtle: a centered rolling mean uses t + 1 through t + k; a target encoding of `station_id` must be computed out-of-fold in time; weather joined on the realized hour is fine for training but the serving loop only has the forecast, so training should use the forecast issued at t too, or the model will over-trust weather.

Lag features are stationary by construction; calendar and static features drift with roadwork and detector replacement.

**Try this.** Construct a toy series, compute a centered 30-minute rolling mean, use it as a feature for a 15-minute target, and quantify the inflation in average precision compared with the trailing mean.

### PhD at Apple

The industry problem this stage solves is point-in-time correctness: computing training features exactly as they would have been available at serving time. Feature stores with time-travel joins exist for this; the plan's answer at small scale is one feature module used by both paths and a golden-file test that compares them bit for bit.

Representation learning replaces hand features at scale (learned embeddings for segments, time and weather), yet hand features remain the strongest baseline for tabular pipelines and are far cheaper to monitor, explain and debug. Choosing which to ship is an engineering decision about maintenance, not only accuracy.

Research threads: causal feature selection under shift (incident features and congestion share a latent cause), invariance-based training, and the marginal value of exogenous streams such as roadwork feeds, event calendars and school schedules.

**Try this.** Specify the point-in-time join for the weather forecast: the feature at t is the forecast for hour t + 1 issued at or before t, not the realized weather. Say how you would store forecast vintages to make training match serving.

### Guided walkthrough

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

### Check yourself

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

### Misconceptions

- 'More features always help.' Correlated features hurt linear models and slow trees; every feature is another thing to monitor for drift.
- 'Realized weather is a fine training feature.' At serving time only the forecast exists. Train on what you will have.
- 'Rolling windows are safe.' Only trailing windows are; centered windows look ahead.

## Lesson 6: Baselines and honest evaluation

_Lives in stages 9 and 11 · 14_train_baselines.py, 18_evaluate_lead_time.py._

**Goal.** Score a forecast so that the number means something: against a baseline, on later data, with metrics that imbalance cannot fool.

### Age 5

Before we say our helper is smart, we compare it with a very simple guess: 'the road will be like it is right now'. If the helper cannot beat that, it is not helping.

We also test it on days it has never seen, like a real quiz, not on the days it practiced on.

**Try this.** For one week, guess tomorrow's weather as 'same as today'. Keep score. How often is the lazy guess right? That is the score a smart guesser has to beat.

### Middle school

Two simple guesses, called baselines. Persistence: whatever the road is doing now, it keeps doing. Profile rate: how often this station is slow at this time of day, on average.

Two measurements. Of all the times we shouted 'slow!', how often was it really slow? That is precision. Of all the real slow times, how many did we catch? That is recall.

Because slow times are rare, maybe 5 % of rows, a helper that always says 'not slow' is 95 % 'accurate' and completely useless. That is why accuracy is not the score.

And we test on later weeks than we trained on, never on the same days, because two rows five minutes apart look almost the same and would make the quiz too easy.

**Try this.** There are 100 rows and 5 are slow. A helper says 'slow' 10 times, and 4 of those are right. Compute precision and recall. Now compute accuracy for a helper that never says 'slow'.

### High school

The confusion matrix gives true and false positives and negatives. Precision = TP / (TP + FP); recall = TP / (TP + FN); F1 is their harmonic mean. Sliding the threshold traces the precision-recall curve; its area is average precision (AP), the headline metric here. ROC-AUC exists too but looks flattering under imbalance because true negatives are cheap.

For probabilities: log loss and the Brier score punish confident wrong answers. Calibration asks whether, among rows predicted at about 0.7, about 70 % turn out positive; a reliability diagram shows it.

The split is chronological: train on months 1 to 8, validate on 9 to 10, test on 11 to 12, with a one-day gap so lags and horizons never straddle a boundary. Skill is reported relative to persistence at each horizon.

**Try this.** Rank ten predictions by probability with three positives among them, and compute average precision by hand as the mean of precision at each positive's rank.

### Stanford undergrad

Shuffled K-fold is wrong for a time series because autocorrelation places near-copies of every test row in the training fold, and because the estimand is future performance, which only forward validation approximates. Use blocked or rolling-origin validation with a gap of at least the longest lag plus the horizon.

AP equals the mean over positives of the precision at their rank, and its no-skill value equals the prevalence. Log loss and Brier are proper scoring rules: they are minimized in expectation only by the true probabilities, which is what makes them honest. The Brier score decomposes into calibration plus refinement, and the reliability diagram with expected calibration error shows the first term.

Uncertainty: rows are dependent, so bootstrap over blocks (days) rather than rows to get intervals for AP.

**Try this.** Implement rolling-origin evaluation with three folds on one station's data and compare the spread of AP across folds with the spread from shuffled 3-fold. Explain the difference.

### Stanford grad

Evaluation is the main scientific control in this project. Effective sample size under spatio-temporal dependence is small; threshold and label sweeps create multiple comparisons; and per-row metrics hide what a user experiences, which is per-episode: lead time from first alert to onset, share of episodes caught, and false alarms per station-day.

Forecasting practice reports skill scores against persistence and climatology (the profile rate); the Brier skill score can be negative while AP looks respectable, and both should be shown. Calibration drifts between validation and test months, so recalibration cadence is itself a design parameter.

A single test window is one draw from a non-stationary process. Rolling folds give a distribution of scores; report it.

**Try this.** Derive the Brier skill score against the profile-rate climatology and construct an example where AP improves over the baseline while the skill score is negative.

### PhD at Apple

Offline metrics are proxies. The product metric is what drivers do (reroutes, realized travel time), and its correlation with AP is unknown until measured; A/B tests in traffic suffer interference because routing changes the traffic everyone else sees, a textbook case of performative prediction.

Evaluation infrastructure at scale: metric definitions versioned with the label, backtests reproducible from logged features, slicing by segment, time and weather to surface failure modes, and guardrails such as never regressing worst-slice recall.

Research threads: evaluating forecasts under intervention, distribution-free control of alert rates (risk-controlling prediction sets), and calibration under covariate and label shift using unlabeled target data.

**Try this.** Design an interleaving or switchback experiment that measures the usefulness of alerts without exposing every driver to a possibly bad model, and state the interference assumptions it needs.

### Guided walkthrough

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

### Check yourself

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

### Misconceptions

- 'Cross-validation is always better than a single split.' Shuffled CV on a time series leaks and reports fantasy scores; only forward splits are valid here.
- 'Beating the baseline once is enough.' Report it per horizon and per fold; a model can beat persistence at 60 minutes and lose at 15.
- 'Calibration is a detail.' It is the difference between a probability and a score, and the threshold rule assumes a probability.

## Lesson 7: Logistic regression from scratch: the math the model runs on

_Lives in stage 3 and stage 9 · 03_train_from_scratch.py, 15_train_logreg.py, 00_CONCEPTS.md._

**Goal.** Derive, implement and check the training loop that every later model generalizes: weighted sum, sigmoid, cross-entropy, gradient step.

### Age 5

Imagine a seesaw. Each clue pushes it one way ('slow!') or the other ('fine'). Some clues are heavy, like a crash just ahead, and some are light, like a Tuesday.

Learning means finding out how heavy each clue should be. We look at lots of old days, and every time the seesaw tips the wrong way we make the wrong clues a bit lighter and the right clues a bit heavier.

**Try this.** Guess a friend's favorite color. When you are wrong, change your guess a little toward what they said, not all the way. Keep going until you are right most of the time.

### Middle school

Give each clue a weight, a number. Add up clue × weight for every clue, plus one extra number called the bias. That total is the score.

Squash the score into 0 to 1 with an S-shaped curve called the sigmoid, so it reads as a probability: a big positive score becomes nearly 1, a big negative score nearly 0, and zero becomes 0.5.

To learn the weights, take an old row, compute the probability, compare with what really happened (1 or 0), and nudge each weight a small step in the direction that would have made the guess better. Do that thousands of times and the weights settle.

**Try this.** One clue: 'speed dropped in the last 15 minutes' is 1 or 0. Weight 2, bias −1. Compute the score and the sigmoid for both clue values. Which one says 'probably slow'?

### High school

Score z = w·x + b. Probability p = σ(z) = 1 / (1 + e^(−z)). Loss for one row, the cross-entropy: −[y ln p + (1 − y) ln(1 − p)], which is small when confident and right and huge when confident and wrong.

The gradient is surprisingly clean: ∂loss/∂w = (p − y)·x and ∂loss/∂b = (p − y). The sigmoid's derivative is p(1 − p), and the chain rule cancels it against the log's derivative, leaving error times input.

The update is w ← w − η (p − y) x with a small learning rate η. That line is the whole of `03_train_from_scratch.py`; everything in lesson 9 is this with more layers.

**Try this.** Take x = (1, 0.5), y = 1, w = (0, 0), b = 0, η = 0.1. Compute p, the gradient, and the new w and b. Then compute p again with the new weights and confirm it moved toward 1.

### Stanford undergrad

Logistic regression is maximum likelihood for a Bernoulli outcome with a logit-linear mean. The negative log-likelihood is convex in (w, b) (its Hessian X^T diag(p(1 − p)) X is positive semidefinite), so gradient descent with a suitable step reaches the global optimum; L2 regularization is a Gaussian prior and gives the MAP estimate.

Standardizing features conditions the problem; the safe learning rate scales inversely with the largest eigenvalue of the Hessian. Mini-batch SGD trades noise for speed; Adam is unnecessary for a convex problem but harmless.

Without regularization the first-order conditions force the mean predicted probability to equal the observed rate, so the model is calibrated in-sample given an intercept. The decision boundary is a hyperplane; interactions must be built by hand.

**Try this.** Derive the gradient from the negative log-likelihood using the chain rule through σ, and show from the stationarity condition on the intercept that the mean residual is zero at the optimum.

### Stanford grad

As a generalized linear model, logistic regression uses the canonical link of the Bernoulli family, which is why the gradient has the score-residual form and why iteratively reweighted least squares (Fisher scoring) converges in a few steps. The MLE is asymptotically normal, giving Wald and likelihood-ratio tests, and quasi-complete separation sends coefficients to infinity unless regularized.

Class reweighting is a prior shift: multiplying positive weights by r shifts the learned intercept by about ln r and destroys calibration; undo it with a log-odds offset before reporting probabilities.

Logistic regression is a one-layer network with sigmoid output; softmax is its multiclass form; the maximum-entropy view derives the same model from moment constraints. Its limits are representational, not statistical.

**Try this.** Show that reweighting positives by a factor r changes the asymptotic log-odds by ln r, and derive the correction that restores calibration to the original prevalence.

### PhD at Apple

At scale the same model appears as sparse high-dimensional logistic regression in click-through prediction, trained with distributed SGD or L-BFGS, feature hashing, and per-coordinate learning rates; it persists in production for latency, interpretability and calibration properties that deeper models lack.

The loss lives on in every network as binary cross-entropy with logits, computed through log-sum-exp for numerical stability. Recent theory clarifies what gradient descent does on this loss: on separable data it converges in direction to the maximum-margin classifier (Soudry et al., 2018), which is why early stopping acts as regularization.

Open questions of practical weight: the interaction between class reweighting, calibration and decision thresholds under label shift, and generalization for linear classifiers in the interpolating regime.

**Try this.** State and sketch the proof that gradient descent on logistic loss over linearly separable data converges in direction to the max-margin solution, and explain what it implies for the choice of stopping time.

### Guided walkthrough

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

### Check yourself

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

### Misconceptions

- 'A bigger learning rate learns faster.' Past a limit set by the curvature it diverges; the loss goes up, not down.
- 'The weights say which feature matters most.' Only after standardization, and only for the linear part; correlated features share weight arbitrarily.
- 'Logistic regression is obsolete.' It is the calibration-friendly baseline every other model must beat, and it is inside every neural network's output layer.

## Lesson 8: Gradient boosted trees

_Lives in stage 9 · 16_train_gbt.py, LightGBM._

**Goal.** Understand why hundreds of small trees, each fixing the last one's mistakes, are the production model for tabular data, and how to tune them without fooling yourself.

### Age 5

A decision tree is a game of twenty questions. Is the road ahead slower than 40? Is it raining? Is it five o'clock? Each answer takes you down a branch to a guess.

Boosting is playing many tiny question-games in a row. Each new game only tries to fix what the games before it got wrong, and at the end all the little answers are added up.

**Try this.** Play twenty questions to guess an animal. Notice that the best questions split the possibilities in half. A bad question ('is it a zebra?') wastes a turn.

### Middle school

One tree splits the rows by asking questions about the clues and ends in leaves, each with a score. One tree is rough and jumpy.

Boosting builds hundreds of small trees. Each is trained on the mistakes left over by the ones before, and its score is shrunk by a 'learning rate' before being added, so no single tree can dominate. The sum of all the shrunken trees is the model.

LightGBM is a fast program that does this. It bins each clue into 255 buckets so it can try splits quickly, and it can stop adding trees when the score on held-out data stops improving.

**Try this.** Write ten rows with a speed and whether the road was slow 30 minutes later. Try the split 'speed < 40' and count slow and not-slow on each side. Try 'speed < 55'. Which split separates them better?

### High school

A tree chooses the split that most reduces the mix of classes in its leaves (the gain). Depth and the number of leaves control how detailed it can be. Boosting: start with a constant F₀, then repeatedly fit a small tree h_m to the residuals (for logistic loss, y − p) and set F_m = F_{m−1} + η h_m.

The knobs that matter: `num_leaves` (63 here), `learning_rate` (0.05), `min_data_in_leaf` (200, so a leaf cannot fit noise), `feature_fraction` and `bagging_fraction` (randomness that reduces overfitting), and early stopping on validation average precision. `is_unbalance` reweights the rare positives. `station_id` is passed as a categorical, not one-hot.

Thousands of trees at a small learning rate usually beat hundreds at a large one, and early stopping picks the count for you.

**Try this.** Explain in your own words why 3000 trees at η = 0.05 can beat 300 trees at η = 0.5, and what early stopping is protecting against.

### Stanford undergrad

Gradient boosting is gradient descent in function space (Friedman, 2001). Modern libraries use the second-order expansion of the loss: with per-row gradient g and Hessian h, a leaf's optimal value is −Σg / (Σh + λ) and a split's gain is the increase in Σ(Σg)² / (Σh + λ) across the two children. For logistic loss g = p − y and h = p(1 − p).

Histogram-based split finding bins features to 255 values (one byte per cell), which sets memory at roughly rows × features bytes; LightGBM grows leaf-wise (best leaf first) rather than level-wise, and adds gradient-based one-side sampling and exclusive feature bundling for speed.

Regularization is λ, minimum child weight, shrinkage and subsampling. Importance by gain is biased toward high-cardinality features; TreeSHAP gives exact per-row attributions.

**Try this.** Derive the optimal leaf weight and the split gain from the second-order Taylor expansion of a general loss, then specialize to logistic loss.

### Stanford grad

Boosted trees win on tabular data for identifiable reasons (Grinsztajn et al., 2022): they are invariant to feature scaling and rotation-non-invariant in a way that suits heterogeneous columns, and they are robust to uninformative features. Their failure modes are just as identifiable: piecewise-constant extrapolation outside the training range, sensitivity of split thresholds to distribution shift, and poorly calibrated raw scores, hence the isotonic step.

Monotone constraints encode physics: the probability of a slow episode should not increase with downstream speed. Class weighting, threshold moving and focal loss are three ways to handle imbalance with different effects on calibration.

Hyperparameter search must use the chronological validation split, and choosing among many configurations on that split is itself a fit; nested rolling-origin evaluation is the honest cost.

**Try this.** Show that for a well-specified model class weighting is equivalent to shifting the decision threshold, then explain concretely why the equivalence fails for finite trees with early stopping.

### PhD at Apple

In production, boosted ensembles are compiled to branch-free code or shipped on device through Core ML tree ensembles, quantized, and watched for drift through the histogram of values arriving at each split. Retraining at scale uses warm starts and careful control of tree count so that latency budgets hold.

Research threads: margin-based theory of boosting; hybrids with networks (differentiable trees, GBT-initialized MLPs, deep forests); uncertainty for ensembles (natural-gradient boosting, quantile boosting, conformal on top); and the persistent empirical finding that on ten-million-row engineered tabular data a boosted ensemble is the strongest and cheapest baseline.

The senior judgment is choosing when not to replace it: a model that is 0.01 AP better and ten times harder to monitor is a worse product.

**Try this.** Propose a set of monotone constraints for this problem, argue which are physically justified from kinematic wave theory and which are merely plausible, and describe how you would test that a constraint does not cost skill.

### Guided walkthrough

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

### Check yourself

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

### Misconceptions

- 'More trees always help.' Past the validation peak they hurt, which is what early stopping finds.
- 'Feature importance shows causation.' It shows what the trees used; a proxy for the cause ranks as high as the cause.
- 'Tree scores are probabilities.' They are ranked well and calibrated badly; isotonic regression fixes that in lesson 11.

## Lesson 9: Neural networks and backpropagation, on the M1

_Lives in stage 9 · 17_train_mlp.py, 17b, PyTorch on MPS._

**Goal.** Build a network by hand once, check its gradients, then let PyTorch do it on the Mac's GPU and know what changed.

### Age 5

A neural network is a lot of tiny seesaws stacked in layers. The first layer looks at the clues, the next layer looks at what the first layer said, and so on until the last one gives the answer.

When the answer is wrong, the 'oops' travels backwards through the layers so every seesaw adjusts a little. That is called backpropagation.

**Try this.** Play the whisper game down a line of friends. When the message comes out wrong, work backwards to find where it changed. That is what the network does with its mistake.

### Middle school

One layer computes weighted sums of its inputs, exactly like lesson 7, and then applies a simple bend: negatives become zero (ReLU). Stacking layers lets the network learn combinations of clues, such as 'raining AND rush hour', which a single seesaw cannot.

Training is the same loop as before: guess, compare, adjust. The adjusting uses the chain rule to pass the error back through every layer. PyTorch does that calculus for you.

The Mac mini has a small GPU. PyTorch can run the arithmetic there through a backend called MPS. For tiny networks the CPU is nearly as fast, because telling the GPU what to do costs more than the work.

**Try this.** Inputs a = 1, b = 2, weights 0.5 and −1, bias 0.2. Compute the weighted sum, then apply ReLU. Now flip the sign of the first weight and do it again.

### High school

Forward pass: h = ReLU(W₁x + b₁), p = σ(w₂·h + b₂). Loss: cross-entropy. Backward pass: ∂L/∂w₂ = (p − y) h; ∂L/∂h = (p − y) w₂; through ReLU multiply by 1 where the pre-activation was positive and 0 elsewhere; ∂L/∂W₁ = δ₁ xᵀ.

Optimizers: plain gradient descent, momentum, Adam (adaptive step per weight). Overfitting controls: early stopping, dropout, weight decay. Practical rules: standardize inputs, start with a learning rate of 1e-3, batches of a few hundred, and make sure the loss falls on a tiny subset before training on everything.

On the M1: `device = torch.device('mps')`, move the model and each batch with `.to(device)`, float32 only.

**Try this.** Implement the one-hidden-layer forward and backward pass in numpy and check every gradient with finite differences at ε = 1e-5. Do not move on until the largest difference is below 1e-6.

### Stanford undergrad

Backpropagation is reverse-mode automatic differentiation on a computational graph: one forward and one backward pass give every parameter's gradient at a cost independent of parameter count. Initialization (He for ReLU, Xavier for tanh) keeps activations and gradients at unit scale; without it deep nets vanish or explode.

Batch and layer normalization stabilize training; Adam's bias correction matters in the first steps; warmup and cosine schedules are cheap wins. Early stopping on validation is the main regularizer for small tabular nets and is equivalent to L2 for linear models.

Engineering: keep the dataset on the device, batch in large chunks, avoid per-batch Python work. MPS specifics: fp32 and fp16, no fp64, `PYTORCH_ENABLE_MPS_FALLBACK=1` for unsupported ops, and a launch-overhead floor that makes tiny models CPU-competitive. Measure rows per second at batch 256 and 8192 on both devices before choosing.

**Try this.** Benchmark the MLP on CPU and MPS at batch sizes 256, 2048 and 8192; find the crossover and explain it in terms of kernel launch overhead versus arithmetic.

### Stanford grad

Overparameterized networks generalize for reasons still being mapped: the implicit bias of SGD, double descent, and flat minima are the working vocabulary. On tabular data of this size they rarely beat boosted trees (Grinsztajn et al., 2022) unless the data has structure the trees cannot see, which here is the spatial graph.

Sequence models: GRU and LSTM gating, truncated backpropagation through time, dilated temporal convolutions and attention. For traffic, DCRNN couples diffusion convolution on the road graph with a sequence-to-sequence GRU; STGCN alternates gated temporal convolutions with spectral graph convolutions; Graph WaveNet learns an adaptive adjacency. The plan's challenger, a shared GRU plus two rounds of neighbor mixing, is a stripped-down DCRNN.

Training recipe for the 12 × N × 6 windows: whole dataset in memory, batches of timestamps, positive class weight per horizon, early stopping on validation average precision.

**Try this.** Write the message-passing update h ← h + W_u h_up + W_d h_down as a matrix expression with directed adjacency matrices A_up and A_down, and relate two rounds of it to a K = 2 diffusion convolution.

### PhD at Apple

On-device deep learning brings constraints that training clusters hide: Core ML export, fp16 on the Neural Engine, static shapes, a supported-operator list, energy per inference, and thermal throttling. The unified-memory model on Apple silicon removes copies between CPU and GPU but makes many workloads bandwidth-bound.

What transfers from the Mac mini to a cluster: recipes, evaluation discipline, and the habit of checking gradients; what does not: batch-size and learning-rate scaling laws.

Research threads: spatio-temporal transformers and adaptive graphs at city scale; physics-informed losses that respect kinematic wave direction; time-series foundation models used zero-shot against small specialized models; and uncertainty through deep ensembles, dropout at inference or conformal wrappers.

**Try this.** Outline the Core ML export path for the spatio-temporal net: which layers convert cleanly, how to handle a variable number of stations, and what you would measure on device before shipping.

### Guided walkthrough

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

### Check yourself

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

### Misconceptions

- 'Deep learning needs a data center.' A 0.1 M-parameter network trains in an hour on a Mac mini; what needs a data center is a different problem.
- 'A network will beat the trees.' On this data, usually not; the graph structure is its one real advantage.
- 'Training loss going down means it works.' Validation AP is the only number that means it works.

## Lesson 10: Jams move: sequences, neighbors and the road graph

_Lives in stages 5, 8 and 9 · the registry, neighbor features, 17b._

**Goal.** See why a slowdown downstream predicts your station's future, and how models that read the road as a sequence and a graph exploit that.

### Age 5

When the front car in a line stops, the stopping spreads backwards, car by car, like a wave running down a slinky.

So if the road just ahead of you got slow, you will be slow soon. Our helper watches the road ahead to see the wave coming.

**Try this.** Line up toy cars and stop the first one. Watch how the 'stop' moves backwards along the line. Which car stops last?

### Middle school

Jams start at a bottleneck, a crash or a busy merge, and grow backwards against the direction of travel at roughly 10 to 15 mph. A station two miles upstream of a jam will be inside it in about ten minutes.

That is why the neighbor clues, especially the downstream station's speed, predict so well at 15 and 30 minutes, and why a model that looks at all the stations together, a 'graph' model, can learn the wave itself instead of being told about it.

**Try this.** A jam's tail moves backwards at 12 mph. How long until it reaches a station 3 miles upstream? What about 6 miles? Which horizon does each match?

### High school

Time series vocabulary: autocorrelation (speed now versus five minutes ago correlates at about 0.9), seasonality (daily and weekly), lags as features; persistence is strong precisely because autocorrelation is strong.

Space: the road is a line graph, and the registry's postmiles give each station an upstream and a downstream neighbor. A sequence model (GRU) reads the last 12 steps in order and keeps a memory; graph mixing lets each station's memory hear its neighbors' memories.

The physics: between two traffic states the boundary moves at w = Δflow / Δdensity. When a road goes from flowing to jammed, that slope is negative, a backward-moving wave, and its size sets how far ahead a station's future can be read from stations downstream.

**Try this.** Plot the autocorrelation of one station's speed for lags 1 to 48 (four hours). Describe the shape and say what it implies for the lag features.

### Stanford undergrad

Kinematic wave theory (Lighthill-Whitham-Richards): conservation ∂k/∂t + ∂q/∂x = 0 with a fundamental diagram q = Q(k). A discontinuity between states 1 and 2 moves at w = (q₂ − q₁)/(k₂ − k₁); in the congested branch w is about −10 to −15 mph, and queues discharge at capacity when the bottleneck clears.

Implication for prediction: the horizon over which a station's future is readable from its neighbors is roughly the distance to the nearest downstream disturbance divided by |w|. Beyond that, the calendar and incidents carry the signal.

Models along the ladder: ARIMA on one series, vector autoregression across stations, a GRU as a nonlinear state-space model, graph convolution as spatial smoothing with learned weights, and direct multi-horizon heads.

**Try this.** From one station's scatter of flow against occupancy, estimate capacity and the slope of the congested branch, and convert the slope into a shockwave speed in mph.

### Stanford grad

The spatio-temporal literature on PeMS data: DCRNN (diffusion convolution with forward and backward random-walk matrices on a directed graph, encoder-decoder GRU, scheduled sampling), STGCN (gated temporal convolutions with Chebyshev graph convolutions), Graph WaveNet (adaptive adjacency with dilated causal convolutions), then transformer variants. Benchmarks report MAE, RMSE and MAPE at 15, 30 and 60 minutes on METR-LA and PEMS-BAY.

The stable findings: gains over persistence grow with horizon; learned or adaptive graphs help; simple models are competitive at 15 minutes. The plan's two-hop mixing corresponds to a K = 2 diffusion step, and respecting wave direction (separate upstream and downstream weights) is a physically motivated inductive bias.

**Try this.** Implement diffusion convolution with the registry's directed adjacency and compare its validation average precision at 30 and 60 minutes with LightGBM's hand-made neighbor features.

### PhD at Apple

City-scale graphs have hundreds of thousands to millions of segments with uneven probe coverage; practical systems use supersegments, embedding tables and hierarchy (the Google Maps ETA graph network is the public reference point), and update hidden states incrementally in streaming rather than recomputing windows.

Research threads: coupling learned models with simulators (cell transmission, microsimulation) for counterfactual questions such as rerouting a share of drivers; causal effects of incidents on propagation; transfer across cities; and wave physics as a regularizer in the loss.

**Try this.** Sketch a hybrid where the GRU predicts a residual over a kinematic-wave forecast, and argue where along the road and at which horizons the residual should carry the most signal.

### Guided walkthrough

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

### Check yourself

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

### Misconceptions

- 'Traffic is random.' Propagation is mechanical and fast to learn; what is hard is the onset of the first disturbance.
- 'The graph model replaces the physics.' It learns the physics from data; encoding wave direction helps it learn faster.
- 'Longer windows are better.' Beyond an hour the extra history mostly adds noise; the calendar carries the slow signal.

## Lesson 11: From scores to alerts: calibration, thresholds, lead time

_Lives in stages 10 and 11 · 16b_calibrate_select.py, 18_evaluate_lead_time.py._

**Goal.** Make the probabilities honest, choose when to warn, and measure the warning the way a driver would: how early, and how often wrongly.

### Age 5

Our helper says '70 % chance of slow'. If it says that a hundred times, it should be right about seventy of them. If it is only right forty times, its sureness is broken and we fix it.

Then we decide how sure it has to be before we shout 'slow!'. Shout too easily and people stop listening; shout too rarely and they get stuck.

**Try this.** Guess ten coin flips and say how sure you are each time. Keep score. Was your 'very sure' really more right than your 'not sure'?

### Middle school

Calibration means the probabilities are honest. We check by grouping predictions, all the 0.6-to-0.7s together, and counting how many really were slow. If the model is over-confident, we fit a correction curve, called isotonic regression, using data it did not train on.

Then the threshold. Warn at 0.3 and you catch more jams but cry wolf more; warn at 0.7 and the opposite. We pick the threshold that keeps, say, half of the warnings right, and we accept the recall that comes with it.

Lead time is the number of minutes between our first warning and the jam actually starting. That is the number a driver cares about.

**Try this.** Write ten predictions and what happened. Split them into two bins, below 0.5 and above. In each bin, compare the average prediction with the share that were slow.

### High school

Reliability diagram: bin rows by predicted probability, plot the observed share of positives per bin; the gap is calibration error. Two fixes fit on the validation split: Platt scaling (a sigmoid on the score) and isotonic regression (a monotone step function). Isotonic is more flexible and needs more data.

The threshold comes from the validation precision-recall curve at the precision the alert has to keep. Episode metrics: an episode begins at the first step of a run of label = 1; the alert is the first t in the two hours before that with p ≥ τ; lead = onset − alert; a false alarm is an alert not followed by an episode within 60 minutes, counted per station-day.

Trade-off: lower τ, longer lead and more false alarms. Plot lead time against false alarms per station-day for several τ and choose with the user in mind.

**Try this.** Make up a sequence of probabilities and labels over two hours. Compute the lead time and count false alarms at τ = 0.5 and τ = 0.3.

### Stanford undergrad

Proper scoring rules decompose into calibration and refinement; a model can rank perfectly and be badly calibrated, which is the tree-ensemble case. Isotonic regression is the pool-adjacent-violators algorithm and can overfit on a small validation set, so it needs thousands of positives; Platt scaling is safer with fewer.

Class reweighting shifts the prior, so calibrate after reweighting, on data with the true prevalence. Decision theory gives the threshold for calibrated probabilities: warn when p > C_FA / (C_FA + C_miss). Alerts are a policy, not a classifier: hysteresis (raise at τ_on, clear at a lower τ_off) prevents flapping, and the policy, not the model, is what gets evaluated on episodes.

**Try this.** Prove the cost-optimal threshold formula from expected cost and compute it for a false alarm costing 1 and a miss costing 4. Then explain why the formula is useless if the probabilities are not calibrated.

### Stanford grad

Three horizons produce three probabilities that can disagree; combine them into an expected time-to-slow or a hazard curve, or enforce monotone structure. Conformal methods give alert sets with distribution-free control of error rates over episodes, at the cost of exchangeability assumptions that time series violate and must be handled with weighting.

Calibration drifts and per-station calibration is noisy; hierarchical shrinkage toward the pooled map is the practical middle. Human factors from warning science: false-alarm ratios above roughly a half erode compliance, which sets a floor on the precision the policy must keep.

**Try this.** Formulate alerting as a sequential decision with a constraint on false alarms per station-day, and propose a policy with hysteresis that respects the constraint. Say how you would evaluate it on the test split.

### PhD at Apple

In products, calibration and thresholds live in a policy layer separate from the model, versioned, and tested with real users; alert budgets are enforced globally and evaluated on outcomes rather than on precision.

Research threads: risk-controlling prediction sets (Angelopoulos, Bates and colleagues), calibration under covariate and label shift with unlabeled target data, and decision-focused learning where the model is trained for the policy's objective rather than for log loss.

**Try this.** Design an experiment that estimates the causal effect of an alert on a driver's reroute decision and travel time, given that alerts change the traffic other drivers experience.

### Guided walkthrough

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

### Check yourself

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

### Misconceptions

- '0.5 is the natural threshold.' It is only right when a false alarm and a miss cost the same, which they never do.
- 'Calibration fixes ranking.' It is monotone; it changes the numbers, not the order.
- 'A high AP means useful alerts.' Only the episode metrics, lead time and false alarms, say that.

## Lesson 12: Shipping it on one Mac mini: memory, scheduling, monitoring

_Lives in stages 12 and 13 · 19_predict_live.py, 20_monitor.py, launchd/._

**Goal.** Run the whole thing unattended on 16 GB, know when it is failing, and retrain without breaking what works.

### Age 5

Our helper has to run all by itself every five minutes on one small computer that must never fall asleep.

It also has to check its own work. If it starts guessing badly, someone should find out, and it should learn again from the new days.

**Try this.** Set a timer to water a plant every day. What happens if the phone with the timer runs out of battery? How would you notice?

### Middle school

The Mac mini has 16 GB of memory shared by everything: the screen, the browser, and our program. A year of Fremont data is about 10 million rows; as 60 columns of numbers that is about 2.5 GB, so it fits, but only if we store numbers compactly and never run two big jobs at once. The Chrome downloader alone takes a gigabyte.

Jobs are scheduled by macOS with launchd: ask 511 every 15 minutes, predict every 5 minutes, check accuracy weekly, retrain monthly.

Monitoring means comparing last week's predictions with what really happened once the PeMS file for those days arrives, and checking that the clues still look like the ones the model trained on.

**Try this.** Compute 10,000,000 rows × 60 columns × 4 bytes. Convert to gigabytes. How many years fit in 12 GB if nothing else is running?

### High school

The serving loop: load the three models and the calibration file once; every five minutes build the last 13 steps of features with the same code that built the training table, predict, calibrate, apply the thresholds, write a JSON file, and append to a log.

Memory rules: float32, categorical ids, monthly Parquet parts, one heavy job at a time, three gigabytes kept free. Scheduling rules: launchd plists with `StartInterval`, Energy settings set to never sleep, `caffeinate` around the long download.

Monitoring: population stability index per feature against the training distribution, and average precision recomputed over the last seven days when PeMS posts them. Retrain monthly, or sooner on a real drop. The known gap: there is no real-time speed feed yet, so the loop runs in replay mode on yesterday's file.

**Try this.** Write the launchd plist for the predictor with a 300-second interval, and explain in two sentences why cron is worse on a machine that sleeps.

### Stanford undergrad

Train-serve skew is the classic failure: a second implementation of the features drifts within weeks. The defense is one module imported by both paths plus a golden-file test that recomputes one stored day and compares bit for bit. Point-in-time correctness follows: serving must only see what was available at t, and so must the training table.

Operational mechanics: idempotent jobs, atomic model swaps (write to a temporary name, rename), predictions logged with the newest input timestamp they used, and delayed scoring once labels arrive a day later. Drift: PSI = Σ (p_i − q_i) ln(p_i / q_i) over bins; a shift of one standard deviation in a normal feature gives PSI near 0.5, well past the 0.2 alarm.

Memory on unified hardware: measure with `psutil` and `tracemalloc`, set `OMP_NUM_THREADS`, and treat swap as failure. Reproducibility: pinned requirements, seeds, and a manifest with content hashes.

**Try this.** Implement PSI with ten quantile bins from the training distribution and compute it for a feature shifted by 0.25, 0.5 and 1.0 standard deviations. Where would you set the alarm?

### Stanford grad

Treat the pipeline as a reliability problem: service objectives for freshness and latency, alerts on missing data (silent failure is the default failure), canary evaluation of a retrained model on the most recent month before it replaces the champion, rollback, and a champion-challenger record.

Retraining is a control problem: frequency against drift speed against cost, with the risk of feedback loops once alerts change behavior. Continual learning without forgetting is unsolved in general; periodic full retrains with the newest month always in the window are the robust default.

Governance: raw immutable, derived reproducible, lineage in the manifest, and metric definitions versioned with the label so that a change in one is visible in the other.

**Try this.** Design the canary rule that decides when a retrained model may replace the old one, using a statistical test that respects temporal dependence, and state its false-promotion rate.

### PhD at Apple

At scale the same stage becomes streaming feature pipelines with exactly-once semantics, online feature stores, model registries, shadow deployments and staged rollouts, and on-device deployment with privacy-preserving telemetry. The Mac mini is a faithful miniature of on-device budgets: memory, energy, thermal throttling, and no operator watching.

Research threads: performative prediction and feedback loops (next month's training data contains the effect of this month's alerts), monitoring without labels through uncertainty and density-ratio drift detectors, and continual learning with guarantees.

**Try this.** Propose how you would detect that the model's own alerts are changing the label distribution, and how you would correct the training data or the objective for it.

### Guided walkthrough

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

### Check yourself

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

### Misconceptions

- 'It works on my laptop, so it will run unattended.' Sleep, reboots, a changed login page and a full disk are the normal failures; none show up in a demo.
- 'Monitoring is accuracy.' Freshness and missing data fail first; accuracy is the last thing to notice.
- 'More memory would fix it.' Filtering at ingest, float32 and one job at a time fix it; memory would only hide the design.

