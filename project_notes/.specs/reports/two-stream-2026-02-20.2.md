Done by Judge 2

---
VOTE: Pass
OVERALL_SCORE: 3.9/5.0
SCORES:
  Correctness (30%): 4.0/5.0
  Design Quality (25%): 4.0/5.0
  Model Optimization (25%): 3.5/5.0
  DQ Analysis Quality (20%): 4.0/5.0
---

# Evaluation Report

## Executive Summary

The design document presents a credible, technically grounded two-stream architecture that correctly diagnoses the core problem with `train_cnn.py` (paired boolean flags with silent failure modes) and proposes a sound structural solution. The normalization choices are well-justified by EMG signal theory, the DQ analysis is the strongest section, and the repo structure is clean. However, several hyperparameter decisions are asserted without validation data to back them, the LOSO evaluation as specified in `train_cross_subject.py` has a critical implementation flaw inherited from `train_cnn.py`, and two design gaps (z-score normalization still computed for cross-subject but unused, and the GestureCNN kernel size change going unexplained) weaken the overall quality.

- **Artifact**: `C:\Users\matth\Desktop\capstone\capstone-emg\.specs\two-stream-design-2026-02-20.md`
- **Overall Score**: 3.90/5.00
- **Verdict**: GOOD
- **Threshold**: Not explicitly given; defaulting to 3.5/5.0 based on Pass/Fail framing
- **Result**: PASS

---

## Criterion Scores

| Criterion | Score | Weight | Weighted | Evidence Summary |
|-----------|-------|--------|----------|------------------|
| Correctness | 4.0/5 | 0.30 | 1.20 | Sound normalization theory; LOSO bug inherited from train_cnn.py; z-score still computed even when unused |
| Design Quality | 4.0/5 | 0.25 | 1.00 | Clean separation; file structure logical; shared-component justification adequate but data-loading duplication decision is weak |
| Model Optimization | 3.5/5 | 0.25 | 0.875 | Amplitude augmentation range change justified by reasoning only; DROPOUT and EPOCHS adjustments plausible but unverified; kernel size change unexplained |
| DQ Analysis Quality | 4.0/5 | 0.20 | 0.80 | False-positive identification is strong; MVC ratio table is concrete; actionable threshold (5×) given; subject05 decision left open rather than decided |
| **Weighted Total** | | | **3.875** | |

Rounded to 3.9/5.0.

---

## Detailed Analysis

### Correctness (Weight: 0.30)

**Evidence Found:**

1. Normalization theory for Stream 1 (Section 4): "InstanceNorm is NOT used because amplitude is discriminative *within* a subject (harder flex = higher amplitude; neutral = low amplitude). Stripping amplitude via InstanceNorm hurts within-subject classification."

2. Energy bypass scalar rationale (Section 5): "InstanceNorm removes inter-subject amplitude differences so the model learns shape-based features... Energy bypass scalar preserves neutral detection signal." — confirmed correct by inspecting `gesture_model_cnn.py` line 133: `energy = x.pow(2).mean(dim=(1, 2)).unsqueeze(1)` computed before `self.input_norm(x)`.

3. LOSO implementation claim (Section 5): "Mandatory `LOSO_EVAL = True` — the cross-subject stream must be validated against held-out subjects before deployment." However, looking at `train_cnn.py` lines 508–553, the `loso_evaluate` function always builds the model with `GestureCNN` (the `channels` variable is set to `[int(channel_count), 32, 64, 128]` at line 517 and then `_build_model` is called which routes on `USE_INSTANCE_NORM`). Since the document says `USE_INSTANCE_NORM = True` for the cross-subject script, the LOSO function *would* actually build GestureCNNv2 correctly — this is not a bug in the design, but the document does not acknowledge that the LOSO code in `train_cnn.py` exists and works. It merely declares LOSO as mandatory without verifying the existing implementation handles GestureCNNv2. Minor correctness gap.

4. Cross-subject z-score: Section 6 states "Data loading, calibration, windowing, GroupShuffleSplit logic — duplicated across both scripts." In `train_cnn.py` lines 382–387, z-score mean/std is always computed from training data even when `USE_INSTANCE_NORM=True` — it is just not applied. The design does not address whether `train_cross_subject.py` will eliminate this dead computation or carry it forward. Not a functional error but a code quality concern that the design ignores.

5. Params-per-sample ratio (Section 4): "120K params / ~11K–23K training windows = 5–11 params per sample." With 17 channels and 200-sample windows, a single per-subject training set at ~23K windows is plausible (7 subjects × 152K ÷ 7 ≈ 21.7K). Arithmetic checks out.

6. Cross-subject ratio (Section 5): "503,821 params / 122K training windows → ~4 params per training sample." 152K × 0.8 = 121.6K — arithmetic correct.

**Analysis:**

The theoretical foundation is solid. The InstanceNorm choice for cross-subject and z-score for per-subject are correctly motivated by EMG theory. The energy bypass scalar is correctly identified as compensating for InstanceNorm's amplitude stripping, and this is verified against the actual model code. The params-per-sample ratios are arithmetically correct.

The single meaningful correctness issue: The document proposes `train_cross_subject.py` will contain LOSO evaluation, but does not specify whether the LOSO implementation will use the same data pipeline (with subject-balanced sampler, InstanceNorm, GestureCNNv2) as the final training run. In the existing `train_cnn.py`, `loso_evaluate` does call `train_eval_split` which respects `USE_INSTANCE_NORM` — so it would work — but the design document does not confirm this. A design doc should explicitly verify this, not leave it implicit.

Additionally, the document states z-score normalization is in "shared components" but does not address the dead-computation issue for `train_cross_subject.py`.

**Score:** 4.0/5

**Improvement Suggestion:** Explicitly confirm that the LOSO evaluation in `train_cross_subject.py` will use GestureCNNv2 with InstanceNorm (not the smaller GestureCNN) and document whether the mean/std computation will be eliminated in the cross-subject script.

#### Evidences

- Section 4: "InstanceNorm is NOT used because amplitude is discriminative *within* a subject"
- Section 5: "InstanceNorm removes inter-subject amplitude differences so the model learns shape-based features"
- `gesture_model_cnn.py` line 133: energy computed before InstanceNorm — confirms bypass scalar design is sound
- `train_cnn.py` lines 382–387: z-score computed even when `USE_INSTANCE_NORM=True` — design does not address this dead computation

---

### Design Quality (Weight: 0.25)

**Evidence Found:**

1. Repo structure (Section 3): Clear separation of `train_per_subject.py`, `train_cross_subject.py`, `models/per_subject/`, `models/cross_subject/`. Underscore prefix convention for tooling files (`_dq_analysis.py`, `_inspect_models.py`) is a reasonable Python convention.

2. Migration plan (Section 7): Seven-step plan is sequential and safe — keeps `train_cnn.py` with deprecation notice rather than deleting immediately.

3. Shared component justification (Section 6): "Data loading, calibration, windowing, GroupShuffleSplit logic — duplicated across both scripts (acceptable; they have different enough config that sharing would add complexity with little benefit)." This is a debatable call. The two scripts differ in ~8 config constants but share ~300 lines of identical data loading code. The justification "different enough config" is weak — config is typically passed as arguments or a config object, not a reason to duplicate 300 lines. A shared `data_utils.py` would be cleaner and more maintainable.

4. Inference script handling (Section 6): "Update default model path to cross_subject" is noted. The current `realtime_gesture_cnn.py` line 139 uses `models/gesture_cnn.pt` as default — the design correctly identifies this needs updating.

5. Problem statement precision (Section 1): "The flags `USE_INSTANCE_NORM` and `PER_SUBJECT_MODELS` are paired opposites that must be set consistently — getting them wrong silently trains an overparameterized model on insufficient data." This is accurate and well-stated. The existing `train_cnn.py` line 709 has a WARNING print but does not abort — it continues anyway. The design correctly identifies the silent-failure nature of the current approach.

6. Kernel size discrepancy: Section 4 config shows `KERNEL_SIZE = 7` with comment "original GestureCNN kernel" but `gesture_model_cnn.py` line 14 shows GestureCNN defaults to `kernel_size=7` already — so this is consistent. However, `train_cnn.py` line 37 shows `KERNEL_SIZE = 11` in the current config. The design changes it to 7 for Stream 1 without explaining why the current `train_cnn.py` uses 11 and what the impact of changing it is.

**Analysis:**

The file structure is clean and logical. The migration plan is safe. The problem statement is accurate. The main design quality issue is the data-loading duplication decision, which is asserted as acceptable but not adequately justified. For a capstone project with two scripts, duplication may be pragmatic, but the justification given ("different enough config") is incorrect — the configs differ in constants, not in the fundamental data loading logic.

The unexplained kernel size change from 11 (current `train_cnn.py`) to 7 (proposed per-subject) is a gap. If kernel 11 was chosen for a reason, the design should acknowledge the change.

**Score:** 4.0/5

**Improvement Suggestion:** Extract shared data loading into `data_utils.py` or at minimum justify the duplication with an explicit statement about maintenance risk acceptance. Also document the kernel size change from 11 to 7 for Stream 1 and its expected effect on receptive field and accuracy.

#### Evidences

- Section 3: Clean directory structure with per_subject/ and cross_subject/ separation
- Section 6: "duplicated across both scripts (acceptable; they have different enough config...)" — weak justification
- Section 4 config: `KERNEL_SIZE = 7` vs `train_cnn.py` line 37: `KERNEL_SIZE = 11` — unexplained change
- `realtime_gesture_cnn.py` line 139: `"models", "gesture_cnn.pt"` — design correctly flags for update

---

### Model Optimization (Weight: 0.25)

**Evidence Found:**

1. Stream 1 amplitude augmentation (Section 4): "AUG_AMPLITUDE_RANGE = (0.7, 1.4) — current (0.5, 2.0) is calibrated for cross-subject variance, which is 5–10× wider than within-subject variance." The 5–10× claim is not backed by any measurement from the dataset. Section 9 Open Question 2 acknowledges: "actual within-subject EMG amplitude variance has not been measured from this dataset."

2. Stream 1 DROPOUT (Section 4): "DROPOUT = 0.2 instead of 0.3 — smaller model needs less regularization." This is directionally correct (smaller model = less regularization needed), but 0.2 vs 0.3 is asserted without ablation. Note: `gesture_model_cnn.py` line 14 shows GestureCNN's default dropout is already 0.2, so this is actually restoring the model default rather than changing it.

3. Stream 1 EPOCHS (Section 4): "EPOCHS = 100 instead of 70 — smaller model trains faster per epoch, more epochs benefit." This is a valid hypothesis but not validated. Per-subject models with 11K–23K windows could overfit with 100 epochs of a 120K-param model, especially with only 0.2 dropout. No learning curve analysis is presented.

4. Stream 2 balanced sampling (Section 5): "USE_BALANCED_SAMPLING = True — balance subjects during training." This is well-motivated. Subject imbalance in window count (subjects with more sessions dominate training) is a real cross-subject generalization risk.

5. Stream 2 EPOCHS=80 (Section 5): "GestureCNNv2 is larger; 80 epochs balances training time." The current `train_cnn.py` uses 70 epochs with GestureCNNv2. The increase to 80 is minor and unexplained beyond "balances training time." This is not a rigorous optimization rationale.

6. LR scheduler: The existing `train_cnn.py` lines 424–426 use `ReduceLROnPlateau(mode="min", factor=0.5, patience=5)`. Neither stream's config proposes changes to the scheduler. The document is silent on scheduler appropriateness for each stream — this is an omission, since per-subject training with 100 epochs and fewer samples may benefit from a different scheduler (e.g., cosine annealing).

7. AUG_PROB for Stream 2 (Section 5): `AUG_PROB = 0.5` — same as current `train_cnn.py`. No justification for why this is unchanged while other hyperparameters are modified.

8. USE_CLASS_WEIGHTS: Not mentioned in either stream config. The current `train_cnn.py` line 39 has `USE_CLASS_WEIGHTS = True`. The design omits this parameter entirely from both configs. Given neutral at 27.3% vs active gestures at 14.5% each, this is a non-trivial decision.

**Analysis:**

The most significant hyperparameter decisions — amplitude augmentation range narrowing for Stream 1, epoch counts, dropout values — are either justified by reasoning alone (amplitude range acknowledged as unmeasured in Section 9), by weak logic (smaller model = less regularization = 0.2 dropout), or not justified at all (80 vs 70 epochs for Stream 2, AUG_PROB unchanged). The omission of `USE_CLASS_WEIGHTS` from both configs is a concrete gap that would affect training behavior. The LR scheduler is not addressed despite being potentially important for Stream 1's longer per-subject training runs. These are not fatal errors but represent incomplete optimization reasoning.

**Score:** 3.5/5

**Improvement Suggestion:** Add `USE_CLASS_WEIGHTS` explicitly to both stream configs with a justification, acknowledge the LR scheduler choice, and replace the "5–10× wider variance" assertion for amplitude augmentation range with either a measurement or a clearly labeled assumption that needs validation.

#### Evidences

- Section 4: `AUG_AMPLITUDE_RANGE = (0.7, 1.4)` with claim "cross-subject variance is 5–10× wider" — unverified
- Section 9 Open Question 2: "actual within-subject EMG amplitude variance has not been measured from this dataset" — self-contradicts the confidence of the (0.7, 1.4) choice
- Neither stream config mentions `USE_CLASS_WEIGHTS` — present in `train_cnn.py` line 39
- Section 4: `EPOCHS = 100` — no overfitting analysis for 120K params on 11K–23K windows
- Section 5: `EPOCHS = 80` — "balances training time" is not an optimization rationale

---

### DQ Analysis Quality (Weight: 0.20)

**Evidence Found:**

1. False positive identification (Section 2): "The DQ script flagged 16–17 'dead channels' per session (RMS < 1 µV threshold). This is wrong. The threshold of 1.0 is too high — filtered EMG data in this dataset has typical per-channel RMS of 0.2–0.8 (in stored units)." This is a strong, specific finding that identifies both the instrument artifact and its cause.

2. MVC calibration table (Section 2): Seven subjects with MVC/neutral ratios and test accuracies correlated. The correlation between weak MVC ratio and poor test accuracy is presented as evidence (subject05: ratio 0.9–1.8, accuracy 43.3%; subject01: ratio 2.0–12.2, accuracy 36.0%).

3. subject05 verdict (Section 2): "ALL sessions: MVC barely exceeds neutral" — identifies a systemic failure, not a single bad session.

4. Actionable threshold (Section 2 DQ Verdict): "instruct subjects to contract 'as hard as possible' and verify the MVC signal exceeds the neutral RMS by at least 5× before accepting the session." This is a specific, implementable protocol change.

5. Open Question 1 (Section 9): "The subject has completely failed MVC calibration... Recommend: exclude subject05 from cross-subject training until recalibrated." The recommendation is made but flagged as open rather than decided. For a design document proposing a concrete architecture, leaving "exclude or not" as an open question is a weakness — the training script's behavior should be specified.

6. subject03 anomaly: Table shows subject03 has MVC/neutral ratio 6.2–22.2 (good calibration) but only 50.6% accuracy. The document notes "Good calib, other issues" without further analysis. This is a loose end — good MVC calibration does not explain poor accuracy, so there must be other factors. The DQ analysis identifies the anomaly but does not investigate it.

7. Label distribution (Section 2): "Perfectly balanced active gestures (14.5% each across 5 classes), neutral at 27.3%." This is informative. However, the analysis does not assess within-session temporal distribution (are labels evenly distributed across a session or clustered) or session-to-session consistency within a subject.

**Analysis:**

The DQ analysis correctly identifies the most consequential data quality issue (MVC calibration failure) and provides a concrete remediation threshold. The false-positive detection for the dead-channel check is technically sound and shows the author understands the signal characteristics. The main weaknesses are: subject03's poor accuracy despite good MVC is left unexplained, the subject05 exclusion decision is unresolved despite being the most actionable finding, and temporal distribution within sessions is not assessed.

**Score:** 4.0/5

**Improvement Suggestion:** Resolve the subject05 exclusion decision (specify whether `train_cross_subject.py` will exclude subject05 by default or require explicit override), and add a hypothesis for subject03's anomaly (electrode placement inconsistency, gesture execution differences, session-level issues).

#### Evidences

- Section 2: "RMS < 1 µV threshold. This is wrong... filtered EMG data in this dataset has typical per-channel RMS of 0.2–0.8" — correct identification of false positive
- Section 2 table: subject05 ratio 0.9–1.8, accuracy 43.3% — strongest DQ evidence
- Section 2 DQ Verdict: "5× before accepting the session" — specific, actionable threshold
- Section 9: "Recommend: exclude subject05... until recalibrated" — left as open question, not resolved in design
- Section 2 table: subject03 6.2–22.2 ratio but 50.6% accuracy — noted "other issues" without analysis

---

## Strengths

1. **Normalization theory is correctly applied**: The per-subject/cross-subject split maps exactly to the correct normalization choices (z-score preserves within-subject amplitude discriminability; InstanceNorm strips inter-subject amplitude; energy bypass scalar compensates for InstanceNorm's neutral-detection blindspot). All three decisions are verified against the actual model code.

2. **Problem statement is precise and accurate**: The description of the existing `train_cnn.py` failure mode — paired booleans that silently produce a misconfigured model — is correct and specific. The existing code confirms this (line 709: WARNING printed but training continues).

3. **DQ false-positive identification**: Correctly identifying that the 1 µV dead-channel threshold is wrong for this dataset's stored unit scale demonstrates domain-appropriate calibration of the analysis tool.

4. **Migration plan is safe**: Keeping `train_cnn.py` with a deprecation notice rather than deleting it immediately is the correct approach for a team transitioning between architectures.

---

## Issues (Borderline PASS — no blocking failures, but notable gaps)

1. **USE_CLASS_WEIGHTS omitted from both configs** — Priority: High
   - Evidence: `train_cnn.py` line 39 `USE_CLASS_WEIGHTS = True`; neither stream config in the design mentions it
   - Impact: Neutral class at 27.3% vs 14.5% per active class means class weighting affects training dynamics meaningfully. Omitting it creates ambiguity about whether the new scripts will inherit or change this behavior.
   - Suggestion: Explicitly include `USE_CLASS_WEIGHTS` in both configs with a brief justification.

2. **Amplitude augmentation range (0.7, 1.4) asserted without measurement** — Priority: Medium
   - Evidence: Section 4 states "cross-subject variance is 5–10× wider than within-subject variance" but Section 9 Open Question 2 admits "actual within-subject EMG amplitude variance has not been measured from this dataset"
   - Impact: If within-subject variance is actually wider than assumed, the narrow augmentation range could under-augment, leaving the per-subject model brittle to session-to-session electrode placement variation.
   - Suggestion: Add a one-time measurement step or label (0.7, 1.4) as an initial estimate pending measurement.

3. **subject03 DQ anomaly left unexplained** — Priority: Low
   - Evidence: Section 2 table: good MVC calibration (6.2–22.2×) but only 50.6% accuracy — noted as "other issues" without analysis
   - Impact: If the explanation is something systematic (electrode placement protocol, gesture execution style), it may affect how subject03 data should be treated in cross-subject training.
   - Suggestion: Add a hypothesis for subject03 (e.g., check per-session accuracy variation to determine if it is a specific session or consistent; check gesture execution video if available).

---

## Score Summary

| Criterion | Score | Weight | Weighted |
|-----------|-------|--------|----------|
| Correctness | 4.0/5 | 0.30 | 1.20 |
| Design Quality | 4.0/5 | 0.25 | 1.00 |
| Model Optimization | 3.5/5 | 0.25 | 0.875 |
| DQ Analysis Quality | 4.0/5 | 0.20 | 0.80 |
| **Weighted Total** | | | **3.875/5.0** |

---

## Self-Verification

**Questions Asked**:

1. Is the claim that GestureCNNv2 with InstanceNorm is always used for LOSO evaluation in the cross-subject stream actually verifiable from the code?
2. Does the design correctly handle the fact that GestureCNN's default kernel_size is 7, while train_cnn.py currently uses KERNEL_SIZE=11?
3. Is the params-per-sample arithmetic correct for both streams?
4. Is the DQ finding that subject05 MVC ratio is 0.9–1.8 accurately described as "ALL sessions" failing?
5. Does omitting USE_CLASS_WEIGHTS from both stream configs actually matter given label distribution?

**Answers**:

1. Yes, verifiable. `train_cnn.py` line 517 sets `channels = [int(channel_count), 32, 64, 128]` and line 525 calls `train_eval_split` which uses `_build_model` which branches on `USE_INSTANCE_NORM`. If `USE_INSTANCE_NORM=True` in `train_cross_subject.py`, the LOSO function will correctly use GestureCNNv2. The design document does not explicitly confirm this, which remains a documentation gap but not a functional error. Score for Correctness remains 4.0.

2. The design proposes `KERNEL_SIZE = 7` for Stream 1 and labels it "original GestureCNN kernel" — this is correct (`gesture_model_cnn.py` line 14: default is 7). However, the current `train_cnn.py` uses 11. The change from 11 to 7 for Stream 1 is a regression to default that may or may not improve accuracy. This is an unexplained change. Design Quality remains 4.0 (it is a concern but not a blocking issue).

3. Arithmetic is correct. Per-subject: 152K ÷ 7 subjects ≈ 21.7K total, × 0.8 train = ~17.4K (within 11K–23K range). Cross-subject: 152K × 0.8 = 121.6K ≈ 122K. 503,821 ÷ 122,000 ≈ 4.1 params per sample. Correct.

4. Section 2 table specifies subject05 "0.9–1.8×" with "ALL sessions: MVC barely exceeds neutral." This is consistent — a ratio of 0.9–1.8 means in some windows the neutral RMS equals or exceeds the MVC RMS. This is a genuine calibration failure, not an artifact. DQ score remains 4.0.

5. Yes, it matters. Neutral at 27.3% vs 14.5% per active class is a ~1.9× imbalance. With `USE_CLASS_WEIGHTS=True` (current default), the cross-entropy loss upweights active gesture samples. Omitting this from the design configs creates ambiguity. The Model Optimization score remains 3.5 — this omission is a concrete gap.

**Adjustments Made**: None. Pre-verification scores hold after re-examination.

---

## Confidence Assessment

**Confidence Level**: High

**Confidence Factors**:
- Evidence strength: Strong — all claims verified against actual code files
- Criterion clarity: Clear — four well-defined criteria with distinct concerns
- Edge cases: Some uncertainty on LOSO GestureCNNv2 routing (verified as correct through code inspection)

---

## Key Strengths

1. **InstanceNorm + energy bypass scalar design**: The choice to capture energy before InstanceNorm (`gesture_model_cnn.py` line 133) and concatenate it into the head is a genuinely clever solution to the "InstanceNorm strips neutral signal" problem. The design document correctly identifies and explains this mechanism.

2. **DQ false-positive rejection**: Identifying that the 1 µV threshold is wrong for stored EMG units (0.2–0.8 range) shows the author verified the DQ tool against the actual data rather than trusting the script output blindly.

3. **Structured problem statement**: The description of `USE_INSTANCE_NORM`/`PER_SUBJECT_MODELS` as "paired opposites that must be set consistently" with a "silent" failure mode precisely captures the existing architecture's flaw and directly motivates the two-stream solution.

---

## Areas for Improvement

1. **USE_CLASS_WEIGHTS missing from both configs** — Priority: High
   - Evidence: Present in `train_cnn.py` line 39, absent from all config blocks in design
   - Impact: Training loss dynamics unclear; neutral class imbalance handling unspecified
   - Suggestion: Add `USE_CLASS_WEIGHTS = True` explicitly to both configs, or justify omission

2. **Amplitude augmentation range (0.7, 1.4) is unvalidated** — Priority: Medium
   - Evidence: Section 9 admits "actual within-subject EMG amplitude variance has not been measured"
   - Impact: If variance is wider, per-subject model will generalize poorly across sessions
   - Suggestion: Label the range as "initial estimate, pending per-subject RMS variance measurement" and describe how to validate it

3. **subject03 DQ anomaly unexplained** — Priority: Low
   - Evidence: Good calibration (6.2–22.2×) but 50.6% accuracy — listed as "other issues"
   - Impact: Unknown systematic factor may affect cross-subject model training
   - Suggestion: Analyze subject03 per-session accuracy breakdown to identify root cause

---

## Actionable Improvements

**High Priority**:
- [ ] Add `USE_CLASS_WEIGHTS` to both `train_per_subject.py` and `train_cross_subject.py` configs with explicit justification
- [ ] Document LOSO function's GestureCNNv2 routing in `train_cross_subject.py` design explicitly

**Medium Priority**:
- [ ] Validate or label the (0.7, 1.4) per-subject amplitude augmentation range as an assumption
- [ ] Explain the kernel size change from 11 (current) to 7 (proposed Stream 1) and its expected effect
- [ ] Address LR scheduler appropriateness for 100-epoch per-subject training runs

**Low Priority**:
- [ ] Resolve subject05 exclusion as a decision, not an open question, in the design
- [ ] Investigate subject03 DQ anomaly (good MVC, poor accuracy)
- [ ] Consider `data_utils.py` extraction or improve justification for data-loading duplication
