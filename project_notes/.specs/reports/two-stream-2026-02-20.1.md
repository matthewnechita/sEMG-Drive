Done by Judge 1

---
VOTE: Pass
OVERALL_SCORE: 3.6/5.0
SCORES:
  Correctness (30%): 3.5/5.0
  Design Quality (25%): 4.0/5.0
  Model Optimization (25%): 3.5/5.0
  DQ Analysis Quality (20%): 3.5/5.0
---

# Evaluation Report

## Executive Summary

The two-stream design document is a coherent, well-reasoned architecture proposal that correctly identifies the core problems in the existing monolithic `train_cnn.py` and proposes a defensible solution. However, it contains a material factual error regarding `AUG_AMPLITUDE_RANGE` (presented as a config constant that does not exist in the codebase), a misleading claim about sanity check placement, and several underspecified areas. The DQ analysis is directionally sound but lacks methodological grounding for its threshold claims. The design earns a pass but requires targeted revisions before implementation.

- **Artifact**: `C:\Users\matth\Desktop\capstone\capstone-emg\.specs\two-stream-design-2026-02-20.md`
- **Overall Score**: 3.60/5.00
- **Verdict**: ACCEPTABLE
- **Threshold**: Not explicitly stated; applying 3.0/5.0 as minimum pass
- **Result**: PASS

---

## Criterion Scores

| Criterion | Score | Weight | Weighted | Evidence Summary |
|-----------|-------|--------|----------|-----------------|
| Correctness (30%) | 3.5/5 | 0.30 | 1.05 | AUG_AMPLITUDE_RANGE doesn't exist as config; sanity check claim is wrong; ReduceLROnPlateau omitted |
| Design Quality (25%) | 4.0/5 | 0.25 | 1.00 | Clean file structure, clear stream separation, acceptable code duplication rationale |
| Model Optimization (25%) | 3.5/5 | 0.25 | 0.875 | Hyperparameter choices are reasonable but unvalidated; augmentation range is a guess by design's own admission |
| DQ Analysis Quality (20%) | 3.5/5 | 0.20 | 0.70 | MVC table is actionable; false positive debunking is correct but threshold justification is thin |
| **Weighted Total** | | | **3.625** | |

---

## Detailed Analysis

### Correctness (Weight: 0.30)

**Practical Check**: Cross-referenced all config constants in the design against `train_cnn.py` source code.

**Evidence Found:**

1. The design proposes `AUG_AMPLITUDE_RANGE = (0.7, 1.4)` (Section 4) and `AUG_AMPLITUDE_RANGE = (0.5, 2.0)` (Section 5) as top-level script constants. However, in `train_cnn.py`, amplitude scaling is hardcoded inside `_aug_amplitude` at line 259: `factors = np.random.uniform(0.5, 2.0, ...)`. There is no `AUG_AMPLITUDE_RANGE` config constant anywhere in the codebase. The design presents this as if it already exists or is trivially added, but the function signature `_aug_amplitude(xb, p)` does not accept a range parameter. Implementing the per-subject narrow range requires refactoring `_aug_amplitude` to accept the range — a non-trivial code change that the design never flags.

2. Section 4 states: "Config sanity check moved to `train_per_subject.py` (where the mismatch is more likely)." This is wrong in two ways. First, the existing sanity check in `train_cnn.py` (line 709-717) catches `USE_INSTANCE_NORM=True AND PER_SUBJECT_MODELS=True`. After the split, `train_per_subject.py` will have `USE_INSTANCE_NORM=False` and `PER_SUBJECT_MODELS=True` hardcoded — the mismatch condition can never occur within that script. The check would be useless there. Second, in `train_cross_subject.py` the mismatch can also never occur since both flags are fixed. The entire sanity check becomes unnecessary, not relocated. The design's claim about where the mismatch is "more likely" is backwards and misleading.

3. The design omits the `ReduceLROnPlateau` scheduler (train_cnn.py lines 424-426) from both stream configs. This scheduler is active in the current codebase with `patience=5, factor=0.5` and meaningfully affects convergence. Neither `train_per_subject.py` nor `train_cross_subject.py` config blocks mention whether it is retained, modified, or removed. This is a significant training parameter omission.

4. The design says `KERNEL_SIZE = 7` for per-subject (Section 4 config block). The existing `train_cnn.py` has `KERNEL_SIZE = 11` (line 37), and the GestureCNNv2 stem uses kernel_size=11 (gesture_model_cnn.py line 97). Changing to kernel_size=7 for GestureCNN is valid (it is the original GestureCNN kernel per gesture_model_cnn.py line 14 default), but the design should explain this choice explicitly. The current codebase default of 11 was presumably chosen for a reason that is now being silently reverted.

5. The claim that `USE_AUGMENTATION = True` with `AUG_PROB = 0.4` for per-subject is a change from the current default of `USE_AUGMENTATION = False` (train_cnn.py line 51). This is a meaningful change that could affect existing per-subject model baselines, but the design acknowledges this implicitly by listing it as a change.

6. Cross-subject architecture claim: "Pools all 152K windows from 7 subjects → 122K training windows" — 80% of 152K = ~121.6K. This arithmetic is correct.

7. Per-subject param ratio: "120K params / ~11K–23K training windows = 5–11 params per sample." A single subject with 4 sessions × ~3K windows per session = ~12K windows; with 8 sessions ~24K. At 80% train split: 9.6K–19.2K. The 11K–23K range is approximately consistent.

8. The statement "InstanceNorm is NOT used because amplitude is discriminative *within* a subject" is correct EMG signal theory — within-subject amplitude encodes contraction intensity and is a genuine discriminative feature for gesture classification.

**Analysis**: The AUG_AMPLITUDE_RANGE error is the most critical flaw. It is a specific, concrete technical claim about config structure that is wrong — the constant does not exist and the function would need refactoring. The sanity check claim is wrong but lower impact (the check would just be dead code). The scheduler omission is a specification gap. These are not trivial issues; they would cause confusion during implementation.

**Score: 3.5/5**

**Improvement**: Acknowledge that `_aug_amplitude` must be refactored to accept a `scale_range` parameter tuple, and flag this as an implementation task in the migration plan. Remove the sanity check relocation claim or replace it with "the sanity check becomes unnecessary and should be removed."

---

### Design Quality (Weight: 0.25)

**Practical Check**: Verified proposed file structure against existing repo layout. Confirmed `gesture_model_cnn.py` exports `GestureCNN`, `GestureCNNv2`, `CnnBundle`, `load_cnn_bundle`, `quick_finetune`.

**Evidence Found:**

1. Section 3 proposes a clear repo structure with `models/per_subject/` and `models/cross_subject/` subdirectories. This is a genuine improvement over the current flat `models/` directory where per-subject and cross-subject models coexist with similar names.

2. "The old `train_cnn.py` is replaced by the two dedicated scripts. It can be kept for reference during transition." — clear migration path with deprecation window.

3. Section 6: "Data loading, calibration, windowing, GroupShuffleSplit logic — duplicated across both scripts (acceptable; they have different enough config that sharing would add complexity with little benefit)." The rationale for duplication is stated and defensible, though debatable. A shared `data_loading.py` module would prevent drift, but the tradeoff is acknowledged.

4. The `_dq_analysis.py` and `_inspect_models.py` naming convention (underscore prefix) correctly signals tooling/non-production scripts. This is a good convention choice.

5. Section 6 claims `realtime_gesture_cnn.py` requires "no changes needed except updating default model path to cross_subject." The current default path in `realtime_gesture_cnn.py` line 139 is `models/gesture_cnn.pt` — updating to `models/cross_subject/gesture_cnn_v2.pt` is a trivial one-line change. This is correctly characterized.

6. The shared `gesture_model_cnn.py` approach is sound — both streams use the same model classes and `CnnBundle` format, ensuring backward compatibility with existing saved models. The `load_cnn_bundle` function's `_resolve_architecture` method correctly handles both arch types.

7. The two streams are clearly delineated with separate purposes, architectures, normalization strategies, and output directories. The design avoids the current anti-pattern of boolean-flag-controlled branching within a single script.

8. Minor issue: The migration plan in Section 7 says "Move existing per-subject `.pt` files to `models/per_subject/`" but doesn't address updating inference command-line calls to use new paths. Users who have scripts referencing `models/Matthew_cnn.pt` would break.

**Analysis**: The repo structure is clean and logical. Stream separation is genuinely well-handled. The duplication rationale, while debatable, is at least stated. The design correctly identifies what changes and what stays stable. The missing migration detail (path updates for existing model references) is a minor gap.

**Score: 4.0/5**

**Improvement**: Add a step to the migration plan noting that any existing scripts or aliases that reference per-subject model paths (e.g., `models/Matthew_cnn.pt`) must be updated to `models/per_subject/Matthew_cnn.pt`.

---

### Model Optimization (Weight: 0.25)

**Practical Check**: Checked existing hyperparameters in `train_cnn.py` against proposed values. Verified parameter counts via model architecture.

**Evidence Found:**

1. **Per-subject epochs**: Design proposes `EPOCHS = 100` vs current 70. Rationale: "GestureCNN is smaller, trains faster per epoch, more epochs benefit." This is plausible but unverified — no learning curve data is cited. The existing model achieves "81.4% per-subject" which presumably includes training to convergence. Going to 100 epochs could either help (more convergence time) or not (if ReduceLROnPlateau already converges by epoch ~50).

2. **Per-subject dropout**: `DROPOUT = 0.2` — this matches GestureCNN's default in `gesture_model_cnn.py` (line 14: `dropout=0.2`). The rationale is sound: smaller models need less regularization. However, the existing `train_cnn.py` uses `DROPOUT = 0.3` even for per-subject, so this is a meaningful change.

3. **AUG_AMPLITUDE_RANGE = (0.7, 1.4) for per-subject**: Section 9 Open Question 2 explicitly states "actual within-subject EMG amplitude variance has not been measured from this dataset. Could be validated..." This is a significant admission — the core hyperparameter difference between streams (the amplitude augmentation range) is unvalidated. The design proposes a value without evidence.

4. **Cross-subject AUG_PROB = 0.5 vs per-subject 0.4**: No justification given for the difference. The current `train_cnn.py` has `AUG_PROB = 0.5`. This seems like an arbitrary reduction for per-subject with no stated rationale.

5. **Cross-subject EPOCHS = 80 vs current 70**: Rationale: "GestureCNNv2 is larger; 80 epochs balances training time." This reasoning is odd — larger models typically need *more* epochs, not fewer. The current 70 is already low. The design gives 100 to the smaller per-subject model and only 80 to the larger cross-subject model. The justification "balances training time" is a computational convenience argument rather than a convergence argument.

6. **LR per-subject = 3e-4 vs cross-subject = 1e-4**: The rationale that a smaller model can use a higher LR is reasonable in principle (larger gradient updates are less catastrophic for a simpler loss landscape). The 3e-4 value is consistent with Adam's commonly cited optimal default.

7. **`USE_CLASS_WEIGHTS` is not mentioned in either stream's config block.** The existing `train_cnn.py` has `USE_CLASS_WEIGHTS = True` (line 39). Since label distribution is "perfectly balanced active gestures (14.5% each)" per the DQ section, class weights are approximately uniform anyway (neutral at 27.3% would get slightly lower weight). Omitting this flag from the design is a minor spec gap.

8. **`USE_BALANCED_SAMPLING = False` for per-subject**: Correct. Within a single subject, there is no subject imbalance. The class imbalance (slightly more neutral) could still warrant class weighting, but the design doesn't address this.

9. **`LOSO_EVAL = False` for per-subject**: Correct and well-justified — LOSO measures cross-subject generalization, which is not the per-subject stream's concern.

10. **`LOSO_EVAL = True` for cross-subject**: Mandatory and correctly specified. However, the design doesn't address that LOSO is computationally expensive — with 7 subjects and 80 epochs each, this is 7 × 80 = 560 epochs of training. For a capstone project this matters.

**Analysis**: The hyperparameter choices are broadly defensible but the amplitude augmentation range is explicitly unvalidated (Open Question 2), the epoch count ordering is backwards (more epochs for the smaller model), and AUG_PROB reduction for per-subject has no stated rationale. The scheduler omission (from the Correctness section) compounds here — whether ReduceLROnPlateau remains active materially affects whether epoch count matters.

**Score: 3.5/5**

**Improvement**: Justify the epoch 100 (per-subject) vs 80 (cross-subject) ordering explicitly — if it's because LOSO multiplies total training time, say so. Commit to measuring actual within-subject amplitude variance before finalizing `AUG_AMPLITUDE_RANGE = (0.7, 1.4)` rather than treating it as a spec decision.

---

### DQ Analysis Quality (Weight: 0.20)

**Practical Check**: Evaluated the MVC ratio table, false positive analysis, and actionable recommendations against stated methodology.

**Evidence Found:**

1. **False positive debunking** (Section 2, "What is a false positive"): "The DQ script flagged 16–17 'dead channels' per session (RMS < 1 µV threshold). This is wrong. The threshold of 1.0 is too high — filtered EMG data in this dataset has typical per-channel RMS of 0.2–0.8 (in stored units)." This analysis is correct in direction — 1 µV is indeed a threshold appropriate for raw (unfiltered, unnormalized) EMG but not for already-normalized data. However, the claim "typical per-channel RMS of 0.2–0.8 (in stored units)" is asserted without showing the computation. No percentile distributions are shown, no channel-level RMS histogram is cited. The reader must trust the claim.

2. **MVC/Neutral ratio table**: The table presents per-subject ratio ranges and correlates them with test accuracy. The correlation direction is consistent (subject05 has the lowest ratios AND one of the lowest accuracies; subject02 has solid ratios AND the highest accuracy). However, subject03 is an anomaly: "6.2–22.2× | 50.6% | Good calib, other issues" — the table assigns good calibration but poor accuracy, then punts to "other issues" without elaboration. This is a gap in the analysis.

3. **subject05 verdict**: "MVC calibration failed across all sessions; consider recollection or excluding MVC normalization for this subject." This is actionable and specific. The recommendation to exclude from cross-subject training (Section 9, Open Question 1) is correctly flagged but left open rather than decided.

4. **DQ Verdict section**: "PASS overall — no missing data, no corrupt files, no channel/label inconsistencies. FLAG subject05. WARN subject01, subject04, subject06." This tiered verdict (PASS / FLAG / WARN) is a clean, actionable structure.

5. **"Improve the MVC collection protocol: instruct subjects to contract 'as hard as possible' and verify the MVC signal exceeds the neutral RMS by at least 5× before accepting the session."** — This is a specific, actionable protocol change. The 5× threshold is stated as a criterion but no prior literature or empirical basis is cited for why 5× is the right threshold (vs. 3× or 10×).

6. **Label distribution**: "Perfectly balanced active gestures (14.5% each across 5 classes), neutral at 27.3%." With 5 classes × 14.5% + 27.3% = 99.8% — arithmetic checks out. The "perfectly balanced" characterization is appropriate.

7. **No mention of temporal autocorrelation or session-to-session drift.** EMG signals have known session-to-session electrode placement variability. The DQ analysis doesn't assess whether within-subject accuracy degradation across sessions is consistent with expected electrode drift. This is a missed analysis opportunity.

8. **No assessment of signal-to-noise ratio per session** beyond the MVC/neutral ratio proxy. The MVC ratio is a reasonable SNR proxy but not the same thing.

**Analysis**: The DQ section is the most practically useful part of the document. The MVC ratio table is informative and the tiered verdict is actionable. The main weaknesses are: subject03's anomaly is unexplained, the false positive threshold claim is asserted without computation details, and the 5× MVC acceptance threshold is not grounded in prior work. These are gaps in rigor, not in actionability.

**Score: 3.5/5**

**Improvement**: Explain subject03's 50.6% accuracy despite good calibration (is it fewer sessions, data collection problems, signal artifact?). Provide the actual channel RMS distribution statistics that justify the "0.2–0.8 (in stored units)" claim rather than asserting it.

---

## Strengths

1. **Correct Normalization Rationale**: The document accurately explains why InstanceNorm helps cross-subject but hurts per-subject (Section 4 rationale). The energy bypass scalar explanation is technically sound and grounded in the actual model code (`gesture_model_cnn.py` lines 131-139 confirm the pre-norm energy capture).

2. **Clean Stream Separation**: The two-stream design eliminates the error-prone boolean-flag pattern from `train_cnn.py`. Having separate scripts with fixed configs prevents the silent misconfiguration risk correctly identified in Section 1.

3. **Mandatory LOSO for Cross-Subject**: Correctly mandating `LOSO_EVAL = True` for the cross-subject stream (Section 5) and honestly reporting expected LOSO accuracy of 60–75% rather than inflating expectations.

4. **Honest Open Questions**: Section 9 explicitly calls out three unresolved decisions (subject05 exclusion, amplitude augmentation validation, LOSO accuracy threshold). This is intellectually honest and prevents false confidence in the design.

5. **Backward-Compatible CnnBundle**: Correctly identifying that the `CnnBundle` format and `load_cnn_bundle` require no changes — the inference script works with either stream's output.

---

## Issues

1. **AUG_AMPLITUDE_RANGE Does Not Exist** - Priority: High
   - Evidence: `train_cnn.py` line 259 hardcodes `np.random.uniform(0.5, 2.0)` with no configurable range parameter
   - Impact: The key differentiating hyperparameter between the two streams cannot be implemented without refactoring `_aug_amplitude`
   - Suggestion: Add to migration plan: "Refactor `_aug_amplitude(xb, p)` to `_aug_amplitude(xb, p, scale_range=(0.5, 2.0))` to make the range configurable via the top-level constant"

2. **Sanity Check Claim is Wrong** - Priority: Medium
   - Evidence: Design says check moves to `train_per_subject.py`; after split the mismatch condition cannot occur in either script
   - Impact: Misleading — implies the check is still useful when it is not
   - Suggestion: Replace with "The sanity check in `train_cnn.py` becomes unnecessary after the split (the condition it guards cannot occur in either dedicated script) and should be removed"

3. **ReduceLROnPlateau Scheduler Omitted** - Priority: Medium
   - Evidence: `train_cnn.py` lines 424-426 define scheduler with `patience=5, factor=0.5`; design config blocks omit it entirely
   - Impact: Readers implementing from the design don't know whether to keep or change the scheduler; affects convergence behavior, especially for epoch count decisions
   - Suggestion: Explicitly include `SCHEDULER: ReduceLROnPlateau(patience=5, factor=0.5)` in both stream configs or explain if it's being removed

4. **Subject03 Anomaly Unexplained** - Priority: Low
   - Evidence: Table shows subject03 "Good calib, other issues" with 50.6% accuracy; no elaboration given
   - Impact: Readers cannot understand whether subject03's data should be included in cross-subject training, filtered, or treated specially
   - Suggestion: Add one sentence explaining the suspected cause (e.g., fewer sessions, muscle anatomy outlier, data collection artifact)

---

## Score Summary

| Criterion | Score | Weight | Weighted |
|-----------|-------|--------|----------|
| Correctness (30%) | 3.5/5 | 0.30 | 1.050 |
| Design Quality (25%) | 4.0/5 | 0.25 | 1.000 |
| Model Optimization (25%) | 3.5/5 | 0.25 | 0.875 |
| DQ Analysis Quality (20%) | 3.5/5 | 0.20 | 0.700 |
| **Weighted Total** | | | **3.625/5.0** |

---

## Self-Verification

**Questions Asked**:
1. Does the design correctly describe the relationship between InstanceNorm and amplitude discrimination, as verified against the actual model code?
2. Is `AUG_AMPLITUDE_RANGE` a real config constant in `train_cnn.py`, or is the amplitude range hardcoded?
3. Is the claim that LOSO accuracy is "expected to be 60–75%" grounded in any evidence, or is it speculative?
4. Does the sanity check in `train_cnn.py` become unnecessary or merely relocated after the stream split?
5. Is the per-subject 100-epoch / cross-subject 80-epoch ordering internally consistent with the stated rationale?

**Answers**:
1. YES — `gesture_model_cnn.py` lines 131-139 confirm energy is captured before `self.input_norm(x)`. The InstanceNorm explanation is correct. `GestureCNN` uses BatchNorm1d (not InstanceNorm), correctly allowing amplitude to be discriminative.
2. NO — `train_cnn.py` line 259 hardcodes `np.random.uniform(0.5, 2.0)` inside `_aug_amplitude`. No `AUG_AMPLITUDE_RANGE` constant exists. This is a factual error in the design document.
3. SPECULATIVE — No empirical basis is provided. The 60–75% figure appears to be a rough estimate based on general EMG cross-subject literature expectations rather than preliminary experiments on this dataset. Section 9 doesn't address this as an open question.
4. UNNECESSARY, not relocated — After splitting into dedicated scripts, `USE_INSTANCE_NORM` and `PER_SUBJECT_MODELS` are no longer user-settable in the same script. The condition `USE_INSTANCE_NORM AND PER_SUBJECT_MODELS` becomes structurally impossible. The design's claim that it's "moved" to `train_per_subject.py` is wrong.
5. INCONSISTENT — The stated rationale for 100 epochs (per-subject) is "GestureCNN is smaller, trains faster per epoch." For cross-subject 80 epochs, the rationale is "GestureCNNv2 is larger; 80 epochs balances training time." But if the cross-subject model is larger and therefore slower per epoch, and if the cross-subject dataset is ~10× bigger, it needs MORE total training time to converge, not fewer epochs. The ordering (100 per-subject, 80 cross-subject) is backwards relative to typical convergence requirements for larger models on larger datasets. The stated rationale ("balances training time") is a wall-clock-time argument, not a convergence-quality argument.

**Adjustments Made**:
- Confirmed AUG_AMPLITUDE_RANGE error is real and material (not just cosmetic). Score for Correctness confirmed at 3.5 rather than raised.
- Confirmed the epoch-count ordering inconsistency — mentioned in Model Optimization analysis, confirmed as a real issue rather than a benefit of the doubt situation. Score remains 3.5.
- The LOSO 60–75% figure is speculative but disclosed as an expectation, not a result, which is acceptable for a design document. No score adjustment.
- Confirmed sanity check claim is wrong. Included in Correctness issues.

---

## Confidence Assessment

**Confidence Level**: High

**Confidence Factors**:
- Evidence strength: Strong (all claims verified against actual source code)
- Criterion clarity: Clear
- Edge cases: Handled (AUG_AMPLITUDE_RANGE error verified by reading actual function body)

---

## Key Strengths

1. **InstanceNorm Rationale is Technically Correct**: The explanation of why InstanceNorm helps cross-subject (strips amplitude variability) and hurts per-subject (amplitude is discriminative within a subject) is verified against the actual model architecture in `gesture_model_cnn.py`. The energy bypass scalar design is correctly described.

2. **MVC Calibration DQ is Actionable**: The MVC/neutral ratio table with per-subject accuracy correlation is the most original and useful analysis in the document. The tiered verdict (FLAG/WARN) provides clear guidance for data recollection prioritization.

3. **Stream Separation Eliminates Silent Misconfiguration**: The current `train_cnn.py` flag-pairing problem (identified in Section 1) is real and non-trivial. Splitting into dedicated scripts with fixed configs is the correct architectural solution.

---

## Areas for Improvement

1. **AUG_AMPLITUDE_RANGE Implementation Gap** - Priority: High
   - Evidence: Constant does not exist; `_aug_amplitude` hardcodes `np.random.uniform(0.5, 2.0)`
   - Impact: The primary differentiator between per-subject and cross-subject augmentation cannot be implemented without code refactoring
   - Suggestion: Add to migration plan step 3/4: "Refactor `_aug_amplitude` to accept configurable `scale_range` parameter"

2. **ReduceLROnPlateau Scheduler Missing from Specs** - Priority: Medium
   - Evidence: Active in `train_cnn.py` lines 424-426; absent from both stream config blocks
   - Impact: Implementers lack complete training configuration; epoch count choices are harder to justify without scheduler behavior
   - Suggestion: Include scheduler config in both stream constants blocks

3. **Subject03 Anomaly** - Priority: Low
   - Evidence: "Good calib, other issues" with 50.6% accuracy — no elaboration
   - Impact: Unclear if subject03 data is trustworthy for cross-subject training
   - Suggestion: One sentence identifying the suspected cause or flagging it as requiring investigation

---

## Actionable Improvements

**High Priority**:
- [ ] Refactor `_aug_amplitude` to accept `scale_range` parameter and expose `AUG_AMPLITUDE_RANGE` as a top-level constant in both training scripts
- [ ] Remove or correct the sanity check relocation claim; note the check becomes structurally unnecessary after the split

**Medium Priority**:
- [ ] Add `ReduceLROnPlateau` scheduler specification to both stream config blocks
- [ ] Clarify epoch ordering rationale: if 80 cross-subject epochs is a wall-clock-time decision, say so explicitly and note the quality tradeoff

**Low Priority**:
- [ ] Explain subject03's accuracy anomaly despite good calibration
- [ ] Add path migration note to the migration plan for existing model references
