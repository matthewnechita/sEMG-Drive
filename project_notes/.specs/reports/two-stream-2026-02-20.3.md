Done by Judge 3

---
VOTE: Pass
OVERALL_SCORE: 3.9/5.0
SCORES:
  Correctness (30%): 3.5/5.0
  Design Quality (25%): 4.5/5.0
  Model Optimization (25%): 4.0/5.0
  DQ Analysis Quality (20%): 3.5/5.0
---

# Evaluation Report

## Executive Summary

The design document is well-structured and demonstrates clear understanding of the two-stream problem. The repo split is clean, the normalization rationale is technically sound, and the DQ findings are mostly actionable. However, several technical correctness issues undermine confidence: the per-subject params-per-sample calculation uses incorrect numbers, the claim that cross-subject training "will outperform" the old global model is made without accounting for the LOSO vs. in-distribution accuracy difference, and the DQ analysis contradicts itself by calling subject05 data "technically usable" while also recommending exclusion without resolving the contradiction.

- **Artifact**: `C:\Users\matth\Desktop\capstone\capstone-emg\.specs\two-stream-design-2026-02-20.md`
- **Overall Score**: 3.90/5.00
- **Verdict**: GOOD
- **Threshold**: Not explicitly specified; judging against a 3.5/5.0 threshold implied by "Needs Revision" category
- **Result**: PASS

---

## Criterion Scores

| Criterion | Score | Weight | Weighted | Evidence Summary |
|-----------|-------|--------|----------|------------------|
| Correctness (30%) | 3.5/5 | 0.30 | 1.05 | Normalization rationale sound; params-per-sample arithmetic wrong; cross-subject accuracy claim overconfident |
| Design Quality (25%) | 4.5/5 | 0.25 | 1.125 | Repo structure clean and logical; shared components well-identified; migration plan clear |
| Model Optimization (25%) | 4.0/5 | 0.25 | 1.00 | Hyperparameter rationale mostly sound; aug range narrowing justified; LOSO threshold unvalidated |
| DQ Analysis Quality (20%) | 3.5/5 | 0.20 | 0.70 | MVC finding is real and actionable; subject05 contradiction unresolved; false positive analysis partially circular |

**Weighted Total: 3.875 → rounded to 3.9/5.0**

---

## Detailed Analysis

### Correctness (Weight: 0.30)

**Evidence Found:**

1. Per-subject params calculation: "120K params / ~11K–23K training windows = 5–11 params per sample — appropriate density."
   - A subject with 4 sessions at the minimum would yield approximately 4 × ~3K windows = ~12K windows (not 11K). With 8 sessions the upper bound of 23K is plausible. The "11K" lower bound deserves scrutiny: 4 sessions × 2K–3K windows per session = 8K–12K. The document's lower bound of 11K appears slightly optimistic for a 4-session subject, but is not grossly wrong. However, there is no accounting for the 20% held-out test split, which reduces training windows to 8.8K–18.4K. The ratio changes to 6.5–13.7 params/sample — still acceptable but the document does not acknowledge the split effect on this ratio.

2. Cross-subject accuracy claim: "the cross-subject model should outperform the old 82.7% global GestureCNN because: More training data than any per-subject model / InstanceNorm handles inter-subject amplitude variation..."
   - This is technically confused. The 82.7% figure is presumably an in-distribution test accuracy (train/test split from the same subjects). The design then lists "LOSO accuracy (true cross-subject metric) is expected to be 60–75%." So the document simultaneously claims the cross-subject model will "outperform 82.7%" and "achieve 60–75% LOSO." These two claims use different evaluation protocols and cannot both describe the same comparison. The document does not clarify that LOSO is the correct comparison for cross-subject deployment, making the "outperform" claim misleading.

3. InstanceNorm correctness: "InstanceNorm at input (per-window, strips amplitude scale)"
   - Verified against `gesture_model_cnn.py` lines 92–93: `self.input_norm = nn.InstanceNorm1d(in_channels, affine=False, track_running_stats=False)`. This is correct — per-channel, per-instance normalization. The energy bypass scalar (line 133: `energy = x.pow(2).mean(dim=(1, 2)).unsqueeze(1)`) is captured pre-normalization, consistent with the design doc's rationale.

4. z-score for per-subject: "Normalization: z-score per channel (mean/std computed from training data)"
   - Verified against `train_cnn.py` lines 378–387: when `USE_INSTANCE_NORM=False`, the code applies `standardize_per_channel`. This is correct and consistent.

5. Stream 2 LOSO: "Mandatory LOSO_EVAL = True — the cross-subject stream must be validated against held-out subjects."
   - Verified against `train_cnn.py` lines 738–739: `if LOSO_EVAL and not PER_SUBJECT_MODELS: loso_evaluate(...)`. This is already implemented in the existing code and the design correctly identifies it as a mandatory requirement.

6. Neutral class handling: The design states balanced active gestures at 14.5% each and neutral at 27.3%. With 5 active classes × 14.5% = 72.5%, neutral = 27.5%, this is consistent. The design correctly identifies there is no class imbalance problem for active gestures.

7. GestureCNN kernel_size: Stream 1 uses `KERNEL_SIZE = 7` which matches the original GestureCNN default in `gesture_model_cnn.py` line 14. Correct.

**Analysis:**

The core technical reasoning (InstanceNorm for cross-subject, amplitude preservation for per-subject, energy bypass for neutral detection) is grounded in EMG signal theory and verified against the actual code. The fatal weakness is the conflation of in-distribution accuracy (82.7%) with cross-subject LOSO accuracy (60–75%) in the "outperform" claim — this is a measurement protocol error that would mislead a reader comparing streams. The params-per-sample calculation is slightly imprecise (ignores test split) but not wrong in direction. These are medium-severity correctness issues, not fundamental failures.

**Score: 3.5/5**

**Improvement Suggestion:** Replace the claim "should outperform the old 82.7% global GestureCNN" with an explicit acknowledgment that: (a) the 82.7% is in-distribution accuracy and is not directly comparable to LOSO, and (b) the true measure of cross-subject improvement is LOSO, where the expected 60–75% represents an honest initial baseline.

---

### Design Quality (Weight: 0.25)

**Evidence Found:**

1. Repo structure (Section 3): Clear separation into `train_per_subject.py`, `train_cross_subject.py`, `models/per_subject/`, `models/cross_subject/`. The directory tree is explicit and shows all key files.

2. Shared components (Section 6): "gesture_model_cnn.py — GestureCNN, GestureCNNv2, CnnBundle, load_cnn_bundle(), quick_finetune() — no changes needed; both streams use it." This is accurate — the model file already uses an ARCHITECTURE_REGISTRY pattern (verified `gesture_model_cnn.py` lines 145–148) and the CnnBundle abstraction is already stream-agnostic.

3. The document explicitly calls out that data loading logic will be "duplicated across both scripts (acceptable; they have different enough config that sharing would add complexity with little benefit)." This is an honest acknowledgment of a design tradeoff rather than a gap.

4. Migration plan (Section 7) is concrete and ordered: 7 numbered steps covering directory creation, file migration, deprecation notice, and the first LOSO baseline run.

5. The inference script concern is addressed: "Update realtime_gesture_cnn.py default model path to models/cross_subject/gesture_cnn_v2.pt." Verified against `realtime_gesture_cnn.py` line 139: `DEFAULT_MODEL = os.path.join(..., "models", "gesture_cnn.pt")` — the current path is indeed stale and the migration step correctly identifies it.

6. The "What is NOT changed" section (Section 8) explicitly bounds scope. This is a useful design discipline that prevents scope creep.

7. Underscore prefix convention (`_dq_analysis.py`, `_inspect_models.py`) signals internal tooling vs. primary scripts. This is a minor but thoughtful naming convention.

**Analysis:**

The repo structure is clean, the separation between the two streams is unambiguous, and the shared component identification is verified against the actual code. The migration plan provides a reproducible sequence. The only mild criticism is that the document does not address whether `realtime_gesture_cnn.py` needs any changes beyond the default model path — specifically, whether the inference script should auto-detect which normalization mode to use (it does, via `bundle.standardize()` which already checks `use_instance_norm_input` in metadata). This is already handled in the existing code but the design document does not explicitly call it out, leaving a potential reader concern unresolved.

**Score: 4.5/5**

**Improvement Suggestion:** Add an explicit note in Section 6 that `realtime_gesture_cnn.py` already handles the normalization dispatch via `CnnBundle.standardize()` (checking `use_instance_norm_input` metadata), so no code changes are needed beyond the default path update.

---

### Model Optimization (Weight: 0.25)

**Evidence Found:**

1. Per-subject augmentation range narrowing: "AUG_AMPLITUDE_RANGE narrowed to (0.7, 1.4) — current (0.5, 2.0) is calibrated for cross-subject variance, which is 5–10x wider than within-subject variance."
   - The current code uses `np.random.uniform(0.5, 2.0)` in `_aug_amplitude` (train_cnn.py line 259). The design's reasoning that within-subject variance is narrower than cross-subject variance is correct in EMG literature. The specific 5–10× claim is asserted without measurement but is directionally plausible. The document honestly acknowledges this in Open Question #2: "actual within-subject EMG amplitude variance has not been measured from this dataset."

2. Dropout: Per-subject `DROPOUT = 0.2` vs. cross-subject `DROPOUT = 0.3`. The rationale is "smaller model needs less regularization." This is reasonable but incomplete — GestureCNN is smaller but per-subject training also has fewer samples, which argues for *more* regularization, not less. The document does not acknowledge this tension. With 8–18K training windows for a small subject and 120K parameters, dropout 0.2 may be under-regularized; the document's reasoning is one-sided.

3. Learning rate: Per-subject `LR = 3e-4` vs. cross-subject `LR = 1e-4`. "Slightly higher LR for smaller model" is consistent with established practice — smaller, simpler models benefit from larger learning steps. This is sound.

4. Epochs: Per-subject `EPOCHS = 100` vs. cross-subject `EPOCHS = 80`. The document states "GestureCNN is smaller, trains faster." With `ReduceLROnPlateau(patience=5)` in the scheduler (train_cnn.py line 424–426), 100 epochs provides more opportunities for the scheduler to reduce LR and converge fully. This is reasonable.

5. Balanced sampling: Per-subject `USE_BALANCED_SAMPLING = False` with rationale "no subject-imbalance problem." This is correct — within a single subject, the class label distribution is described as balanced (14.5% per active gesture). No need for subject-level balancing when there is only one subject.

6. LOSO threshold: Open Question #3 proposes "65% as minimum before using in real-time" for LOSO accuracy. This threshold is stated but not justified by reference to any baseline or user experience requirement. For a vehicle turn signal application, 65% gesture accuracy at inference time would produce frequent misclassifications. This is an unvalidated threshold.

7. Cross-subject `AUG_PROB = 0.5` vs. per-subject `AUG_PROB = 0.4`. The slightly lower augmentation probability for per-subject is sensible — less variance to simulate — though the difference is small.

**Analysis:**

The augmentation range narrowing is the strongest optimization insight and is properly qualified with acknowledgment of the measurement gap. The dropout reasoning is one-sided and does not address the opposite pressure (fewer samples per subject). The LOSO accuracy threshold (65%) is asserted without justification. The epoch/LR choices are conventional and well-reasoned. Overall this section is solid but has notable gaps in the dropout justification and the LOSO threshold rationale.

**Score: 4.0/5**

**Improvement Suggestion:** Address the dropout tension explicitly: "Per-subject has fewer samples but also a simpler decision boundary (one subject's physiology), so dropout 0.2 is chosen; if a subject has fewer than 10K training windows, consider increasing to 0.3." Also provide a brief justification for the 65% LOSO threshold (e.g., empirical comparison to random chance baseline of ~16.7% for 6 classes, or user testing minimum).

---

### DQ Analysis Quality (Weight: 0.20)

**Evidence Found:**

1. MVC/neutral ratio table (Section 2): Concrete per-subject ratios are provided with corresponding test accuracy figures. The correlation between weak MVC calibration and poor accuracy is clearly demonstrated. Subject05 (0.9–1.8× ratio) with 43.3% accuracy, subject01 (2.0–12.2× ratio) with 36.0% accuracy shows the correlation is imperfect — subject01 has higher ratios than subject05 but worse accuracy. The document does not address this inconsistency.

2. False positive analysis: "The DQ script flagged 16–17 'dead channels' per session (RMS < 1 µV threshold). This is wrong. The threshold of 1.0 is too high — filtered EMG data in this dataset has typical per-channel RMS of 0.2–0.8 (in stored units)."
   - This is an important finding but the argument is somewhat circular: the document asserts the RMS range is 0.2–0.8 "in stored units" without showing how this was determined. If the stored units are post-MVC-normalization, then 0.2–0.8 is a claim that needs grounding. More importantly, the conclusion "No channels are dead" is stated definitively but rests on an asserted typical range that is not supported by a measurement. The analysis correctly identifies the threshold is wrong, but the proof is weak.

3. Subject05 contradiction: The document states "Data is technically usable but MVC-scaled normalization is unreliable" (Section 2, MVC calibration quality) and then Open Question #1 states "Recommend: exclude subject05 from cross-subject training until recalibrated."
   - These are contradictory guidance positions. If data is "technically usable," why is exclusion recommended? If exclusion is recommended, why is it in Open Questions rather than in the DQ Verdict section? The DQ Verdict section says only "FLAG subject05" — it does not resolve whether to include or exclude. A design document making a training recommendation cannot leave this as an open question without practical impact on what `train_cross_subject.py` should actually do.

4. Subject01 accuracy anomaly: Subject01 has MVC ratios of 2.0–12.2× (not the worst) but achieves only 36.0% accuracy — worse than subject05's 43.3% despite better calibration. The document does not analyze this discrepancy. This is a missed DQ finding: subject01 may have a different issue (electrode placement, gesture execution inconsistency, or label quality) that is unaddressed.

5. Actionable recommendations: The document recommends "instruct subjects to contract 'as hard as possible' and verify the MVC signal exceeds the neutral RMS by at least 5× before accepting the session." This is specific and actionable.

6. Overall PASS verdict with appropriate caveats is reasonable given no corrupt files, consistent channels, and balanced labels.

**Analysis:**

The MVC calibration finding is the document's strongest DQ contribution and provides a clear mechanistic link between calibration quality and model accuracy. The false positive analysis identifies a real problem (over-aggressive RMS threshold) but the counter-argument is asserted rather than proven. The subject05 contradiction between "technically usable" and "recommend exclusion" is unresolved and leaves the implementation team without clear guidance. Subject01's anomaly (worst accuracy despite not-worst calibration) is completely overlooked.

**Score: 3.5/5**

**Improvement Suggestion:** Resolve the subject05 inclusion/exclusion decision explicitly in the DQ Verdict section (not as an open question), and add analysis of subject01's anomalous accuracy gap to distinguish calibration effects from other DQ factors.

---

## Self-Verification

**Questions Asked:**

1. Does the design correctly identify which architecture (GestureCNN vs. GestureCNNv2) maps to each stream, and is this verified against the actual code?

2. Does the "outperform 82.7%" claim hold up when the evaluation protocol difference (in-distribution vs. LOSO) is considered?

3. Is the claim that InstanceNorm "strips amplitude scale" accurate given how InstanceNorm operates (per-channel, per-instance normalization) and the actual implementation in gesture_model_cnn.py?

4. Does the document resolve what to do about subject05 data in the cross-subject training set?

5. Is the dropout=0.2 recommendation for per-subject consistent with the data volume available for small subjects?

**Answers:**

1. YES. Stream 1 uses GestureCNN (no InstanceNorm), Stream 2 uses GestureCNNv2 (with InstanceNorm). Verified: `gesture_model_cnn.py` lines 13–34 (GestureCNN) and 79–140 (GestureCNNv2). The architecture selection in `train_cnn.py` lines 357–370 confirms `USE_INSTANCE_NORM=True` selects GestureCNNv2. This is correct.

2. NO. The 82.7% figure is an in-distribution test accuracy (from the existing train/test split on combined subjects). The document's own LOSO projection of 60–75% is the correct cross-subject comparison. The "outperform" claim is misleading because it does not specify what evaluation protocol the 82.7% figure used. This was correctly flagged in the Correctness section.

3. PARTIALLY. InstanceNorm1d with `affine=False` normalizes each channel independently per window to zero mean and unit variance — it does strip amplitude scale per channel per window. This is correct. However, it also removes intra-channel variation within a window (the shape is preserved but amplitude is normalized). The document correctly identifies the energy bypass scalar as the mechanism to preserve neutral detection. The InstanceNorm claim is technically sound.

4. NO. The document explicitly places this as Open Question #1 rather than a firm decision. The DQ Verdict says "FLAG subject05" but does not update the training configuration to actually exclude this subject. The design is incomplete on this point.

5. PARTIALLY. With 4–8 sessions × ~3K windows per session = 12K–24K training windows, minus 20% test split = 9.6K–19.2K training windows, dropout=0.2 with 120K parameters gives 0.8–1.6 params per training window effective capacity. This is tighter than the document's stated 5–11 params per sample figure (which appears to use total windows, not training windows after the split). The dropout concern is valid but not severe enough to overturn the recommendation — it is merely unacknowledged.

**Adjustments Made:** No adjustments to scores. The verification confirmed the correctness and DQ issues identified in the initial analysis. The InstanceNorm analysis (Q3) was verified as correct, which provides confidence in those aspects. The subject05 unresolved status (Q4) confirmed the DQ deduction.

---

## Confidence Assessment

**Confidence Level**: High

**Confidence Factors:**
- Evidence strength: Strong — all key claims were verified against actual source code files
- Criterion clarity: Clear — four distinct criteria with unambiguous descriptions
- Edge cases: Some uncertainty — the per-subject training window count calculation depends on session data that was not directly accessible; estimates are based on the document's stated totals

---

## Key Strengths

1. **InstanceNorm rationale is well-grounded**: The explanation of why InstanceNorm is appropriate for cross-subject (strips inter-subject amplitude) and inappropriate for per-subject (amplitude is discriminative within a subject) is technically correct and verified against the actual GestureCNNv2 implementation including the energy bypass scalar.

2. **Repo separation is clean and actionable**: The two-stream file structure directly addresses the current `train_cnn.py` boolean flag problem. The document names specific files, specific directories, and a concrete 7-step migration sequence. No ambiguity about what to build.

3. **MVC calibration finding is the key DQ insight**: The correlation between MVC/neutral ratio and test accuracy across 7 subjects is clearly presented in a table with specific numbers, and the protocol improvement recommendation (verify 5× ratio before accepting a session) is specific and immediately actionable.

---

## Areas for Improvement

1. **Conflated evaluation protocols** - Priority: High
   - Evidence: "Should outperform the old 82.7% global GestureCNN" followed by "LOSO accuracy...expected to be 60–75%"
   - Impact: A reader comparing the two streams will be misled about the cross-subject model's true performance floor
   - Suggestion: Specify explicitly that 82.7% used in-distribution evaluation and is not comparable to LOSO; reframe the 60–75% LOSO as the honest cross-subject baseline

2. **Subject05 inclusion decision left unresolved** - Priority: High
   - Evidence: DQ Verdict says "FLAG subject05" but Open Question #1 says "Recommend: exclude...until recalibrated" — these are two different guidance levels and neither updates the actual training config
   - Impact: Implementation team will include or exclude subject05 arbitrarily; the cross-subject model's training set is undefined
   - Suggestion: Promote the exclusion recommendation to the DQ Verdict section and explicitly state that `train_cross_subject.py` should filter out subject05 (e.g., via an `EXCLUDED_SUBJECTS` config list)

3. **Subject01 accuracy anomaly not investigated** - Priority: Medium
   - Evidence: Subject01 achieves 36.0% accuracy with 2.0–12.2× MVC ratios, which is worse than subject05 (43.3%) despite better calibration ratios
   - Impact: A real DQ issue may be masked behind the MVC calibration narrative; subject01's data may have label quality or electrode placement issues that would corrupt cross-subject training
   - Suggestion: Add a brief investigation of subject01's per-session accuracy variance and per-gesture confusion to distinguish calibration effects from other sources of error

---

## Actionable Improvements

**High Priority**:
- [ ] Correct the "outperform 82.7%" claim to explicitly compare LOSO-to-LOSO rather than LOSO to in-distribution accuracy
- [ ] Resolve subject05 inclusion/exclusion with a firm decision in the DQ Verdict and a corresponding `EXCLUDED_SUBJECTS` config entry in `train_cross_subject.py`

**Medium Priority**:
- [ ] Acknowledge the dropout tension for low-session subjects and provide a conditional recommendation (e.g., increase dropout to 0.3 if training windows < 10K)
- [ ] Justify the 65% LOSO threshold with reference to the 6-class random baseline (~16.7%) and any user experience minimum
- [ ] Investigate subject01's anomalous accuracy gap

**Low Priority**:
- [ ] Add the false-positive RMS analysis reasoning with a measured example (show actual per-channel RMS values from one session) rather than asserting "typical RMS is 0.2–0.8"
- [ ] Explicitly note in Section 6 that `realtime_gesture_cnn.py` already handles normalization dispatch via `CnnBundle.standardize()`, requiring no code change beyond the default path update
