# Evaluation Report: CNN Cross-Subject Generalization Improvement Plans

## Executive Summary

Three plans were evaluated for achieving cross-subject EMG gesture recognition generalization. All three correctly identify the core problems (global z-score normalization, evaluation splitting by file not subject, lack of augmentation) and propose similar high-level solutions. Solution A provides the strongest combination of technical precision, honest self-critique, and practical architecture design. Solution B is thorough and methodical but introduces some questionable recommendations (replacing BatchNorm with InstanceNorm throughout the network, and both per-window z-score AND InstanceNorm simultaneously). Solution C is the most comprehensive in scope (8 changes, including prototypical networks and session-level adversarial domains) but spreads effort thin and has notable technical gaps in its normalization strategy.

- **Artifact**: `.specs/cnn-improvement-plan.a.md`, `.specs/cnn-improvement-plan.b.md`, `.specs/cnn-improvement-plan.c.md`
- **Overall Scores**: A: 3.76, B: 3.30, C: 3.39
- **Verdict**: Solution A: GOOD | Solution B: ADEQUATE | Solution C: ADEQUATE

---

## Criterion Scores

| Criterion | Weight | Sol A | Sol B | Sol C |
|-----------|--------|-------|-------|-------|
| Technical Soundness | 0.30 | 4.0 | 3.0 | 3.5 |
| Cross-Subject Impact | 0.25 | 4.0 | 3.5 | 3.5 |
| Implementation Feasibility | 0.20 | 3.5 | 3.5 | 3.0 |
| Completeness | 0.15 | 3.5 | 3.0 | 3.5 |
| Clarity & Actionability | 0.10 | 3.0 | 3.5 | 3.0 |

**Weighted Totals:**
- **Solution A:** (4.0 x 0.30) + (4.0 x 0.25) + (3.5 x 0.20) + (3.5 x 0.15) + (3.0 x 0.10) = 1.20 + 1.00 + 0.70 + 0.525 + 0.30 = **3.73**
- **Solution B:** (3.0 x 0.30) + (3.5 x 0.25) + (3.5 x 0.20) + (3.0 x 0.15) + (3.5 x 0.10) = 0.90 + 0.875 + 0.70 + 0.45 + 0.35 = **3.28**
- **Solution C:** (3.5 x 0.30) + (3.5 x 0.25) + (3.0 x 0.20) + (3.5 x 0.15) + (3.0 x 0.10) = 1.05 + 0.875 + 0.60 + 0.525 + 0.30 = **3.35**

---

## Detailed Analysis

### Technical Soundness (Weight: 0.30)

#### Solution A

**Evidence Found:**
- Root cause table (Section 1.1) correctly identifies four factors: amplitude scaling, spatial pattern shift, temporal dynamics, baseline drift. Each maps to a specific codebase deficiency.
- Correctly identifies the `GroupShuffleSplit` bug: "splits by file path, not by subject" (Section 1.2). Verified against `train_cnn.py` line 318-321: `splitter = GroupShuffleSplit(...)` uses `groups` which are file paths (line 168).
- Instance normalization rationale (Change 2) is technically sound: `nn.InstanceNorm1d(channels[0], affine=False, track_running_stats=False)` applied at input only. The distinction from BatchNorm is correctly explained: "BatchNorm computes running statistics across the training set, making it subject-dependent."
- The signal energy bypass (Q2/Revision 1) for neutral gesture under instance norm is a genuine insight. The solution acknowledges: "Instance normalization on a near-zero signal amplifies noise to unit variance, potentially making neutral windows look like active gestures."
- The `GestureCNNv2` architecture correctly uses residual connections with channel attention (SE-blocks). The parameter count estimate of ~180K is plausible.
- The subject-adversarial section honestly notes: "With 7 this is borderline" and correctly downgrades it from HIGH to MEDIUM impact in revisions.
- The default kernel_size in the proposed `GestureCNNv2` is 7, which matches `gesture_model_cnn.py` defaults (line 12: `kernel_size=7`) but differs from the training config (`KERNEL_SIZE=11`). This is not flagged as a deliberate change or justified.

**Analysis:** The ML reasoning is sound throughout. The instance norm approach is well-justified, and the signal energy bypass shows genuine domain understanding. The adversarial training caveats are honest. The residual architecture is reasonable. Minor issue: doesn't explicitly address that training uses kernel_size=11 while proposing 7.

**Score:** 4.0/5

**Improvement:** Justify the kernel_size change from 11 (current training config) to 7, or explicitly recommend keeping 11.

#### Solution B

**Evidence Found:**
- Root cause analysis identifies the same core issues as A. Adds the observation about BatchNorm: "BatchNorm locks subject-specific statistics" (Section 1.2, row 5). This is a valid point.
- Proposes BOTH per-window z-score AND replacing BatchNorm with InstanceNorm throughout: "Use BOTH. Apply per-window z-score at the input (Option A) AND replace BatchNorm with InstanceNorm in the convolutional blocks (Option B)." This is technically questionable. InstanceNorm with `affine=True` throughout the network (as proposed) still has learnable scale/shift parameters that could re-introduce subject-specific patterns. The combination of per-window z-score AND InstanceNorm throughout is redundant -- the first InstanceNorm layer already normalizes per-instance; subsequent InstanceNorm layers normalize the feature maps, which is a different operation than input normalization.
- The `GestureCNNWithDANN` architecture (Change 4) includes `return_subject_logits` flag for inference, but the architecture has per-window normalization baked into `forward()`. This is correct.
- The L2 normalization alternative (Revision A, Q2) is an interesting idea but loses per-channel zero-centering, which may hurt if there are consistent channel-level DC offsets after MVC calibration.
- Solution B mentions `gesture_model_cnn.py` line 18 has `nn.BatchNorm1d(out_ch)`. The actual line 18 is `blocks.append(nn.BatchNorm1d(out_ch))`. This is correct content but the line number is line 18, which is actually correct.
- The silence gate (Revision B) uses a fixed threshold `SILENCE_RMS_THRESHOLD = 0.01` but doesn't address how to set this across subjects with different baseline amplitudes -- this is the exact problem being solved.

**Analysis:** The dual normalization strategy (per-window z-score + InstanceNorm throughout) is overcomplicated and not well-justified. InstanceNorm throughout the network is a more aggressive change than needed -- the issue is at the INPUT normalization, not the intermediate feature normalization. BatchNorm in intermediate layers is fine because it normalizes feature maps, not raw subject-specific signals. The adversarial training section is solid. The L2 normalization alternative shows creative thinking but has flaws.

**Score:** 3.0/5

**Improvement:** Separate input normalization (where InstanceNorm genuinely helps) from intermediate normalization (where BatchNorm is appropriate and well-studied). The recommendation to replace all BatchNorm with InstanceNorm is not well-supported.

#### Solution C

**Evidence Found:**
- Root cause analysis (Section 1) is clear and accurate. The "subject-specific CNN" summary is concise: "The model learns to classify (subject_identity + gesture) rather than gesture alone."
- The normalization approach (Change 2) diverges from A and B: instead of per-window instance normalization, it proposes MVC-only normalization with clip-and-scale: `emg_norm = np.clip(emg_norm, -5.0, 5.0) / 5.0`. This is a simpler approach but is technically weaker because it DEPENDS on MVC calibration data being accurate and present. Solution C itself acknowledges this risk in Q1: "I did not verify this for all 42 files."
- The adversarial training (Change 4) uses BatchNorm, not InstanceNorm, in the adversarial architecture. This is inconsistent with the goal of cross-subject generalization. The solution also uses `BatchNorm1d` in the `GestureCNNv2` architecture (Change 5). Neither architecture incorporates per-window normalization.
- The session-level adversarial domain idea (Revision 3, from Q3) is creative: using ~35 session IDs instead of 7 subject IDs as domain labels. However, this conflates within-subject session variation (electrode re-placement, different days) with cross-subject variation, which are different kinds of distribution shift. It's not clear this would produce better subject-invariant features.
- The prototypical network (Change 8) is a significant addition not present in A or B. The implementation is technically correct but adds substantial complexity.
- The `electrode_shift_augmentation` (Change 6) proposes circular channel shifting, then Q2 correctly identifies this as invalid for Delsys Trigno sensors. This self-correction is good.
- The normalization approach lacks any mechanism to handle the instance normalization problem with neutral/silent windows.

**Analysis:** The MVC-only normalization is a weaker approach than instance normalization because it depends entirely on calibration data quality. The GestureCNNv2 architecture uses BatchNorm throughout, which is inconsistent with cross-subject goals. The session-level adversarial idea is creative but theoretically questionable. The prototypical network fallback adds value but is complex.

**Score:** 3.5/5

**Improvement:** The normalization strategy (Change 2) should include per-window normalization as a complement to MVC calibration, not rely solely on calibration data. Add InstanceNorm or per-window z-score to the architecture.

---

### Cross-Subject Impact (Weight: 0.25)

#### Solution A

**Evidence Found:**
- Instance normalization at input removes the dominant source of cross-subject variance (amplitude differences) while preserving temporal shape. The signal energy bypass addresses the neutral-gesture failure mode.
- Data augmentation with channel amplitude scaling (0.5-2.0), temporal shift, channel dropout, and temporal stretch directly simulates inter-subject variations.
- The estimated LOSO improvement table is conservative and realistic: Instance Norm +15-25%, Augmentation +5-10%, Architecture +3-7%.
- Final estimate of 60-75% LOSO (from 30-50% baseline) is realistic for 7 subjects with 6 classes. The admission that ">70% LOSO would make the cross-subject model viable" sets an honest bar.
- The fallback suggestion of few-shot calibration (30 seconds + fine-tune head only) is practical.

**Analysis:** The combination of instance normalization + augmentation directly addresses the three main sources of cross-subject variance (amplitude, timing, spatial). The impact estimates are grounded and not overly optimistic. The plan correctly prioritizes the highest-impact changes first.

**Score:** 4.0/5

**Improvement:** Consider whether the 7-subject dataset is sufficient for the adversarial training to contribute meaningful improvement, and provide clearer criteria for when to abandon it.

#### Solution B

**Evidence Found:**
- Per-window normalization + InstanceNorm throughout is more aggressive than A's approach and should remove more subject-specific information. However, this risks removing gesture-discriminative information as well.
- The L2 normalization alternative (preserving inter-channel ratios) is a creative approach to the information loss problem.
- Adversarial training is presented at MEDIUM-HIGH impact with lambda schedule and combined loss. The 0.3 weight coefficient is reasonable.
- The silence gate (Revision B) addresses neutral gesture detection under normalization, but with a fixed threshold that may not generalize.
- The quick_finetune function (Change 7) provides a practical fallback: freeze features, fine-tune head only, 20 epochs on calibration data.
- Expected outcomes table shows cumulative estimates from ~25-40% to ~75-88% with calibration. The 75-88% with 70-second calibration seems optimistic.

**Analysis:** The normalization approach is aggressive and may over-normalize. The dual normalization strategy (z-score + InstanceNorm everywhere) could strip too much signal. The quick_finetune fallback is a solid addition. The expected outcomes with calibration (75-88%) may be overly optimistic for 7 subjects.

**Score:** 3.5/5

**Improvement:** Provide empirical justification for the claim that InstanceNorm throughout (not just at input) helps cross-subject generalization. Literature support would strengthen this claim.

#### Solution C

**Evidence Found:**
- MVC-only normalization (Change 2) is the weakest normalization strategy of the three. It relies entirely on calibration data being present and accurate. The clip-and-scale (`np.clip(emg_norm, -5.0, 5.0) / 5.0`) is a crude transformation that maps everything to [-1, 1] but doesn't normalize the distribution shape.
- The calibration data validation step (Revision 1, Q1) is a practical addition that A and B lack.
- Adversarial training with session-level domains (Revision 3) is creative but may conflate within-subject and cross-subject variance.
- Subject-balanced batch sampling (Change 7) ensures each batch represents all subjects, which helps gradient quality.
- Prototypical network (Change 8) provides a principled few-shot adaptation mechanism.
- Target thresholds (Section 6) are the most conservative: LOSO mean 35% -> 60% target -> 75% stretch.
- The `GestureCNNv2` architecture uses BatchNorm, not InstanceNorm. This means the model still accumulates training-subject statistics in its running averages. At inference on a new subject, these statistics create mismatch.

**Analysis:** The normalization strategy is the weakest of the three solutions. MVC-only normalization addresses amplitude scale but not distribution shape. The BatchNorm architecture means intermediate features are still normalized using training-subject statistics. The subject-balanced sampler and prototypical network add value but don't compensate for the fundamental normalization weakness.

**Score:** 3.5/5

**Improvement:** Replace BatchNorm with InstanceNorm (at least at input) in the proposed architectures. MVC calibration is a good first step, but the model needs self-contained normalization to handle new subjects whose MVC calibration may be imperfect.

---

### Implementation Feasibility (Weight: 0.20)

#### Solution A

**Evidence Found:**
- Code snippets are complete and syntactically correct for all 7 changes.
- The `GestureCNN` modification (Change 2) adds `use_instance_norm_input` parameter while preserving the existing constructor signature. Backward compatible.
- The `EMGAugmentor` class (Change 3) operates on `(C, T)` tensors and integrates via `AugmentedEMGDataset` -- clean PyTorch Dataset pattern.
- The `GestureCNNv2` (Change 4) uses standard PyTorch modules: `nn.Conv1d`, `nn.BatchNorm1d`, `nn.AdaptiveAvgPool1d`. Residual connections are straightforward.
- Bundle format update (Section 5) preserves backward compatibility: old bundles without `use_instance_norm_input` default to `False`.
- The `_resolve_architecture` update (Q5) correctly extends the existing function with a new branch for `GestureCNNv2`. However, the return types differ between the two branches (`arch_type, in_channels, num_classes, arch` vs `arch_type, channels, dropout, kernel_size`), which would require changes to `load_cnn_bundle` as well. This inconsistency is noted but not fully resolved in the code.
- The `CnnBundle.standardize()` update (Section 4) is minimal and correct.

**Analysis:** The code snippets are well-structured and would integrate with the existing codebase. The backward compatibility approach is sound. The `_resolve_architecture` return type inconsistency is a minor implementation detail that would need resolution but doesn't block the plan.

**Score:** 3.5/5

**Improvement:** Provide a complete `_resolve_architecture` and `load_cnn_bundle` implementation that handles both architecture types with consistent return types.

#### Solution B

**Evidence Found:**
- LOSO implementation (Change 1) includes DataLoader usage for batch processing of test data, which is more memory-efficient than A's approach of loading all test data into GPU at once.
- The `GestureCNNWithDANN` (Change 4) is a complete architecture with `return_subject_logits` flag. The deployment strategy (extract features + gesture head into standard GestureCNN) is practical but the code snippet for weight extraction is fragile: `deploy_model.head[2].weight.data = adv_model.gesture_head.weight.data` assumes a specific head structure.
- The `quick_finetune` function (Change 7) is complete and correct: freezes `model.features`, trains `model.head`, unfreezes after.
- Per-window normalization in `forward()` (lines 152-154 of Change 2) adds `mean`/`std` computation at inference time. This is negligible overhead for a (1, C, 200) tensor.
- The recommended `norm_mode` field in architecture dict (Revision A) adds configuration complexity but provides flexibility.
- Code uses numpy augmentation (`np.random`) in `EMGAugment`, not PyTorch. This means augmentation results won't be GPU-accelerated, but for a CPU-bound training pipeline this is fine.

**Analysis:** Implementation is feasible. The DataLoader-based test evaluation and the quick_finetune function are practical additions. The weight extraction code for DANN deployment is fragile. The numpy-based augmentation is functional but slightly inconsistent with the PyTorch-based model.

**Score:** 3.5/5

**Improvement:** The DANN weight extraction code should use state_dict mapping rather than index-based access to head layers.

#### Solution C

**Evidence Found:**
- The `EMGAugmenter` (Change 3) uses numpy, which is fine but means augmentation happens on CPU before tensor conversion. The `AugmentedEMGDataset` stores data as numpy and converts in `__getitem__`, which works but creates a numpy copy per access.
- The `SubjectBalancedSampler` (Change 7) is a complex piece of code with pointer management and re-shuffling logic. The `__iter__` method yields individual indices rather than batches, which works with DataLoader but the semantics are subtle.
- The `ProtoNet` (Change 8) is a clean implementation. However, integrating it with the realtime script would require significant changes: the prototype computation, storage, and nearest-prototype classification are all new inference-time operations not present in the current `CnnBundle` abstraction.
- The `GestureCNNAdversarial` (Change 4) uses `BatchNorm1d` in the feature extractor, which contradicts the cross-subject goal. The deployment strategy (extract features + gesture_head as standard GestureCNN) is the same as B's.
- The architecture registry (Revision 4) is a clean pattern: `ARCHITECTURE_REGISTRY = {"GestureCNN": GestureCNN, "GestureCNNv2": GestureCNNv2}`.
- The calibration data validation step (Revision 1) is a practical addition but the fallback (per-file z-score) creates mixed normalization within the same training set, which could confuse the model.
- The `electrode_shift_augmentation` is initially proposed then correctly retracted for Delsys sensors.
- Total estimated effort of 12-16 days is realistic but ambitious for Phase 5 (prototypical network) integration.

**Analysis:** The implementations are more complex than A or B, with additional components (SubjectBalancedSampler, ProtoNet, calibration validation) that increase integration effort. The SubjectBalancedSampler is non-trivial code that could have edge cases. The ProtoNet would require significant realtime script changes not fully specified. The mixed normalization fallback is a practical but imperfect solution.

**Score:** 3.0/5

**Improvement:** Simplify the SubjectBalancedSampler using PyTorch's WeightedRandomSampler with subject-based weights. Provide a complete realtime script integration plan for the ProtoNet path.

---

### Completeness (Weight: 0.15)

#### Solution A

**Evidence Found:**
- Training pipeline: LOSO evaluation, augmented dataset, training loop modification for DANN, normalization bypass. All covered.
- Evaluation methodology: LOSO with per-subject and per-gesture reporting. Covered.
- Architecture changes: GestureCNN modification (instance norm), GestureCNNv2 with residual + attention. Covered.
- Real-time inference: Section 4 details minimal changes to `realtime_gesture_cnn.py`. The updated `CnnBundle.standardize()` is provided.
- Bundle format: Section 5 provides complete updated bundle dictionary.
- Missing: No subject-balanced batch sampling. No prototypical network fallback. No calibration data validation step.

**Analysis:** Covers the four required areas (training, evaluation, architecture, real-time) adequately. Missing some secondary components that C covers (balanced sampling, prototypical network, calibration validation).

**Score:** 3.5/5

**Improvement:** Add a calibration data validation step before training, and consider subject-balanced batch sampling.

#### Solution B

**Evidence Found:**
- Training pipeline: LOSO, augmented dataset, DANN training loop, normalization bypass. Covered.
- Evaluation methodology: LOSO with per-subject reporting and classification_report. Covered.
- Architecture changes: GestureCNN with instance norm + per-window z-score, GestureCNNWithDANN. Covered.
- Real-time inference: Mentions that `bundle.standardize()` becomes a no-op but does not provide updated realtime code. The silence gate is mentioned but not integrated into the realtime script.
- Bundle format: Described but not provided as a complete dictionary.
- Quick_finetune: A complete function for head-only fine-tuning.
- Missing: No detailed bundle format specification. The realtime integration is underspecified -- "requires zero changes" is stated but not demonstrated.
- The L2 normalization alternative adds a configuration dimension (norm_mode) but doesn't fully specify how to A/B test it.

**Analysis:** Covers training and evaluation well. Architecture changes are thorough (arguably over-thorough with dual normalization). Real-time inference changes are underspecified. The quick_finetune function adds completeness for the adaptation path.

**Score:** 3.0/5

**Improvement:** Provide a complete bundle format specification and demonstrate the realtime inference path with the new normalization.

#### Solution C

**Evidence Found:**
- Training pipeline: LOSO, augmented dataset, DANN training, balanced sampler, MVC normalization. Covered extensively.
- Evaluation methodology: Section 6 provides primary metric (LOSO), secondary metrics (F1, confusion matrix, latency), and target thresholds table. This is the most complete evaluation methodology.
- Architecture changes: GestureCNNAdversarial, GestureCNNv2, ProtoNet. Three architectures provided.
- Real-time inference: Section 5 describes the "New Subject Experience" for both zero-shot and few-shot scenarios. However, the realtime code changes are thinly specified.
- Bundle format: Architecture registry (Revision 4) addresses loading but doesn't specify the full bundle dictionary.
- Files to modify: Section 9 lists all files and changes -- useful reference.
- Calibration data validation: Revision 1 adds a pre-training check.

**Analysis:** The most complete plan in terms of scope -- covers evaluation methodology with target thresholds, multiple architecture options, deployment scenarios, and calibration validation. However, breadth comes at the cost of depth in some areas (realtime integration, bundle format).

**Score:** 3.5/5

**Improvement:** Provide a complete bundle format specification for both GestureCNNv2 and ProtoNet deployment paths.

---

### Clarity & Actionability (Weight: 0.10)

#### Solution A

**Evidence Found:**
- Implementation roadmap (Section 3) has 5 phases with specific tasks per phase. Phases are ordered logically (evaluation -> normalization -> architecture -> adversarial -> deployment).
- Verification questions (Section 6) are self-asked and answered with concrete mitigations and code snippets.
- Revisions (Section 7) explicitly state what changed and why.
- The expected outcomes table (Section 8) provides per-change estimates.
- However, the plan doesn't specify which files to modify (no equivalent to C's Section 9). A developer would need to infer this.
- The weekly timeline (Weeks 1-4) is ambitious but provides scheduling context.

**Analysis:** Well-structured with clear progression. The verification Q&A format is effective. Missing a "files to modify" summary that C provides.

**Score:** 3.0/5

**Improvement:** Add an explicit "files to modify" section listing each file and the specific changes needed.

#### Solution B

**Evidence Found:**
- Implementation roadmap (Section 3) has 6 phases with numbered tasks. Each phase specifies which changes to implement and includes a "Re-run LOSO" checkpoint.
- Section 4 provides a modified `main()` function showing how LOSO integrates with the existing flow. The distinction between LOSO for evaluation and final model training on all subjects is explicitly called out: "LOSO is for EVALUATION. The final deployable model trains on ALL 7 subjects."
- Verification questions and revisions are thorough.
- Section 9 (Files to Modify) is a clear table mapping files to changes.
- The two normalization options (z-score vs L2) with a `norm_mode` config adds decision points a developer would need to navigate.

**Analysis:** The clearest actionable structure of the three. The modified `main()` function and the files-to-modify table are particularly useful. The dual normalization option adds a decision point that may confuse implementation.

**Score:** 3.5/5

**Improvement:** Recommend a default normalization mode rather than leaving it as a decision point.

#### Solution C

**Evidence Found:**
- Implementation roadmap (Section 4) has 6 phases with time estimates (days). The progression is logical.
- Section 5 describes deployment scenarios concretely (zero-shot vs few-shot).
- Section 6 provides target thresholds in a table format with baseline, target, and stretch goals.
- Section 9 lists files to modify with changes.
- However, the plan has 8 changes, making it complex to follow. The conditional nature of some changes (Change 6: conditional, Change 8: optional) adds ambiguity.
- The session-level adversarial domain change (Revision 3) is introduced late in the document and changes the semantics of Change 4 significantly.

**Analysis:** The plan is comprehensive but complex. Eight changes with conditional/optional flags and late revisions make it harder for a developer to follow a clear path. The target thresholds table is a useful addition.

**Score:** 3.0/5

**Improvement:** Consolidate the plan into a clear "default path" (Changes 1-5) and "optional extensions" (Changes 6-8) to reduce cognitive load.

---

## Self-Verification

### Questions Asked:

1. **Am I penalizing Solution B unfairly for the dual normalization recommendation?** Could per-window z-score + InstanceNorm throughout actually be beneficial?

2. **Am I giving Solution A too much credit for the signal energy bypass?** Is this a well-known technique or a genuine insight?

3. **Is Solution C's MVC-only normalization actually weaker than instance normalization, or could it be equally effective if calibration data is reliable?**

4. **Am I being biased by Solution A's length and structure?** Is there a length bias affecting my scoring?

5. **Does Solution C's session-level adversarial idea have merit that I'm underweighting?**

### Answers:

1. **Re-examination:** Per-window z-score at input + InstanceNorm in intermediate layers is not necessarily redundant. The input z-score normalizes raw EMG amplitude; InstanceNorm in feature maps normalizes learned feature distributions per-sample. However, Solution B recommends InstanceNorm with `affine=True`, which adds learnable scale/shift parameters. These parameters are trained on the training subjects' feature distributions. While InstanceNorm is better than BatchNorm for cross-subject work (no running statistics), the combination with per-window z-score is not standard in EMG literature. I maintain the 3.0 score -- the recommendation is overcomplicated and not well-justified relative to A's simpler approach.

2. **Re-examination:** The signal energy bypass (computing RMS before instance normalization and concatenating to features) is a known technique in speech processing (energy features) but less commonly applied in EMG. It addresses a real failure mode. However, it's not a breakthrough insight -- it's a reasonable engineering solution. I maintain the score but note it's not unique to Solution A's author.

3. **Re-examination:** MVC-only normalization is genuinely weaker because: (a) it depends on calibration data being present and accurate for ALL sessions, (b) MVC calibration doesn't normalize distribution shape (skewness, kurtosis), only scale, (c) the clip-and-scale to [-1, 1] is a lossy transformation that discards information about signal magnitude above the threshold. Instance normalization handles all of these automatically without requiring external calibration data. However, MVC normalization is physiologically meaningful (mapping to % of maximum contraction), which instance norm is not. I maintain 3.5 for C's cross-subject impact -- the approach works but is less robust.

4. **Length check:** Solution A is ~638 lines, Solution B is ~671 lines, Solution C is ~672 lines. All three are comparable in length. I don't see evidence of length bias in my scoring. The differences are in content quality, not length.

5. **Re-examination:** Session-level adversarial domains (~35 domains instead of 7) do have a theoretical advantage: more domains make the adversarial task harder, potentially producing more general invariance. However, sessions from the SAME subject share subject-specific characteristics, so the adversary would learn to be invariant to session-level variation (electrode re-placement, different days) but not necessarily to cross-subject variation. The two types of variation overlap but are not identical. I maintain my assessment that this is creative but theoretically questionable. No score adjustment needed.

### Adjustments Made:

No score adjustments after verification. The analysis holds under scrutiny.

---

## Strengths

### Solution A
1. **Signal energy bypass for neutral gesture**: A concrete solution to a real failure mode of instance normalization, with minimal parameter overhead (1 extra parameter).
2. **Honest impact estimates**: The admission that DANN is "borderline" with 7 subjects and the downgrade from HIGH to MEDIUM impact shows intellectual honesty.
3. **Complete architecture with channel attention**: The SE-block style attention in `GestureCNNv2` is well-suited for EMG where channel importance varies by gesture.

### Solution B
1. **DataLoader-based test evaluation**: More memory-efficient than loading all test data at once.
2. **Quick_finetune function**: A complete, practical implementation of few-shot adaptation.
3. **Dual normalization options**: Offering both z-score and L2 normalization with a configurable mode provides flexibility.

### Solution C
1. **Calibration data validation**: The pre-training check for missing calibration data (Revision 1) addresses a practical data quality issue.
2. **Target threshold table**: Providing baseline, target, and stretch goals for LOSO accuracy, worst-subject accuracy, and macro F1 gives clear success criteria.
3. **Session-level adversarial domains**: Creative approach to the small-N subject problem, even if theoretically questionable.

---

## Areas for Improvement

### Solution A - Priority: Medium
- **Missing calibration validation**: No check for whether calibration data exists in all files before training with MVC normalization.
- **No files-to-modify summary**: A developer would need to trace through the plan to identify which files need changes.

### Solution B - Priority: High
- **Overcomplicated normalization**: The recommendation to use BOTH per-window z-score AND InstanceNorm throughout is not well-justified and may strip too much signal.
- **Underspecified realtime integration**: Claims "zero changes" to realtime script but doesn't demonstrate this.

### Solution C - Priority: High
- **Weak normalization strategy**: MVC-only normalization without instance norm leaves the model vulnerable to calibration data quality issues and doesn't normalize distribution shape.
- **BatchNorm in cross-subject architecture**: Using BatchNorm in GestureCNNv2 and GestureCNNAdversarial contradicts the cross-subject goal.
- **Complexity**: 8 changes with conditional/optional flags increase implementation risk.

---

## Score Summary

| Criterion | Sol A Score | Sol A Weighted | Sol B Score | Sol B Weighted | Sol C Score | Sol C Weighted |
|-----------|------------|----------------|------------|----------------|------------|----------------|
| Technical Soundness (0.30) | 4.0 | 1.20 | 3.0 | 0.90 | 3.5 | 1.05 |
| Cross-Subject Impact (0.25) | 4.0 | 1.00 | 3.5 | 0.875 | 3.5 | 0.875 |
| Implementation Feasibility (0.20) | 3.5 | 0.70 | 3.5 | 0.70 | 3.0 | 0.60 |
| Completeness (0.15) | 3.5 | 0.525 | 3.0 | 0.45 | 3.5 | 0.525 |
| Clarity & Actionability (0.10) | 3.0 | 0.30 | 3.5 | 0.35 | 3.0 | 0.30 |
| **Weighted Total** | | **3.73** | | **3.28** | | **3.35** |

---

## Confidence Assessment

**Confidence Level:** High

**Confidence Factors:**
- Evidence strength: Strong -- all claims verified against actual codebase (`train_cnn.py`, `gesture_model_cnn.py`, `realtime_gesture_cnn.py`)
- Criterion clarity: Clear -- the five criteria map well to the plan contents
- Edge cases: Handled -- self-verification addressed potential biases (length, normalization strategy merits)

---

## Actionable Improvements

**High Priority:**
- [ ] Solution A should add calibration data validation (adopt C's Revision 1)
- [ ] Solution B should simplify normalization to instance norm at input only, keeping BatchNorm in intermediate layers
- [ ] Solution C should replace BatchNorm with InstanceNorm at minimum in the input layer of proposed architectures

**Medium Priority:**
- [ ] All solutions should provide a complete, updated `_resolve_architecture` and `load_cnn_bundle` implementation
- [ ] Solution A should add a files-to-modify summary

**Low Priority:**
- [ ] Solution B should recommend a default normalization mode rather than leaving it configurable
- [ ] Solution C should consolidate the 8 changes into a clear default path vs optional extensions
