# Evaluation Report: CNN Cross-Subject Generalization Improvement Plans

## Executive Summary

Three improvement plans were evaluated for achieving cross-subject EMG gesture recognition. All three correctly identify the core problems (global z-score normalization, broken GroupShuffleSplit evaluation, lack of augmentation) and propose overlapping solution sets. Solution A provides the most technically rigorous plan with the best balance of correctness, self-awareness of limitations, and actionability. Solution B is comprehensive and nearly as strong but occasionally proposes conflicting normalization strategies (both per-window z-score AND InstanceNorm throughout, plus L2 normalization as an alternative) without clearly resolving which to use. Solution C introduces the most diverse set of techniques (subject-balanced sampling, prototypical networks, session-level adversarial domains) but has a critical technical weakness in its primary normalization strategy and includes a flawed augmentation (electrode shift on non-contiguous sensors, later self-corrected).

- **Artifacts**: `.specs/cnn-improvement-plan.a.md`, `.specs/cnn-improvement-plan.b.md`, `.specs/cnn-improvement-plan.c.md`
- **Overall Scores**: A: 3.80, B: 3.48, C: 3.24
- **Verdict**: Solution A -- GOOD; Solution B -- ACCEPTABLE; Solution C -- ACCEPTABLE

---

## Criterion Scores

| Criterion | Weight | Sol A | Sol B | Sol C |
|-----------|--------|-------|-------|-------|
| Technical Soundness | 0.30 | 4.0 | 3.5 | 3.0 |
| Cross-Subject Impact | 0.25 | 4.0 | 3.5 | 3.5 |
| Implementation Feasibility | 0.20 | 3.5 | 3.5 | 3.0 |
| Completeness | 0.15 | 3.5 | 3.5 | 3.5 |
| Clarity & Actionability | 0.10 | 3.5 | 3.0 | 3.0 |

---

## Detailed Analysis

### 1. Technical Soundness (Weight: 0.30)

#### Solution A

**Evidence Found:**

- Root cause analysis (Section 1.1) correctly identifies all four factors: amplitude scaling, spatial pattern shift, temporal dynamics, baseline drift. The table format maps each factor to its specific impact on the current model, citing the actual normalization approach in `train_cnn.py`.
- Section 1.2 correctly identifies the GroupShuffleSplit bug: "splits by file path, not by subject." Verified against `train_cnn.py` line 318-321 where `groups` (file paths) are used. This is accurate.
- The InstanceNorm proposal (Change 2) correctly uses `nn.InstanceNorm1d(channels[0], affine=False, track_running_stats=False)` at the input layer only, keeping BatchNorm in deeper layers. This is a sound choice: InstanceNorm at input removes subject-specific amplitude, while BatchNorm in deeper layers provides training stability. The distinction between InstanceNorm (per-sample) and BatchNorm (population) is correctly explained.
- Q2 and Revision 1 identify the neutral-gesture failure mode of instance normalization and propose a signal energy bypass: `raw_energy = x.pow(2).mean(dim=(1, 2))`. This is a genuine and important concern that neither B nor C addresses as elegantly. The energy bypass solution is technically correct and minimal (1 extra parameter).
- The DANN caveat (Section, Change 5) honestly states "With 7 this is borderline" and recommends starting with `adv_weight = 0.1`. This is appropriate caution.
- Q4 correctly identifies that channel permutation is inappropriate for Delsys Trigno sensors placed on specific muscles. This demonstrates understanding of the physical setup.

**What's wrong:**
- The `GestureCNNv2` architecture (Change 4) uses `ChannelAttention` (squeeze-and-excitation) but applies it to convolutional feature channels, not EMG sensor channels. The naming is misleading. SE blocks are a valid architectural choice, but calling them "ChannelAttention" in an EMG context where "channel" also means "sensor" could cause confusion.
- The estimated impact table (Section 8) claims instance normalization alone gives "+15-25% absolute." This is optimistic. The EMG literature typically shows 10-15% gains from normalization changes alone on small subject pools. The claim is not unreasonable but borders on overconfident.
- The kernel size is changed from 11 (current codebase) to 7 in GestureCNNv2 without justification. The current model uses `KERNEL_SIZE = 11` (verified in `train_cnn.py` line 37).

**Score: 4.0/5.0** -- Mostly correct with appropriate self-awareness of limitations. The neutral-gesture energy bypass is a genuine insight. Minor issues with naming and one unjustified hyperparameter change.

#### Solution B

**Evidence Found:**

- Root cause analysis is thorough and largely overlaps with A. Correctly identifies the same 4 factors and the GroupShuffleSplit bug.
- Change 2 proposes BOTH per-window z-score at input AND replacing all BatchNorm with InstanceNorm throughout. The text says "Recommended: Use BOTH." This is technically questionable. If you apply per-window z-score at the input (making input zero-mean, unit-variance per channel), then InstanceNorm in the first conv block receives already-normalized input and its per-instance normalization becomes largely redundant. Meanwhile, InstanceNorm in deeper layers (replacing BatchNorm) removes the population-level statistics that help training stability. Solution B does not explain why both are needed simultaneously.
- Revision A introduces L2 normalization as an alternative normalization strategy, creating three competing approaches (per-window z-score, InstanceNorm throughout, L2 norm) without clear guidance on which to actually use. The plan says "test both" but this adds implementation complexity.
- Q3 identifies the silent window problem but proposes clipping normalized values to [-5, 5] and a "silence gate" with a hardcoded RMS threshold. This is less elegant than A's energy bypass (which feeds the information to the model) and introduces an additional hyperparameter (SILENCE_RMS_THRESHOLD) that must be tuned.
- The DANN proposal includes a proper lambda schedule (`2/(1+exp(-10*p))-1`) and Revision C caps lambda_max at 0.3, showing appropriate caution.
- Q5 correctly assesses the dataset size vs parameter count tradeoff and makes the architecture change conditional on underfitting evidence. This is good engineering judgment.

**What's wrong:**
- The conflicting normalization strategies (z-score + InstanceNorm + L2) are a significant technical weakness. A developer following this plan would need to make decisions that the plan leaves unresolved.
- The `GestureCNNWithDANN` model embeds per-window normalization directly in `forward()`, but also proposes using InstanceNorm throughout. If the model does both, the first InstanceNorm layer's input is already per-window normalized, making it apply normalization to an already-normalized signal.
- The quick_finetune function (Change 7) freezes `model.features.parameters()` but then unfreezes them at the end "for future use." In a realtime deployment context, this is concerning -- the model weights are modified in-place during inference startup.

**Score: 3.5/5.0** -- Solid technical foundation but the normalization strategy is internally inconsistent. Multiple competing approaches without clear resolution.

#### Solution C

**Evidence Found:**

- Root cause analysis is solid and includes the "System Inventory" table (Section 2) that maps each component to its limitation. This is a useful framing.
- Change 2 proposes MVC-calibration-only normalization with clip-and-scale (`np.clip(emg_norm, -5.0, 5.0) / 5.0`). This is a fundamentally different approach from A and B's instance normalization. The argument is that MVC calibration already maps subjects into a common physiological coordinate system (0=rest, 1=MVC).
- The MVC-only approach has a critical weakness: it assumes reliable MVC calibration data exists for all sessions. Q1 acknowledges this: "I did not verify this for all 42 files." Revision 1 adds a validation step, but the fallback (per-file z-score) reintroduces the very problem being solved.
- The clip-and-scale approach (`/5.0`) is an arbitrary normalization that compresses dynamic range. If a subject's MVC-normalized signal regularly exceeds 5.0 (which is common in EMG), information is lost. This is technically inferior to instance normalization which adapts to whatever range the signal has.
- Change 6 proposes electrode-shift augmentation (circular channel shifting), then Q2 correctly identifies this is inappropriate for Delsys Trigno sensors. Revision 2 corrects this, but the initial proposal reveals a gap in understanding the physical setup.
- Change 7 (SubjectBalancedSampler) is a unique and valid contribution not present in A or B. Ensuring balanced subject representation per batch helps gradient stability.
- Change 8 (Prototypical Network) is a more advanced fallback than the simple head-finetuning in B. The concept is sound but adds substantial complexity.
- Revision 3 proposes using session IDs instead of subject IDs for adversarial training (30-40 domains instead of 7). This is a creative insight that addresses the small-N problem more directly than A or B's approaches. However, it conflates inter-session variability (electrode placement, time of day) with inter-subject variability, which may not produce the desired invariance.

**What's wrong:**
- The MVC-only normalization strategy assumes all subjects have calibration data. The plan acknowledges uncertainty about this but does not resolve it convincingly.
- The clip-and-scale at `/5.0` is an arbitrary constant that may discard discriminative information or leave the model operating in a suboptimal input range.
- The `GestureCNNv2` in Change 5 uses BatchNorm, not InstanceNorm, while the adversarial model in Change 4 also uses BatchNorm by default. This means the architecture still accumulates subject-specific running statistics during training -- the very problem identified in Section 1.2.
- The `GestureCNNAdversarial` (Change 4) does not include per-window normalization in its `forward()` method, unlike Solution B's version. The normalization fix (Change 2) is described as a preprocessing step, not an architectural feature.

**Score: 3.0/5.0** -- Several technically sound individual contributions (balanced sampling, session-level adversarial domains, prototypical networks) but the primary normalization strategy is weaker than A or B's, and the architecture retains BatchNorm throughout.

**Improvement suggestion for all:** Provide empirical evidence or literature citations for the expected accuracy gains. All three plans make accuracy predictions without citations.

---

### 2. Cross-Subject Impact (Weight: 0.25)

#### Solution A

**Evidence Found:**

- Instance normalization at the input is the highest-impact single change for cross-subject EMG. The explanation is clear: "if you normalize each window independently, you remove inter-subject amplitude differences while preserving the temporal shape." This directly addresses the dominant source of cross-subject variance (amplitude scaling).
- The energy bypass for neutral gesture (Revision 1) preserves the most common class (resting state) under instance normalization, preventing a regression that would undermine real-world usability.
- The five augmentations (channel scaling, noise, temporal shift, channel dropout, temporal stretch) all target specific inter-subject variations. The augmentation list is well-motivated and complete.
- Change 6 (MVC consistency fix) correctly removes double normalization, which is a real issue in the current pipeline.
- The expected outcome table estimates 60-75% LOSO with all changes, compared to 30-50% baseline. The fallback recommendation ("collect 2-3 more subjects" or "few-shot calibration") is practical.

**What's wrong:**
- Does not propose subject-balanced batch sampling, which could improve training stability with imbalanced subject data (Matthew has 8 sessions, subject01 has 4).
- The plan does not address what happens when MVC calibration data is missing for some sessions.

**Score: 4.0/5.0** -- The combination of instance normalization + energy bypass + targeted augmentation directly addresses the three main sources of cross-subject variance. The plan correctly prioritizes normalization over architectural changes.

#### Solution B

**Evidence Found:**

- Proposes both per-window z-score and InstanceNorm, which if properly resolved would provide strong cross-subject normalization.
- The L2 normalization alternative (Revision A) is an interesting idea: "preserves inter-channel amplitude ratios that are gesture-discriminative." This could outperform per-channel z-score for gestures distinguished primarily by which channels are active.
- Change 4 (DANN) includes a proper lambda schedule and integrates well with the normalization changes.
- The silence gate (Revision B) addresses the neutral gesture problem but less elegantly than A's energy bypass -- it bypasses the model entirely rather than giving the model the information it needs.
- Quick finetune (Change 7) provides a realistic fallback path.

**What's wrong:**
- The unresolved normalization strategy weakens the impact assessment. If a developer picks the wrong combination, the cross-subject benefit could be less than expected.
- The silence gate approach means the model never learns to handle low-energy windows, creating a dependency on an additional threshold parameter.

**Score: 3.5/5.0** -- Strong cross-subject potential but the unresolved normalization strategy introduces risk. The silence gate is a patch rather than a solution.

#### Solution C

**Evidence Found:**

- The MVC-only normalization approach ("0=resting, 1=MVC") is physiologically motivated and could work well when calibration data is reliable.
- Session-level adversarial domains (Revision 3) is a creative approach to the small-N problem that could provide better generalization than subject-level adversarial training.
- SubjectBalancedSampler (Change 7) directly addresses the data imbalance between subjects.
- Prototypical networks (Change 8) provide the most flexible adaptation mechanism of all three plans.
- The deployment scenario (Section 5) clearly describes both zero-shot and few-shot paths.

**What's wrong:**
- The MVC-only normalization relies on calibration data quality, which varies across sessions. If calibration is noisy or missing, the normalization fails silently.
- The clip-and-scale (`/5.0`) discards information. A subject whose MVC-normalized signal reaches 8.0 on a channel (which happens with EMG overshoots) loses 37.5% of that signal's dynamic range.
- BatchNorm throughout the architecture means the model still accumulates subject-specific statistics. This partially undermines the normalization fix.

**Score: 3.5/5.0** -- Several unique contributions (session-level adversarial, balanced sampling, prototypical networks) give this plan strong cross-subject potential, but the normalization strategy is the weakest of the three.

---

### 3. Implementation Feasibility (Weight: 0.20)

#### Solution A

**Evidence Found:**

- All code snippets are syntactically correct PyTorch. The `GestureCNN` modification (Change 2) maintains the same constructor signature with an added optional parameter `use_instance_norm_input=True`, preserving backward compatibility.
- The `CnnBundle.standardize()` modification correctly checks metadata and returns input unchanged when instance norm is used.
- The bundle format (Section 5) maintains backward compatibility: "Old bundles without `use_instance_norm_input` will default to `False`."
- The `_resolve_architecture` update (Q5) shows both branches for `GestureCNN` and `GestureCNNv2`, which is consistent with the current codebase's approach (verified: `gesture_model_cnn.py` lines 68-81).
- The `EMGAugmentor` class correctly operates on `(C, T)` tensors with proper PyTorch operations.

**What's wrong:**
- The `GestureCNNv2` introduces a new constructor signature `(in_channels, num_classes, ...)` that differs from the current `GestureCNN(channels, num_classes, ...)` where `channels` is a list. The `_resolve_architecture` code in Q5 would need to handle this difference, but the shown code does not fully resolve the parameter mapping.
- The augmentor's `temporal_stretch` method creates a `pad` tensor without specifying the device, which would fail on GPU tensors.

**Score: 3.5/5.0** -- Code is mostly correct and maintains backward compatibility. Minor parameter mapping issues and a device mismatch in augmentation.

#### Solution B

**Evidence Found:**

- LOSO implementation is more detailed than A's, using a DataLoader for batched evaluation (handles memory for large test sets).
- The `GestureCNNWithDANN` integrates per-window normalization, InstanceNorm, and adversarial training into a single class. The `return_subject_logits` flag cleanly separates training and inference paths.
- Bundle compatibility strategy (save as standard GestureCNN by extracting weights) is practical but fragile -- it requires manually mapping state dict keys.
- The `quick_finetune` function modifies model weights in-place during inference, which could be problematic in a multi-threaded realtime context.

**What's wrong:**
- The plan proposes both per-window z-score in `forward()` AND InstanceNorm throughout the network. The code for `GestureCNNWithDANN.forward()` applies per-window z-score, then passes through InstanceNorm1d layers. A developer would need to decide whether this double normalization is intentional.
- The `GestureCNN` modification adds `use_instance_norm` but the `forward()` method unconditionally applies per-window z-score regardless of the flag. This is inconsistent.
- `GestureCNNWithDANN` uses `affine=True` for InstanceNorm, which reintroduces learnable parameters that could encode subject-specific information. This partially undermines the goal.

**Score: 3.5/5.0** -- Implementation details are generally correct but internal inconsistencies in the normalization approach would confuse a developer.

#### Solution C

**Evidence Found:**

- The `EMGAugmenter` uses numpy operations (not torch), which means it must operate on numpy arrays before conversion to tensors. The `AugmentedEMGDataset` stores `self.X = X` as numpy and calls `torch.from_numpy(x)` after augmentation. This works but requires the augmenter to also use numpy.
- The `SubjectBalancedSampler` is a complete implementation with proper reshuffling when a subject's data is exhausted.
- The `GestureCNNAdversarial` uses the current codebase's hyperparameters (`kernel_size=11`, `dropout=0.4`), showing awareness of the existing setup.
- The architecture registry (Revision 4) is a clean extensibility mechanism.

**What's wrong:**
- The `GestureCNNv2` (Change 5) does NOT include per-window normalization or InstanceNorm. It uses BatchNorm throughout. This means deploying GestureCNNv2 still requires external normalization, but Change 2 removes the global z-score in favor of MVC-only normalization with clip-and-scale. The interaction between these changes is unclear.
- The `per_session_normalize` function is shown as a standalone function, not integrated into the existing `load_windows_from_file` which already handles MVC calibration. The interaction with the existing code is not fully specified.
- The prototypical network (Change 8) would require significant changes to `realtime_gesture_cnn.py` that are not fully specified.
- The electrode-shift augmentation (Change 6) was proposed then partially retracted, but the code remains in the plan without clear "do not use" markers.

**Score: 3.0/5.0** -- Several implementation gaps, particularly in how the normalization changes interact with the existing pipeline and the lack of InstanceNorm in the new architecture.

---

### 4. Completeness (Weight: 0.15)

#### Solution A

**Evidence Found:**

- Covers all four areas: training pipeline (LOSO, augmentation, DANN training loop), evaluation methodology (LOSO), architecture changes (GestureCNNv2, instance norm, energy bypass), and real-time inference changes (Section 4).
- Bundle format compatibility is explicitly addressed (Section 5) with a concrete example.
- Roadmap has 5 phases covering weeks 1-4.
- Verification questions (Section 6) cover 5 topics with substantive answers and revisions.

**What's missing:**
- No discussion of class-weighted loss interaction with cross-subject training. The current `USE_CLASS_WEIGHTS = True` computes weights from the full training set, but with LOSO the class distribution changes per fold.
- No explicit guidance on saving the final deployable model (LOSO is for evaluation; the final model trains on all subjects). This is mentioned briefly in Section 3 Phase 5 but not fully specified.

**Score: 3.5/5.0** -- Covers all major areas with good depth. Missing some secondary considerations.

#### Solution B

**Evidence Found:**

- Covers training pipeline (LOSO, augmentation, DANN, mixup, normalization fix), evaluation methodology (LOSO with per-subject and per-class metrics), architecture changes (ResBlock, InstanceNorm, DANN), and real-time inference (silence gate, quick finetune).
- Section 4 explicitly shows how to modify `main()` for LOSO and includes the important insight: "LOSO is for EVALUATION. The final deployable model trains on ALL 7 subjects."
- Files to modify are listed in Section 9 with per-file change descriptions.
- Revision summary table (Section 7) clearly documents what changed and why.

**What's missing:**
- The interaction between per-window z-score, InstanceNorm, and L2 normalization is not resolved. A developer would need to make significant design decisions not covered in the plan.
- No explicit bundle format example (unlike A's Section 5).

**Score: 3.5/5.0** -- Comprehensive coverage with the important LOSO/deployment distinction. Loses points for unresolved normalization strategy.

#### Solution C

**Evidence Found:**

- Covers training pipeline (LOSO, augmentation, DANN, balanced sampling), evaluation methodology (LOSO, secondary metrics table in Section 6), architecture changes (GestureCNNv2, adversarial), and deployment (Section 5 with zero-shot and few-shot scenarios).
- Section 6 "How to Measure Success" includes a target thresholds table with baseline, target, and stretch goals. This is the most explicit success criteria of the three plans.
- Provides the most deployment-oriented view with explicit "New Subject Experience" scenarios.
- Files to modify are not listed in a summary table (unlike B).
- Section 9 has a complete change summary table with impact, effort, and phase assignments.

**What's missing:**
- No explicit bundle format example.
- The prototypical network (Change 8) is described at a conceptual level but integration with `realtime_gesture_cnn.py` is not specified.
- No discussion of how calibration data availability affects the training pipeline.

**Score: 3.5/5.0** -- Good breadth with unique contributions (success metrics table, deployment scenarios). Loses points for incomplete prototypical network integration.

---

### 5. Clarity & Actionability (Weight: 0.10)

#### Solution A

**Evidence Found:**

- Changes are numbered and ranked by expected impact with clear labels (CRITICAL, HIGH, MEDIUM, LOW-MEDIUM).
- Each change has a "Why" section explaining rationale, followed by "What to change" with specific file names.
- Code snippets are self-contained and can be copy-pasted with minimal modification.
- The roadmap (Section 3) has clear phases with numbered steps.
- Verification Q&A provides additional context for decision-making.

**What's wrong:**
- Some code snippets (e.g., the `GestureCNNv2`) are long and might benefit from more inline comments explaining design decisions.
- The `_resolve_architecture` update in Q5 is incomplete (returns different types for different architectures).

**Score: 3.5/5.0** -- Well-organized and actionable. A developer could follow this plan with moderate effort.

#### Solution B

**Evidence Found:**

- Similar organizational structure to A with numbered changes and impact labels.
- Provides Option A and Option B for normalization, then says "use BOTH" -- this creates confusion rather than clarity.
- The dual normalization strategy (Revision A adds a third option) means a developer must make multiple decisions not resolved in the plan.
- The `main()` modification in Section 4 is the most complete of the three plans, showing the full LOSO + final model flow.

**What's wrong:**
- Too many normalization options without clear resolution. A developer following this plan would spend significant time deciding which approach to implement.
- Revision D makes the architecture change conditional but does not specify the exact metrics to evaluate.

**Score: 3.0/5.0** -- Good structure but the unresolved normalization strategy significantly reduces actionability.

#### Solution C

**Evidence Found:**

- Clean organizational structure with numbered sections.
- The "System Inventory" table (Section 2) provides excellent context for understanding the current state.
- The success metrics table (Section 6) gives clear targets.
- Deployment scenarios (Section 5) are concrete and user-focused.

**What's wrong:**
- Change 6 (electrode shift) is proposed, then partially retracted in the revisions. A developer would need to read the entire document to understand which changes to actually implement.
- The `per_session_normalize` function duplicates existing logic in `load_windows_from_file` without explaining the relationship.
- Phase 3 groups augmentation, adversarial training, AND balanced sampling into 3-5 days, which may be overly optimistic.

**Score: 3.0/5.0** -- Good high-level organization but some inconsistencies between initial proposals and revisions reduce clarity.

---

## Self-Verification

### Questions Asked:

1. Am I penalizing Solution B's dual normalization approach too harshly? Is there a valid reason to use both per-window z-score and InstanceNorm throughout?
2. Am I giving Solution A too much credit for the energy bypass? Is this really a significant contribution or a minor detail?
3. Is Solution C's MVC-only normalization actually worse than instance normalization, or could it be better in practice?
4. Am I being biased by length/detail -- Solution A is the longest. Does it actually contain more substance?
5. Have I fairly evaluated each plan's handling of backward compatibility with the existing codebase?

### Answers:

1. **Re-examining B's dual normalization:** Looking at the literature, combining per-window normalization with InstanceNorm in hidden layers is not standard practice. Per-window z-score already ensures zero-mean, unit-variance input per channel. InstanceNorm in the first hidden layer would then normalize the convolution outputs, which is conceptually different (it normalizes feature maps, not raw signals). So there IS a case for using InstanceNorm in hidden layers even after input normalization -- it prevents feature drift in deeper layers. However, Solution B does not articulate this distinction. It presents "use BOTH" without explaining the complementary roles. My assessment stands: the plan should have been clearer.

2. **Re-examining A's energy bypass:** The neutral gesture problem under instance normalization is well-documented in the signal processing literature. If 30-50% of real-time windows are neutral (plausible during normal operation), misclassifying them degrades user experience significantly. The energy bypass is a low-cost, high-value fix. Solution B's silence gate achieves the same goal but bypasses the model entirely, which is less principled. Solution C does not address this issue at all. My assessment of A's contribution here is fair.

3. **Re-examining C's MVC-only normalization:** MVC normalization IS the standard approach in clinical EMG. It has decades of literature support. However, it assumes (a) reliable calibration data, (b) consistent MVC effort across subjects, and (c) that MVC captures the full range of activation seen during gestures. Assumption (b) is often violated -- subjects vary in their willingness/ability to produce true MVC. Instance normalization is more robust because it adapts to whatever the actual signal range is, window by window. The clip-and-scale at `/5.0` further discards information. My assessment that C's normalization is weaker stands, but I should acknowledge that MVC normalization has real merit and could outperform instance normalization if calibration is high quality. No score change.

4. **Length bias check:** Solution A: ~638 lines. Solution B: ~671 lines. Solution C: ~672 lines. All three are similar in length. Solution A is actually the shortest. My scores are not driven by length. The difference is in content quality: A has fewer unresolved ambiguities, better self-correction, and a more coherent normalization strategy.

5. **Backward compatibility:** All three plans maintain backward compatibility with existing bundles by storing dummy mean/std values and checking metadata flags. Solution A provides the most explicit bundle format example (Section 5). Solution C proposes an architecture registry (Revision 4) which is the most extensible approach. Solution B proposes extracting weights from the DANN model into a standard GestureCNN at save time, which is functional but fragile. I should give C slightly more credit for the architecture registry. However, it does not change the overall ranking since this is a small detail within the Feasibility criterion.

### Adjustments Made:
- No score changes after verification. The initial assessment is consistent with the evidence.

---

## Weighted Score Calculation

### Solution A
| Criterion | Score | Weight | Weighted |
|-----------|-------|--------|----------|
| Technical Soundness | 4.0 | 0.30 | 1.20 |
| Cross-Subject Impact | 4.0 | 0.25 | 1.00 |
| Implementation Feasibility | 3.5 | 0.20 | 0.70 |
| Completeness | 3.5 | 0.15 | 0.525 |
| Clarity & Actionability | 3.5 | 0.10 | 0.35 |
| **Total** | | | **3.775** |

### Solution B
| Criterion | Score | Weight | Weighted |
|-----------|-------|--------|----------|
| Technical Soundness | 3.5 | 0.30 | 1.05 |
| Cross-Subject Impact | 3.5 | 0.25 | 0.875 |
| Implementation Feasibility | 3.5 | 0.20 | 0.70 |
| Completeness | 3.5 | 0.15 | 0.525 |
| Clarity & Actionability | 3.0 | 0.10 | 0.30 |
| **Total** | | | **3.45** |

### Solution C
| Criterion | Score | Weight | Weighted |
|-----------|-------|--------|----------|
| Technical Soundness | 3.0 | 0.30 | 0.90 |
| Cross-Subject Impact | 3.5 | 0.25 | 0.875 |
| Implementation Feasibility | 3.0 | 0.20 | 0.60 |
| Completeness | 3.5 | 0.15 | 0.525 |
| Clarity & Actionability | 3.0 | 0.10 | 0.30 |
| **Total** | | | **3.20** |

---

## Key Strengths

### Solution A
1. **Energy bypass for neutral gesture**: Unique and technically correct solution to a real problem with instance normalization. Neither B nor C addresses this as well.
2. **Clean normalization strategy**: InstanceNorm at input only, BatchNorm in deeper layers. No conflicting approaches.
3. **Honest impact assessment**: Explicitly states DANN is "borderline" with 7 subjects and recommends treating it as optional.

### Solution B
1. **Explicit LOSO/deployment distinction**: Section 4 clearly separates evaluation (LOSO) from final model training (all subjects), which is the most important operational insight.
2. **Quick finetune fallback**: Provides a practical path when zero-shot performance is insufficient.
3. **Conditional architecture upgrade**: Revision D ties architecture changes to empirical evidence of underfitting.

### Solution C
1. **Session-level adversarial domains**: Creative solution to the small-N problem for adversarial training (30-40 domains vs 7).
2. **Subject-balanced sampling**: Unique contribution that addresses data imbalance across subjects.
3. **Success metrics framework**: Section 6 provides the most explicit and measurable success criteria of all three plans.

---

## Areas for Improvement

### Solution A - Priority: Medium
- **Evidence**: No subject-balanced batch sampling; no handling of missing calibration data.
- **Impact**: Training could be dominated by subjects with more sessions.
- **Suggestion**: Add a SubjectBalancedSampler (as in C) and a calibration data validation step.

### Solution B - Priority: High
- **Evidence**: Three competing normalization approaches (per-window z-score, InstanceNorm throughout, L2 norm) without resolution.
- **Impact**: A developer cannot implement this plan without making significant design decisions not covered in the document.
- **Suggestion**: Choose ONE normalization strategy and justify it. Test alternatives in LOSO but specify a default.

### Solution C - Priority: High
- **Evidence**: GestureCNNv2 and GestureCNNAdversarial both use BatchNorm, not InstanceNorm. The MVC-only normalization plus BatchNorm still accumulates subject-specific statistics.
- **Impact**: The architecture retains the very problem (population-level statistics) that the normalization change aims to fix.
- **Suggestion**: Replace BatchNorm with InstanceNorm (affine=True) in the new architectures, or add per-window normalization in the forward() method.

---

## Confidence Assessment

**Confidence Level**: High

**Confidence Factors**:
- Evidence strength: Strong -- all claims verified against the actual codebase (`gesture_model_cnn.py`, `train_cnn.py`, `realtime_gesture_cnn.py`)
- Criterion clarity: Clear -- all five criteria are well-defined with specific weights
- Edge cases: Handled -- verified normalization strategies against both training and inference paths

---

## Actionable Improvements

**High Priority**:
- [ ] Solution A should add calibration data validation and subject-balanced sampling
- [ ] Solution B must resolve the competing normalization strategies into a single coherent approach
- [ ] Solution C must replace BatchNorm with InstanceNorm in proposed architectures

**Medium Priority**:
- [ ] All solutions should cite EMG cross-subject literature to support accuracy predictions
- [ ] All solutions should address class weight recomputation in LOSO folds

**Low Priority**:
- [ ] Solution A should justify the kernel size change from 11 to 7
- [ ] Solution C should remove the retracted electrode-shift augmentation code from the document
