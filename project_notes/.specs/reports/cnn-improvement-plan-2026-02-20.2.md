# Evaluation Report: CNN Cross-Subject Generalization Improvement Plans

## Executive Summary

Three improvement plans were evaluated for achieving cross-subject EMG gesture recognition generalization. All three correctly identify the core problems (global z-score mismatch, broken GroupShuffleSplit evaluation, lack of augmentation) and propose overlapping but differently weighted solutions. Solution A provides the most technically precise and practically actionable plan, with a novel signal-energy bypass for instance normalization and honest self-critique of each technique's limitations. Solution B offers strong analysis and a useful dual-normalization alternative (L2 vs z-score) but introduces internal contradictions by recommending both per-window z-score AND InstanceNorm replacement throughout the network. Solution C is the most comprehensive in scope (8 changes including prototypical networks and subject-balanced sampling) but spreads effort thin, lacks instance normalization, and relies on an MVC-calibration-only normalization strategy that does not fully solve the cross-subject problem.

- **Artifact**: `.specs/cnn-improvement-plan.a.md`, `.specs/cnn-improvement-plan.b.md`, `.specs/cnn-improvement-plan.c.md`
- **Overall Winner**: Solution A
- **Confidence Level**: High

---

## Criterion Scores

| Criterion | Weight | Solution A | Solution B | Solution C |
|-----------|--------|------------|------------|------------|
| Technical Soundness | 0.30 | 4.0 | 3.5 | 3.0 |
| Cross-Subject Impact | 0.25 | 4.0 | 3.5 | 3.0 |
| Implementation Feasibility | 0.20 | 4.0 | 3.5 | 3.0 |
| Completeness | 0.15 | 3.5 | 3.5 | 4.0 |
| Clarity & Actionability | 0.10 | 4.0 | 3.5 | 3.5 |
| **Weighted Total** | **1.00** | **3.90** | **3.50** | **3.20** |

---

## Detailed Analysis

### 1. Technical Soundness (Weight: 0.30)

#### Solution A

**Evidence Found:**
- Correctly identifies 4 compounding factors for EMG subject-specificity (amplitude scaling, spatial pattern shift, temporal dynamics, baseline drift) with a clear table mapping each to the current model's specific failure mode.
- Accurately diagnoses the `GroupShuffleSplit` bug: "splits by file path, not by subject" (confirmed at `train_cnn.py` line 318-321 where `groups` is file paths, not subject IDs).
- The InstanceNorm1d proposal at the input layer (`nn.InstanceNorm1d(channels[0], affine=False, track_running_stats=False)`) is technically correct. The `affine=False` choice is deliberate -- it prevents the model from learning subject-specific affine parameters during training. The `track_running_stats=False` ensures no population statistics are accumulated.
- The signal-energy bypass (Q2 answer) addresses a real failure mode: "Instance normalization on a near-zero signal amplifies noise to unit variance, potentially making neutral windows look like active gestures." The proposed fix (`raw_energy = x.pow(2).mean(dim=(1, 2))` concatenated to features) is computationally negligible and architecturally sound.
- The `GestureCNNv2` architecture with SE-attention (`ChannelAttention`) is well-designed. The parameter count estimate of ~180K is reasonable.
- Correctly notes the DANN limitation with 7 subjects: "the adversary has a trivially easy classification task (7 classes)" and appropriately downgrades it to MEDIUM impact.
- Explicitly and correctly rejects channel permutation augmentation for Delsys Trigno sensors (Q4).

**Issues Found:**
- The `temporal_stretch` augmentation uses `range` as a parameter name, which shadows the Python builtin. Minor but would cause a linting warning.
- The LOSO function loads the entire test set into GPU memory at once (`torch.from_numpy(X_test_norm).to(device)`) without batching. For large test sets this could cause OOM on GPU. Solution B correctly uses a DataLoader for the LOSO test evaluation.

**Score: 4.0/5.0**

**Improvement:** Batch the LOSO test evaluation to avoid potential OOM issues.

#### Solution B

**Evidence Found:**
- Root cause analysis is thorough and accurate; correctly identifies all the same issues as Solution A.
- Proposes BOTH per-window z-score at input AND replacing BatchNorm with InstanceNorm throughout: "Use BOTH. Apply per-window z-score at the input (Option A) AND replace BatchNorm with InstanceNorm in the convolutional blocks (Option B)." However, this is technically questionable. InstanceNorm1d with `affine=True` in the conv blocks already normalizes per-instance per-channel. Adding per-window z-score on top is redundant for the input layer and creates a double-normalization at the first conv block's InstanceNorm. The plan does not address this redundancy.
- The dual normalization strategy (Revision A: L2 norm as alternative to z-score) is a genuinely useful idea. L2 normalization preserves inter-channel amplitude ratios, which is a valid concern for EMG where relative channel activation patterns are gesture-discriminative.
- The silence gate proposal (Revision B: RMS threshold for neutral detection) is practical but the threshold value (`0.01`) is arbitrary and depends on the MVC calibration scale, which is not discussed.
- The adversarial training section includes a sigmoid lambda schedule, which is standard DANN practice. However, capping `lambda_max=0.3` without justification beyond "7 subjects is small" is conservative -- the original DANN paper uses 1.0 even with small domain counts.
- The `GestureCNNWithDANN` model applies per-window z-score in `forward()` but also uses `InstanceNorm1d(out_ch, affine=True)` in the conv blocks. This triple normalization (per-window z-score + InstanceNorm at conv1 + InstanceNorm at conv2 + ...) is excessive and could collapse feature variance.

**Issues Found:**
- Internal contradiction: recommends replacing ALL BatchNorm with InstanceNorm (Option B) but the `GestureCNNWithDANN` code still uses `nn.BatchNorm1d(out_ch)` when `use_instance_norm=False`. The default in the DANN model constructor is `use_instance_norm=True`, so this is somewhat moot, but the inconsistency suggests incomplete thinking.
- The L2 normalization alternative (`x.pow(2).sum(dim=(1, 2), keepdim=True).sqrt()`) sums over both channels AND time, producing a single scalar per window. This removes ALL amplitude information, not just per-channel amplitude. It would make different gestures with different overall activation levels harder to distinguish. Solution A's signal-energy bypass is a better approach to the same problem.
- Does not address the neutral-gesture failure mode of per-window normalization as thoroughly as Solution A (silence gate is a workaround, not an architectural fix).

**Score: 3.5/5.0**

**Improvement:** Resolve the redundancy between per-window z-score and InstanceNorm throughout the network. Pick one approach and justify it clearly.

#### Solution C

**Evidence Found:**
- Root cause analysis is accurate but less detailed than A or B (3 factors instead of 4; misses baseline drift).
- The MVC-calibration-only normalization strategy (Change 2) is the most divergent proposal. It recommends removing global z-score AND NOT using instance normalization, instead relying solely on MVC calibration + clip/scale: `emg_norm = np.clip(emg_norm, -5.0, 5.0) / 5.0`. This is a weaker approach because MVC calibration depends on calibration data quality, which varies. If a subject's MVC calibration is unreliable (poor effort, electrode slippage), the normalization fails. Instance normalization is self-contained and requires no calibration data.
- The clip-and-scale approach (`np.clip(x, -5.0, 5.0) / 5.0`) is a hard nonlinearity that destroys signal information above 5x the MVC scale. The magic number 5.0 is unjustified.
- The electrode-shift augmentation (Change 6) with circular channel shifting is initially proposed but then correctly flagged as conditional on sensor layout. However, the initial proposal shows a misunderstanding: Delsys Trigno sensors are not spatially contiguous. This was caught in verification (Q2) but should not have been proposed in the first place.
- The `SubjectBalancedSampler` (Change 7) is a useful contribution not present in A or B. Balanced batches ensure the model sees all subjects equally, preventing gradient bias toward over-represented subjects (Matthew: 8 sessions vs subject01: 4 sessions). The implementation is correct.
- The prototypical network (Change 8) is creative and provides a principled few-shot adaptation mechanism. However, it adds significant complexity and the implementation is incomplete (no training loop shown, no episodic training protocol described).
- The session-level adversarial domains idea (Revision 3, from Q3) is genuinely novel and better-motivated than subject-level DANN. Using ~35 session IDs instead of 7 subject IDs provides a harder adversarial task and could learn more generalizable invariance. However, this contradicts the goal of subject-invariance -- session-invariance within a subject is a different objective than across-subject invariance.
- The `GestureCNNAdversarial` model uses `nn.BatchNorm1d` (not InstanceNorm) in its feature extractor. At inference, BatchNorm uses running stats from training subjects, reintroducing the subject-specificity problem that DANN is supposed to solve. Solution A's use of InstanceNorm + DANN is more internally consistent.

**Issues Found:**
- The `GestureCNNv2` architecture uses `nn.BatchNorm1d(128)` inside `stage2` without a ReLU after it: `nn.Conv1d(64, 128, 1, bias=False), nn.BatchNorm1d(128), ResBlock1d(128, ...)`. The ResBlock starts with Conv1d, meaning BatchNorm output goes directly into another Conv1d without activation. The ResBlock's own internal path has ReLU, but the skip connection path does not have matching dimensions if the projection uses a different kernel size than the ResBlock. Actually, looking more carefully, the `stage2` Sequential applies BatchNorm then immediately the ResBlock. The ResBlock's `forward` is `relu(x + block(x))` where `x` has already been through BN+no-activation. This is architecturally awkward but not broken.
- No mention of the double-normalization problem (MVC + z-score) in the root cause analysis table, even though it is identified later in Change 5. The root cause table says "calibration data exists per-session but normalization stats are computed *after* calibration, creating double-normalization" but this is listed as a current state limitation, not as a root cause of cross-subject failure.
- Calibration data validation (Revision 1) is a useful practical consideration not present in A or B.

**Score: 3.0/5.0**

**Improvement:** Adopt instance normalization as the primary normalization mechanism rather than relying on MVC calibration quality. MVC calibration is a useful preprocessing step but should not be the only defense against inter-subject amplitude differences.

---

### 2. Cross-Subject Impact (Weight: 0.25)

#### Solution A

**Evidence Found:**
- Instance normalization at input is the single highest-impact change for cross-subject EMG, correctly identified as such with an estimated "+15-25% absolute" improvement. This aligns with published EMG literature where per-window normalization consistently provides the largest gains.
- The signal-energy bypass addresses a specific failure mode (neutral gesture) that would otherwise degrade the most common real-time state.
- Data augmentation (channel amplitude scaling, noise, temporal shift/stretch, channel dropout) simulates all major sources of inter-subject variability.
- The plan's expected outcome table is realistic: "60-75% LOSO accuracy... up from an estimated 30-50% baseline." This matches published results for similar sensor setups and subject counts.
- The fallback recommendation ("few-shot calibration mode: collect 30 seconds... and fine-tune only the classifier head") is practical and honest about the limitations of 7 subjects.

**Issues Found:**
- Does not propose subject-balanced batching (Solution C's Change 7), which could matter given the unequal session distribution across subjects.

**Score: 4.0/5.0**

#### Solution B

**Evidence Found:**
- Per-window normalization is correctly identified as the highest-impact change, with the same "+15-25%" estimate.
- The dual normalization strategy (z-score vs L2) provides an experimental framework to find the better approach, which is scientifically sound.
- The "quick_finetune" function for few-shot calibration is well-implemented and provides a realistic fallback.
- Expected outcome table is realistic: "25-40%" baseline, cumulative improvement to "65-78%" with all changes + calibration.

**Issues Found:**
- The recommendation to use BOTH per-window z-score AND InstanceNorm throughout could actually hurt cross-subject performance by over-normalizing, collapsing inter-channel variance that carries gesture information.
- Does not address the neutral-gesture failure mode as robustly as Solution A (silence gate is a threshold-based heuristic; Solution A's energy bypass is an architectural fix).

**Score: 3.5/5.0**

#### Solution C

**Evidence Found:**
- The MVC-calibration-only normalization is the weakest approach for cross-subject generalization. It depends entirely on calibration data quality and consistency. At inference on a new subject, if their MVC calibration is poor (common with untrained subjects who do not contract maximally), the normalization fails.
- The session-level adversarial domains (Revision 3) could be counterproductive: forcing invariance across sessions of the SAME subject does not directly address inter-subject differences. It may even remove useful within-subject temporal consistency.
- The prototypical network (Change 8) is the strongest few-shot adaptation mechanism proposed, better than simple head fine-tuning (Solutions A and B), but it is marked as optional and contingent on Phase 4 results.
- Subject-balanced sampling (Change 7) directly addresses the data imbalance issue (Matthew 8 sessions vs subject01 4 sessions).
- Expected outcome: "60-75% LOSO accuracy" which is realistic but the plan is less likely to achieve it given the weaker normalization strategy.

**Score: 3.0/5.0**

---

### 3. Implementation Feasibility (Weight: 0.20)

#### Solution A

**Evidence Found:**
- All code snippets are syntactically correct PyTorch.
- The `GestureCNN` modification adds `use_instance_norm_input` as a constructor parameter with a default of `True`, maintaining backward compatibility (old bundles default to `False`).
- The `CnnBundle.standardize()` modification checks metadata for `use_instance_norm_input` and returns input unchanged -- zero changes needed in `realtime_gesture_cnn.py`.
- Bundle format evolution is clearly specified with example dict showing both vestigial mean/std and new architecture fields.
- The `_resolve_architecture` extension (Q5 answer) correctly handles both `"GestureCNN"` and `"GestureCNNv2"` types.
- The 4-phase roadmap (Evaluation -> Normalization+Augmentation -> Architecture -> Adversarial) is well-ordered -- each phase depends on the previous.

**Issues Found:**
- The `temporal_stretch` augmentation uses `range` as a parameter name (Python builtin shadow).
- The `GestureCNNv2` forward method references `self.pool_flatten` and `self.classifier` in the Q2 code snippet, but the class definition uses `self.head`. This is an inconsistency between the main architecture definition and the verification answer's modification.

**Score: 4.0/5.0**

#### Solution B

**Evidence Found:**
- Code snippets are mostly correct PyTorch.
- The LOSO evaluation uses DataLoader for test evaluation (better than Solution A's single-batch approach).
- The `quick_finetune` function freezes feature extractor parameters and only trains the head -- correct and practical.
- Bundle compatibility is handled identically to Solution A (mean=0, std=1 as no-ops).
- The `GestureCNNWithDANN` model integrates per-window z-score in `forward()`, which means the normalization is always applied regardless of how the data is preprocessed. This is good for robustness.

**Issues Found:**
- The `GestureCNN` modification puts per-window z-score in `forward()` unconditionally (no flag to disable it). This means even old bundles loaded with this code would get per-window normalization applied, breaking backward compatibility unless the class is versioned.
- The plan says "Replace `BatchNorm1d` with `InstanceNorm1d(out_ch, affine=True)` throughout" but the code shows `use_instance_norm` as a flag, not a wholesale replacement. The DANN model defaults to `use_instance_norm=True`, but the base `GestureCNN` does not -- it defaults to `False`. This inconsistency means Phase 2 and Phase 4 use different normalization approaches.
- The L2 normalization code (`x.pow(2).sum(dim=(1, 2), keepdim=True).sqrt()`) normalizes by a single scalar. The `keepdim=True` produces shape `(B, 1, 1)`, so this divides all channels and time steps by the same scalar. While this preserves inter-channel ratios, it also preserves the temporal shape, which is already a given with z-score normalization per channel.

**Score: 3.5/5.0**

#### Solution C

**Evidence Found:**
- Code snippets are syntactically correct.
- The `ARCHITECTURE_REGISTRY` pattern (Revision 4) is a clean extensibility mechanism not proposed by A or B.
- The `SubjectBalancedSampler` implementation is complete and handles edge cases (re-shuffling when a subject's indices are exhausted).
- The `per_session_normalize` function is clear and correct.
- The calibration data validation step (Revision 1) is a practical robustness measure.

**Issues Found:**
- The `GestureCNNAdversarial` model hardcodes `kernel_size=11` and `dropout=0.4`, matching the current architecture but not allowing configuration through the bundle. This is less flexible than A's approach.
- The bundle deployment code for the adversarial model tries to extract weights from `adv_model.gesture_head` (a single Linear layer) and load them into `deploy_model.head[2]` (the third element of the head Sequential). This requires knowledge that `head[2]` is the Linear layer (after AdaptiveAvgPool1d and Flatten). This is fragile -- if the head structure changes, this indexing breaks silently.
- The `electrode_shift_augmentation` is proposed then retracted, suggesting the plan was not sufficiently vetted before writing.
- The `ProtoNet.forward` uses `torch.cdist` which computes pairwise distances. The negative sign makes it a similarity measure, but the output is labeled as "logits" -- using Euclidean distance as logits without temperature scaling typically produces poorly calibrated probabilities when passed through softmax.

**Score: 3.0/5.0**

---

### 4. Completeness (Weight: 0.15)

#### Solution A

**Evidence Found:**
- Covers: LOSO evaluation, normalization (instance norm), augmentation, architecture upgrade, adversarial training, MVC consistency fix, mixup, real-time inference changes, bundle format, backward compatibility.
- Missing: No subject-balanced sampling strategy. No few-shot calibration function (only mentioned as a recommendation, not implemented). No calibration data validation step.
- The roadmap covers 5 phases with explicit ordering.

**Score: 3.5/5.0**

#### Solution B

**Evidence Found:**
- Covers: LOSO evaluation, dual normalization strategy, augmentation, adversarial training, MVC consistency fix, architecture upgrade, quick-finetune calibration, silence gate, bundle compatibility.
- Includes `quick_finetune()` implementation -- the only solution with a complete few-shot adaptation function.
- Includes dual normalization alternatives with code for both.
- Roadmap covers 6 phases including optional fast adaptation.
- Lists all files to modify with specific changes per file (Section 9).
- Missing: No subject-balanced sampling. No calibration data validation.

**Score: 3.5/5.0**

#### Solution C

**Evidence Found:**
- Covers: LOSO evaluation, MVC normalization, augmentation, adversarial training (with session-level domains), architecture upgrade, electrode-shift augmentation (conditional), subject-balanced sampling, prototypical networks, calibration data validation, architecture registry.
- Most comprehensive scope: 8 changes across 6 phases.
- Includes deployment scenario (Section 5) with zero-shot and few-shot paths.
- Includes success metrics table with baseline, target, and stretch goals (Section 6).
- Includes calibration data validation (Revision 1) -- unique contribution.
- Lists all files to modify (Section 9).
- Missing: No instance normalization (the most impactful single change for cross-subject EMG). The prototypical network implementation is incomplete (no episodic training loop).

**Score: 4.0/5.0**

**Improvement (A):** Add calibration data validation and few-shot fine-tuning implementation.

---

### 5. Clarity & Actionability (Weight: 0.10)

#### Solution A

**Evidence Found:**
- Changes are numbered 1-7 with clear priority labels (CRITICAL, HIGH, MEDIUM, LOW-MEDIUM).
- Each change has: Why, What, code snippet, impact estimate, integration notes.
- The roadmap is week-based with numbered steps.
- Verification questions are answered in-line with code snippets.
- Revisions are explicitly listed with "What changed" and "Why" columns.

**Score: 4.0/5.0**

#### Solution B

**Evidence Found:**
- Changes are numbered 1-7 with impact labels.
- The normalization section offers two options (A and B) then recommends "Use BOTH" -- this is actionable but the recommendation is questionable (as discussed above).
- Roadmap is phase-based with numbered steps.
- Revision table summarizes all changes clearly.
- Files-to-modify section (Section 9) is a useful reference.

**Issues Found:**
- The dual normalization recommendation creates ambiguity: a developer must decide which to implement first, test both, or implement both simultaneously.
- The DANN section is long and could be overwhelming for a developer unfamiliar with adversarial training.

**Score: 3.5/5.0**

#### Solution C

**Evidence Found:**
- Changes are numbered 1-8 with impact and effort estimates.
- Each change has rationale, implementation code, and integration notes.
- The deployment scenario (Section 5) is practical and user-facing -- useful for a capstone project.
- The success metrics table (Section 6) with specific numeric targets is actionable.
- Phase-based roadmap with day estimates.
- Summary table (Section 9) is concise.

**Issues Found:**
- 8 changes across 6 phases is potentially overwhelming. The plan does not clearly indicate which changes are critical vs optional until the summary table.
- The electrode-shift augmentation is proposed then retracted, creating noise in the document.

**Score: 3.5/5.0**

---

## Self-Verification

### Questions Asked:

1. **Am I penalizing Solution C for not using instance normalization when its MVC-calibration approach could be equally valid?**
2. **Am I biased toward Solution A due to its signal-energy bypass being a novel idea, when it might not actually matter in practice?**
3. **Is Solution B's recommendation to use BOTH per-window z-score AND InstanceNorm actually problematic, or is the double normalization a non-issue?**
4. **Am I giving Solution A too much credit for its verification questions when all three solutions have self-verification sections?**
5. **Does Solution C's greater breadth (8 changes, prototypical networks, balanced sampling) deserve more credit than I gave it?**

### Answers:

1. **Re-examined: MVC-calibration-only normalization IS weaker for cross-subject generalization.** MVC calibration requires the new subject to produce a reliable maximum voluntary contraction, which is inconsistent (untrained subjects often produce only 60-70% of true MVC). Instance normalization is self-contained and requires no external calibration data for normalization. The EMG literature consistently shows per-window normalization outperforming MVC-only normalization for cross-subject work. My scoring stands.

2. **Re-examined: The signal-energy bypass IS practically important.** The neutral gesture is the most common state during real-time use (the user is resting most of the time). If instance normalization causes neutral windows to be misclassified as active gestures, the system becomes unusable in practice. Solution B's silence gate addresses this but as a threshold heuristic (fragile), while Solution A's energy bypass is an architectural solution (robust). My scoring stands, but I acknowledge the practical importance may be overstated given that the existing system already uses `MIN_CONFIDENCE=0.65` gating.

3. **Re-examined: Double normalization CAN be problematic.** Per-window z-score at input produces zero-mean, unit-variance data per channel. InstanceNorm1d at the first conv layer's output would then normalize the conv-transformed features, which is fine. But the per-window z-score at input + InstanceNorm1d at input (Solution B's "Option B" says replace BatchNorm throughout, which includes a potential InstanceNorm at the first layer) would be redundant. In Solution B's implementation, the per-window z-score is in `forward()` and `InstanceNorm1d` is in the conv blocks (not at input), so the double normalization happens at the output of the first conv layer, not at the input. This is actually fine -- the z-score normalizes raw input, InstanceNorm normalizes learned features. I was slightly harsh. However, Solution B does not explain this interaction, which is a clarity issue. Adjusting Solution B's Technical Soundness up marginally would be warranted, but the other issues (L2 norm design flaw, backward compatibility break) keep it at 3.5.

4. **Re-examined: Solution A's verification questions are qualitatively better.** Q2 (neutral gesture failure mode) led to a concrete architectural change (energy bypass). Q4 (channel permutation) led to an explicit design decision. Solutions B and C also have good verification questions, but their revisions are more conservative (parameter tuning, conditional flags). Solution A's revisions are more substantive.

5. **Re-examined: Solution C's breadth is both a strength and weakness.** The prototypical network is a genuinely creative idea for few-shot adaptation, and the subject-balanced sampler addresses a real data imbalance issue. However, the prototypical network implementation is incomplete (no episodic training) and the breadth means the plan is harder to execute within a capstone project timeline. The completeness score of 4.0 already reflects the breadth advantage. The lower scores on other criteria reflect the execution quality within that breadth.

### Adjustments Made:

No score adjustments needed after verification. The initial evaluation stands.

---

## Score Summary

| Criterion | Weight | Solution A | Weighted A | Solution B | Weighted B | Solution C | Weighted C |
|-----------|--------|------------|------------|------------|------------|------------|------------|
| Technical Soundness | 0.30 | 4.0 | 1.20 | 3.5 | 1.05 | 3.0 | 0.90 |
| Cross-Subject Impact | 0.25 | 4.0 | 1.00 | 3.5 | 0.875 | 3.0 | 0.75 |
| Implementation Feasibility | 0.20 | 4.0 | 0.80 | 3.5 | 0.70 | 3.0 | 0.60 |
| Completeness | 0.15 | 3.5 | 0.525 | 3.5 | 0.525 | 4.0 | 0.60 |
| Clarity & Actionability | 0.10 | 4.0 | 0.40 | 3.5 | 0.35 | 3.5 | 0.35 |
| **Weighted Total** | **1.00** | | **3.925** | | **3.50** | | **3.20** |

---

## Key Strengths

### Solution A
1. **Signal-energy bypass for neutral gesture**: The only solution that architecturally addresses the InstanceNorm failure mode on low-energy windows, preserving the ability to detect resting state while still normalizing active gesture patterns.
2. **Precise InstanceNorm configuration**: Uses `affine=False, track_running_stats=False` at input, which is the correct choice for subject-invariant normalization (no learnable parameters that could overfit to training subjects).
3. **Honest self-assessment**: The DANN technique is correctly downgraded from HIGH to MEDIUM impact with 7 subjects, and the plan includes explicit fallback guidance.

### Solution B
1. **Dual normalization alternatives**: Proposing both z-score and L2 normalization with a plan to test both is scientifically rigorous.
2. **Complete few-shot calibration implementation**: The `quick_finetune` function is the most production-ready adaptation mechanism across all three solutions.
3. **Conservative adversarial training**: Capping lambda_max at 0.3 is pragmatically safer for a capstone project.

### Solution C
1. **Subject-balanced sampling**: The `SubjectBalancedSampler` addresses data imbalance (8 sessions vs 4 sessions) that A and B ignore.
2. **Calibration data validation**: The pre-training check for missing calibration data is a practical robustness measure unique to this solution.
3. **Prototypical network fallback**: Provides the most principled few-shot adaptation mechanism for new subjects, going beyond simple head fine-tuning.

---

## Areas for Improvement

### Solution A - Priority: Medium
- **Evidence**: No calibration data validation step; no complete few-shot fine-tuning implementation.
- **Impact**: If some session files lack calibration data, training could silently use unnormalized windows.
- **Suggestion**: Add Solution C's calibration data validation step and Solution B's `quick_finetune` function.

### Solution B - Priority: High
- **Evidence**: Recommends both per-window z-score AND InstanceNorm replacement without addressing the interaction between these two normalizations.
- **Impact**: A developer following this plan may implement redundant normalization that collapses feature variance or, worse, may implement them in conflicting ways.
- **Suggestion**: Choose one primary normalization strategy (per-window z-score at input OR InstanceNorm at input) and use standard BatchNorm or InstanceNorm in deeper layers. Do not recommend "Use BOTH" without explaining the interaction.

### Solution C - Priority: High
- **Evidence**: Does not propose instance normalization; relies solely on MVC calibration quality for inter-subject normalization.
- **Impact**: Cross-subject generalization will be significantly worse than solutions using instance normalization. MVC calibration quality is unreliable for untrained subjects.
- **Suggestion**: Add per-window instance normalization as the primary normalization mechanism, keeping MVC calibration as a preprocessing step.

---

## Actionable Improvements

**High Priority:**
- [ ] Adopt Solution A as the primary plan
- [ ] Add Solution C's calibration data validation step to Solution A
- [ ] Add Solution B's `quick_finetune` function to Solution A's deployment phase
- [ ] Add Solution C's `SubjectBalancedSampler` to Solution A's training pipeline

**Medium Priority:**
- [ ] Test Solution B's L2 normalization alternative alongside Solution A's per-channel instance norm
- [ ] Consider Solution C's prototypical network as a Phase 5 fallback if LOSO remains below 65%

**Low Priority:**
- [ ] Investigate Solution C's session-level adversarial domains as an alternative to subject-level DANN

---

## Confidence Assessment

**Confidence Level**: High

**Confidence Factors:**
- Evidence strength: Strong -- all claims were verified against the actual codebase (`train_cnn.py`, `gesture_model_cnn.py`)
- Criterion clarity: Clear -- the evaluation criteria are well-defined with specific weights
- Edge cases: Handled -- verified normalization interactions, backward compatibility, and neutral-gesture failure modes
- Bias check: Verified that scoring was not influenced by solution length (C is longest but scores lowest), confidence of tone, or novelty bias
