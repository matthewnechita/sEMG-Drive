from __future__ import annotations

from pathlib import Path

from create_report_skeletons import _docx_write, _t


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "report_drafts"


def _paragraph(text: str, style: str = "Normal", align: str | None = None) -> tuple[str, dict]:
    return ("paragraph", {"text": text, "style": style, "align": align})


def _page_break() -> tuple[str, str]:
    return ("page_break", "")


def _final_report_draft() -> list[tuple[str, str]]:
    # Keep the draft content in the same tuple format as create_report_skeletons
    # so both scripts share one OOXML writer.
    p = _paragraph
    br = _page_break
    return [
        p("[Project Title]", "Title"),
        p("Final Design Report Draft", "Subtitle"),
        p("Course: ENEL 500", "CoverMeta"),
        p("Team: [Names]", "CoverMeta"),
        p("Sponsor / Supervisor: [Names]", "CoverMeta"),
        p("Date: [Fill in final submission date]", "CoverMeta"),
        p("This draft is pre-filled from the active repo state and should be finalized against the exact experiments reported in the submission.", "Caption"),
        br(),
        p("Table of Contents", "Heading1"),
        p("[Update this in Word after headings and page numbers are finalized.]"),
        br(),
        p("Glossary", "Heading1"),
        _t(
            [
                ["Term", "Definition"],
                ["EMG", "Electromyography."],
                ["MVC", "Maximum voluntary contraction."],
                ["Strict layout", "Fixed pair-number-specific sensor placement policy used by the active training and inference workflow."],
                ["Neutral recovery", "Targeted data-collection protocol for turn-to-neutral transitions."],
                ["CARLA", "Open-source driving simulator used for downstream control evaluation."],
            ],
            widths=[1800, 7200],
            header=True,
        ),
        br(),
        p("AI Use Disclosure and Verification", "Heading1"),
        p("This draft was assembled with AI-assisted writing support and then cross-checked against the active code paths and technical notes in the repository. The final submission should replace this paragraph with the full ENEL 500-compliant disclosure, including the exact tools, versions, dates of use, verification steps, and the location of the maintained AI log."),
        p("At minimum, the final disclosure should state which model or assistant was used, what parts of the report were assisted, whether project data was exposed, and how all technical claims, commands, metrics, and code descriptions were verified by the team."),
        p("Executive Summary", "Heading1"),
        p("This capstone project developed an EMG-driven driving interface built around Delsys Trigno sensors, a strict sensor-placement workflow, convolutional neural network classification, realtime gesture inference, and CARLA-based downstream control evaluation. The project goal was to move from an earlier mixed-layout prototype toward a more reproducible, technically defensible system that could support both offline classification experiments and closed-loop simulator testing."),
        p("The final implemented workflow collects strict-layout EMG data through a custom Python GUI, resamples heterogeneous sensor streams to a shared 2000 Hz time grid, filters the data with fixed notch and bandpass filters, and trains residual 1D CNN models on overlapping EMG windows. The active deployment path focuses on the three-gesture subset of neutral, left_turn, and right_turn because this is the narrowest and most reliable control setting on the current branch. Realtime output is further stabilized through smoothing, confidence gating, hysteresis, and ambiguity rejection to reduce neutral flicker."),
        p("On the systems side, the project integrates the realtime classifier into a CARLA manual-control client. The current simulator layer supports steering overrides from EMG, scenario logging, visible checkpoints, and fixed evaluation scenarios for lane keeping and overtaking. The evaluation plan for both the final report and the research paper is organized around three main result groups: offline model metrics, end-to-end latency metrics, and CARLA scenario metrics. Final numeric results, tables, and figures should be inserted once the experiment set is frozen."),
        p("Project Snapshot", "Heading2"),
        _t(
            [
                ["Area", "Current implementation status"],
                ["Data collection", "Strict-layout GUI collection working; standard and neutral-recovery protocols implemented."],
                ["Preprocessing", "Resampling, filtering, and recalibration support implemented on strict dataset roots."],
                ["Modeling", "GestureCNNv2 active for per-subject and cross-subject training."],
                ["Realtime", "Dual-arm strict inference working with smoothing, hysteresis, and ambiguity rejection."],
                ["Simulator", "CARLA integration active with visible checkpoint scenarios and logged completion state."],
                ["Main reported metrics", "Offline model metrics, latency metrics, and CARLA scenario metrics."],
            ],
            widths=[2200, 6800],
            header=True,
        ),
        p("Motivation and Objectives", "Heading1"),
        p("The core motivation for the project is to investigate whether surface EMG can support practical human-machine interaction for driving-related control tasks. Traditional manual controls assume that drivers can reliably operate the steering wheel, signals, and auxiliary controls through direct mechanical input. An EMG-driven interface could eventually support alternative control pathways for accessibility, assistive interfaces, or other non-traditional interaction settings, but only if the sensing, learning, and deployment pipeline is stable enough to produce repeatable behavior."),
        p("Early versions of the project were limited by inconsistent sensor placement, mixed historical layouts, and runtime instability around the neutral class. Those issues directly affect reproducibility and degrade deployment behavior because the meaning of each input channel can drift between sessions while the neutral class tends to flicker under low-confidence conditions. The current project direction therefore prioritized engineering consistency over breadth of gesture vocabulary."),
        p("The main objectives of the final system are summarized below.", "Compact"),
        _t(
            [
                ["Objective", "Purpose"],
                ["Strict sensor placement", "Preserve a stable channel contract across sessions."],
                ["Reproducible preprocessing", "Align heterogeneous Delsys streams before training and inference."],
                ["Reliable CNN training", "Support subject-specific and pooled models on strict data roots."],
                ["Stable realtime behavior", "Reduce neutral flicker through post-processing and runtime tuning."],
                ["CARLA integration", "Support downstream control experiments in a repeatable simulator environment."],
                ["Evaluation workflow", "Report offline, latency, and scenario-level metrics from logged runs."],
            ],
            widths=[2400, 6600],
            header=True,
        ),
        p("Methodology and Design", "Heading1"),
        p("System Overview", "Heading2"),
        p("The implemented pipeline is collection -> resampling -> filtering -> optional recalibration -> windowing and label generation -> CNN training -> realtime post-processing -> CARLA control evaluation. The active top-level entrypoints are python DelsysPythonGUI.py for collection, python tools/resample_raw_dataset.py, python emg/filtering.py, python tools/recalibrate.py for preprocessing support, python train_per_subject.py and python train_cross_subject.py for model training, python realtime_gesture_cnn.py for standalone inference, and python carla_integration/manual_control_emg.py for simulator integration."),
        p("The final system uses dedicated strict data roots rather than legacy mixed-layout roots. Raw strict collections are stored under data_strict/, resampled strict data under data_resampled_strict/, and active trained bundles under models/strict/. This separation is important because the strict workflow depends on saved per-session channel label metadata and should not silently fall back to incompatible historical sessions."),
        p("Data Collection Pipeline", "Heading2"),
        p("Collection is performed through DelsysPythonGUI.py and DataCollector/CollectDataWindow.py. The GUI supports explicit sensor pairing by pair number, scanning of previously paired sensors, live EMG viewing, and scripted labeled protocols. Only EMG channels are surfaced to the collection workflow, and pair identity is preserved in saved metadata for later strict-layout resolution."),
        _t(
            [
                ["Protocol", "Labels / focus", "Timing summary"],
                ["Standard", "left_turn, right_turn, neutral, signal_left, signal_right, horn", "5 s gesture, 5 s neutral, 5 repetitions, 1 s inter-gesture rest, 0.5 s trim, calibration enabled"],
                ["Neutral recovery", "left_turn -> neutral, right_turn -> neutral", "3 s gesture, 5 s neutral, 6 repetitions, 0 s visible rest, 1.0 s neutral lead trim, calibration enabled"],
            ],
            widths=[1500, 3600, 3900],
            header=True,
        ),
        p("Two labeled collection protocols are currently implemented. The standard protocol uses six active gesture labels: left_turn, right_turn, neutral, signal_left, signal_right, and horn. Each active gesture lasts 5.0 s, each neutral segment lasts 5.0 s, the protocol uses five repetitions, and calibration is enabled by default with 5.0 s neutral rest and 5.0 s MVC segments. Inter-gesture rest is labeled neutral_buffer and trimmed by 0.5 s so that transition regions are not treated as clean gesture examples."),
        p("A separate neutral-recovery protocol was added to support flicker tuning without changing the baseline collection workflow. It repeatedly records left_turn-to-neutral and right_turn-to-neutral transitions with 3.0 s active holds, 5.0 s neutral holds, and six repetitions. The protocol omits an explicit visible neutral-buffer stage, but leading trim is still applied internally so that the immediate release slice is not treated as settled neutral data. This design was chosen because the observed deployment failure mode was instability when returning to neutral after a steering gesture, not a lack of generic steady-neutral examples."),
        p("Raw collections are saved as NPZ files containing the raw EMG matrix, per-channel timestamps, per-sample labels, the protocol event list, metadata, and optional calibration arrays. Standard sessions are saved as <subject>_session<session>_raw.npz, while recovery sessions are saved with a neutral_recovery suffix so they remain separable in later analysis."),
        p("Strict Sensor Placement Workflow", "Heading2"),
        p("The most important design change in the current system is the strict fixed-position sensor policy. Older mixed-layout assumptions were replaced because they made it difficult to guarantee that a given channel index represented the same physical site across sessions. The active helper in emg/strict_layout.py defines strict_pair_v1 and resolves channels by fixed pair identity instead of discovery order or coarse sensor type alone."),
        _t(
            [
                ["Arm", "Pair mapping", "Total channels"],
                ["Right", "1,2,3 = Avanti; 7 = Maize; 9 = Galileo; 11 = Mini", "17"],
                ["Left", "4,5,6 = Avanti; 8 = Maize; 10 = Galileo", "16"],
            ],
            widths=[1200, 6600, 1200],
            header=True,
        ),
        p("For the right arm, the strict mapping assigns pairs 1, 2, and 3 to the three Avanti slots, pair 7 to the Maize sensor, pair 9 to the Galileo sensor, and pair 11 to the Mini sensor. For the left arm, pairs 4, 5, and 6 map to the three Avanti slots, pair 8 maps to Maize, and pair 10 maps to Galileo. This yields 17 strict channels on the right arm and 16 strict channels on the left arm."),
        p("The justification for this policy is straightforward: training and inference only become reproducible if the model sees a stable channel contract. Strict resolution therefore fails closed if required pair numbers are missing, if sensor kinds do not match the expected slots when inferable, or if the final channel count does not match the arm-specific contract. This is intentionally more conservative than the older salvage path because silent remapping would hide acquisition problems and weaken result credibility."),
        p("Preprocessing Pipeline", "Heading2"),
        p("The recommended strict preprocessing order is resampling, filtering, optional recalibration dry run, optional recalibration apply, and retraining. Resampling is required because different Delsys sensor types can stream at different effective sample rates. The downstream pipeline assumes aligned channels by sample index, so the raw data are first interpolated onto a shared common time grid."),
        _t(
            [
                ["Stage", "Key settings / behavior"],
                ["Resampling", "Strict raw files -> 2000 Hz common grid; per-channel interpolation; nearest-neighbor label remap"],
                ["Filtering", "60 Hz notch, 120 Hz notch, 20-450 Hz sixth-order bandpass"],
                ["Recalibration", "Optional replacement if MVC-to-neutral ratio < 1.5x"],
                ["Retraining", "Required whenever filter policy or layout policy changes"],
            ],
            widths=[1800, 7000],
            header=True,
        ),
        p("The current resampler reads strict raw files, estimates per-channel sampling rates from timestamp differences, determines the overlapping valid time interval across channels, builds a common 2000.0 Hz grid, linearly interpolates each channel onto that grid, and remaps labels to the new grid by nearest-neighbor transfer. Resampling metadata is written back into metadata['resampling'] so the transformation remains auditable."),
        p("After resampling, the active filtering stack applies a 60 Hz notch, a 120 Hz notch, and a 20-450 Hz sixth-order bandpass filter. The same filter stack is mirrored in the realtime path so that training and deployment consume similarly processed signals. A recalibration utility is also available as a recovery path for sessions with weak explicit MVC calibration. When the median MVC-to-neutral RMS ratio falls below 1.5x, the script can derive replacement neutral_mean and mvc_scale values directly from labeled filtered data while preserving the original arrays under backup keys."),
        p("Windowing, Label Generation, and Model Architecture", "Heading2"),
        p("Training operates on filtered EMG windows rather than handcrafted feature vectors. The active windowing configuration uses 200-sample windows and a 100-sample step, which correspond to 100 ms windows with 50 ms stride at 2000 Hz. Each window is labeled by the majority class of its constituent samples, while neutral_buffer windows are dropped from training. The per-subject trainer can also remove windows with insufficient label purity through a minimum label confidence threshold."),
        _t(
            [
                ["Component", "Active configuration"],
                ["Windowing", "200 samples, 100-sample step, majority label per window"],
                ["Excluded label", "neutral_buffer"],
                ["Model", "GestureCNNv2 residual 1D CNN with channel attention"],
                ["Feature stages", "32, 64, 128 channels"],
                ["Extra feature", "Pre-normalization energy scalar concatenated into the head"],
            ],
            widths=[2200, 6600],
            header=True,
        ),
        p("The active classifier is GestureCNNv2 in emg/gesture_model_cnn.py. The model consumes input shaped as (batch, channels, time) and applies input InstanceNorm1d followed by a residual 1D CNN backbone with three feature stages at 32, 64, and 128 channels. Each stage includes squeeze-and-excitation channel attention, and the network terminates in global pooling and a linear classifier head."),
        p("A key architectural justification is the explicit energy bypass. The network computes raw window energy before input normalization and concatenates that scalar into the final classification head. This was intentionally added because per-window normalization can suppress amplitude information that helps separate near-zero neutral windows from active gesture windows. The energy bypass therefore gives the model a direct rest-versus-active cue while still benefiting from normalized temporal features."),
        p("Training Configuration", "Heading2"),
        _t(
            [
                ["Setting", "Per-subject", "Cross-subject"],
                ["Epochs", "60", "80"],
                ["Batch size", "512", "512"],
                ["Optimizer", "Adam", "Adam"],
                ["Learning rate", "1e-4", "1e-4"],
                ["Dropout", "0.25", "0.25"],
                ["Label smoothing", "0.05", "0.05"],
                ["Min label confidence", "0.85", "0.75"],
                ["Sampling", "Group-aware file split when possible", "Subject-balanced WeightedRandomSampler"],
            ],
            widths=[2200, 3200, 3200],
            header=True,
        ),
        p("The project maintains two active training scripts. train_per_subject.py trains a subject-specific model for one arm, while train_cross_subject.py trains a pooled one-arm model across subjects. The active per-subject defaults use a 200-sample window, 100-sample step, calibration-aware normalization when quality checks pass, a 0.85 minimum label confidence threshold, a 0.2 test split, a batch size of 512, 60 epochs, Adam optimization with learning rate 1e-4, 0.25 dropout, and 0.05 label smoothing. The current augmentation set includes amplitude scaling, additive Gaussian noise, temporal circular shift, channel dropout, and temporal stretch."),
        p("The cross-subject workflow uses similar architecture and optimizer settings but extends training to 80 epochs and uses subject-balanced sampling through WeightedRandomSampler. This distinction should be preserved in the final report because subject-specific and pooled training answer different questions about personalization and generalization."),
        p("Realtime Inference and Tuning", "Heading2"),
        _t(
            [
                ["Runtime item", "Active value / behavior"],
                ["Gesture subset", "neutral, left_turn, right_turn"],
                ["Mode", "Dual-arm"],
                ["Windowing", "200-sample windows, 100-sample step"],
                ["Preset", "flicker_mild_margin"],
                ["Smoothing", "3"],
                ["Minimum confidence", "0.80"],
                ["Hysteresis", "Enabled with two-frame confirmation"],
                ["Ambiguity rejection", "Softmax reject enabled; min confidence 0.80; min margin 0.10"],
            ],
            widths=[2400, 6400],
            header=True,
        ),
        p("The active realtime entrypoint is realtime_gesture_cnn.py. The current deployment defaults operate in dual-arm mode, restrict the active gesture subset to neutral, left_turn, and right_turn, resample incoming streams to 2000 Hz, and mirror the offline filter stack. If valid calibration arrays are available at runtime, the code also applies neutral mean subtraction and MVC scaling before window classification."),
        p("Realtime output is stabilized through the currently active runtime tuning preset flicker_mild_margin. This preset applies smoothing of 3 frames, a minimum confidence threshold of 0.80, a dual-arm agreement threshold of 0.55, output hysteresis with two-frame confirmation, and softmax ambiguity rejection with a minimum confidence of 0.80 and a minimum top-two-class margin of 0.10. These controls were added because early live tests showed that neutral flicker could dominate deployment quality even when offline accuracy appeared acceptable."),
        p("The active decoder path also supports confidence gating and label restriction, allowing the runtime to deploy a three-gesture subset even when a bundle was trained with a larger label set. This is an important engineering choice because the current reportable control story is strongest in the three-gesture strict setting."),
        p("CARLA Integration", "Heading2"),
        _t(
            [
                ["CARLA item", "Current implementation"],
                ["Entry point", "carla_integration/manual_control_emg.py"],
                ["Control policy", "split_strength_latched_aux_v1"],
                ["Client FPS", "30"],
                ["Graphics mode", "Low graphics by default"],
                ["Vehicle blueprint", "Fixed sedan: vehicle.lincoln.mkz_2020"],
                ["Scenarios", "lane_keep_5min and highway_overtake"],
                ["Logging", "--eval-log-dir, --carla-log, --realtime-log"],
            ],
            widths=[2200, 6600],
            header=True,
        ),
        p("The active CARLA entrypoint is carla_integration/manual_control_emg.py, which launches realtime inference internally and converts published EMG outputs into vehicle-control actions. The client supports low-graphics operation, configurable logging, and visible evaluation scenarios. It does not derive throttle or braking from gestures; those remain wheel or manual inputs. This design keeps the EMG control problem narrow enough to evaluate steering-related behavior without simultaneously solving full longitudinal control."),
        p("The current control policy on the live branch is split_strength_latched_aux_v1. Under this mapping, the left arm acts as the strong or decisive steering arm, while the right arm acts as the weak or fine steering arm. Specifically, left_turn and right_turn on the left arm map to left_strong and right_strong steering, while the same labels on the right arm map to weaker left and right steering values. This removes the earlier requirement that both arms collapse onto the same label before a strong turn could be issued."),
        p("Auxiliary controls were also redesigned as stateful toggles instead of hold-based commands. signal_left on the left arm and signal_right on the right arm toggle the corresponding blinker on edge detection, with cooldown and timeout handling. Reverse is toggled by a dual-horn gesture but is blocked while the vehicle is moving above a small speed threshold. These choices were made because sustained EMG holds are less reliable for discrete auxiliary functions than edge-triggered state changes."),
        p("To support downstream evaluation, the simulator layer now includes two scenario presets implemented in code: a checkpoint-based lane-keeping route and an overtake scenario with a slower lead vehicle. Both use a fixed sedan blueprint and explicit start and finish logic. The lane-keeping scenario begins timing when the vehicle crosses the first checkpoint ahead of the spawn point and completes when the final visible checkpoint is reached. The overtake scenario requires both a successful pass-and-return objective and crossing the finish gate. Scenario state, checkpoint progress, completion time, and per-tick control data are written to CSV logs."),
        p("Product Scope and Functionality", "Heading1"),
        p("The delivered product is a working end-to-end EMG control research platform rather than a consumer-facing driving assistant. In its current form, the system supports: strict one-arm data collection through a GUI; separate standard and neutral-recovery collection protocols; strict-layout preprocessing on dedicated dataset roots; per-subject and cross-subject CNN training; dual-arm realtime inference on the three-gesture deployment subset; post-processing for flicker suppression; and simulator-side evaluation through CARLA with repeatable scenario logging."),
        p("The final system also includes several tuning and evaluation features that were not present in the earlier pipeline: named runtime presets, richer realtime prediction logging, CARLA steering dwell, visible checkpoints, explicit scenario completion logic, split-arm steering roles, and latched auxiliary gesture controls. These additions were motivated by the need to move from a proof-of-concept classifier toward a system that can be tuned, compared, and evaluated in a controlled way."),
        p("At the same time, the scope remains intentionally constrained. The strongest current deployment story is the strict three-gesture steering subset. Signal and reverse gestures are implemented in the CARLA client, but they only become meaningful if the loaded models expose the required labels. Legacy salvage-mode training also still exists for historical data analysis, but it is not the preferred path for the new strict workflow and should not be treated as part of the main delivered system."),
        p("Technical Specifications of Final Product", "Heading1"),
        p("Software stack. The active environment targets Python 3.10 with numpy, scipy, scikit-learn, matplotlib, pythonnet, PySide6, libemg, torch, torchvision, pygame, and the CARLA Python API. Windows with the Delsys SDK bridge is the expected host platform for live collection."),
        p("Acquisition and data specification. Right-arm strict sessions resolve to 17 EMG channels and left-arm strict sessions resolve to 16 EMG channels. Raw multi-sensor acquisitions are resampled to 2000 Hz, then filtered using 60/120 Hz notches and a 20-450 Hz bandpass. Training windows are 200 samples with 100-sample step."),
        p("Model specification. The active classifier is GestureCNNv2, a residual 1D CNN with channel attention and a scalar pre-normalization energy bypass. The active deployment subset is neutral, left_turn, and right_turn. The current default runtime preset is flicker_mild_margin."),
        p("Simulator specification. The CARLA client runs at a 30 FPS client cap in low-graphics mode by default and uses a fixed sedan blueprint for scenario runs. The current scenario presets are lane_keep_5min and highway_overtake, both with visible checkpoints and explicit completion state logging."),
        p("Measures of Success and Validation Tests", "Heading1"),
        p("Success Criteria", "Heading2"),
        p("The project should be judged successful if it demonstrates a repeatable strict-layout pipeline, stable three-gesture realtime inference, reproducible logging for downstream analysis, and a CARLA evaluation path that produces meaningful end-to-end metrics rather than only standalone classifier accuracy."),
        p("Main Metric Set", "Heading2"),
        _t(
            [
                ["Metric group", "Primary reported items"],
                ["Offline", "Balanced accuracy, macro F1"],
                ["Offline supporting", "Accuracy, weighted F1, per-class recall, worst-class recall, confusion-to-neutral rate, neutral prediction false-positive rate"],
                ["Latency", "Classifier, publish, control, and end-to-end latency with mean / median / p90 / p95 / max"],
                ["CARLA", "Mean lane error, lane error RMSE, lane invasions, scenario success / failure, scenario completion time"],
            ],
            widths=[2000, 6800],
            header=True,
        ),
        p("The main reported metric set for both the final report and the research paper should remain aligned. The primary groups are offline model metrics, latency metrics, and CARLA scenario metrics. Prompted realtime behavior metrics can still be used as optional backup analysis, but they should not be the required headline section."),
        p("Offline metrics. Primary offline metrics are balanced accuracy and macro F1. Supporting metrics are accuracy, weighted F1, per-class recall, worst-class recall, the confusion matrix, confusion-to-neutral rate, and neutral prediction false-positive rate."),
        p("Latency metrics. The latency analysis joins realtime and CARLA logs on prediction_seq and reports classifier, publish, control, and end-to-end latency. Each latency family should be summarized with mean, median, p90, p95, and max."),
        p("CARLA metrics. The main simulator metrics are mean lane error, lane error RMSE, lane invasions, scenario success or failure, and scenario completion time. Lane error is defined as the distance from the vehicle to the nearest lane centerline. Completion time is only valid for defined scenarios and is measured from the start checkpoint to the finish condition."),
        p("Validation Workflow", "Heading2"),
        _t(
            [
                ["Step", "Command / output"],
                ["Harvest offline metrics", "python eval_metrics/harvest_model_metrics.py --models-root models/strict"],
                ["Run logged CARLA scenario", "python carla_integration/manual_control_emg.py --scenario lane_keep_5min --eval-log-dir eval_metrics/out/run_01"],
                ["Summarize latency", "python eval_metrics/analyze_latency.py --realtime-log <realtime_csv> --carla-log <carla_csv> --output-json <latency_json> --output-csv <latency_joined_csv>"],
                ["Summarize drive metrics", "python eval_metrics/analyze_drive_metrics.py --log <carla_csv> --output-json <drive_summary_json>"],
                ["Assemble tables", "python eval_metrics/build_eval_tables.py --manifest eval_metrics/table_manifest.csv"],
            ],
            widths=[2200, 6600],
            header=True,
        ),
        p("Offline metrics are harvested from saved model bundles through eval_metrics/harvest_model_metrics.py. Latency results are generated by running manual_control_emg.py with --eval-log-dir and then analyzing the paired realtime and CARLA logs with eval_metrics/analyze_latency.py. CARLA scenario metrics are summarized from the per-tick drive log with eval_metrics/analyze_drive_metrics.py. Final tables can then be assembled with eval_metrics/build_eval_tables.py."),
        p("Results Placeholders", "Heading2"),
        _t(
            [
                ["Insert item", "Status"],
                ["Final offline results table", "Pending final experiment values"],
                ["Final latency table", "Pending logged scenario runs"],
                ["Final CARLA scenario table", "Pending lane-keeping and overtake runs"],
                ["Confusion matrix figure", "Pending selected model"],
                ["Scenario screenshots / figures", "Pending final run captures"],
            ],
            widths=[3200, 5600],
            header=True,
        ),
        p("Validation Discussion", "Heading2"),
        p("The discussion in this section should interpret whether the strict-layout redesign, runtime tuning, and scenario infrastructure materially improved the repeatability and deployment quality of the system. In particular, the final write-up should explain whether the neutral-recovery data path, ambiguity rejection, hysteresis, and CARLA-side dwell reduced instability without making steering overly sticky. Final claims here should be tied directly to the collected metric tables rather than anecdotal drive impressions."),
        p("Tools, Materials, Supplies, and Cost Analysis", "Heading1"),
        p("The main technical resources used by the project are the Delsys Trigno hardware stack, a Windows workstation capable of running the Delsys SDK and PyTorch pipeline, and a CARLA simulator installation for downstream testing. Software dependencies are largely open-source Python packages plus the Delsys SDK bridge and CARLA distribution."),
        p("[Insert the final actual-versus-estimated cost table here, using the Fall-term estimate as the baseline.]"),
        p("[List any consumables, hardware borrowing arrangements, and software licensing assumptions that materially affected project cost.]"),
        p("Reflection on Societal and Environmental Impact", "Heading1"),
        p("The positive societal case for this project is that EMG-based control could contribute to alternative human-machine interfaces for users who cannot rely on conventional manual interaction. The strongest accessibility value would come from robust operation under constrained motion, low fatigue, and realistic deployment conditions. At the same time, the project also raises safety and ethical questions because any control interface for driving requires high reliability and carefully bounded claims."),
        p("Environmental impact is modest at the prototype stage and is mainly associated with hardware use, compute, and simulator-based testing. The more important sustainability consideration is whether simulator-heavy iteration can reduce unnecessary physical testing during early development."),
        p("[Expand this section with your final ethical, safety, privacy, and sustainability reflections.]"),
        p("Reflection on Project Management Approach", "Heading1"),
        p("From a project-management standpoint, one of the strongest lessons in this capstone was that apparent model progress can be misleading if the acquisition and evaluation workflow are not yet stable. A significant amount of later engineering effort was therefore redirected from adding gestures to improving layout control, logging, and reproducibility. This was the correct tradeoff because it turned the project into a system that can be compared and debugged rather than a collection of loosely related scripts."),
        p("[Add the final schedule, milestone, communication, and risk-management reflection here.]"),
        p("Reflection on Economics and Cost Control", "Heading1"),
        p("The economic theme of the project is that engineering time and experimental quality were more constrained resources than raw hardware cost. Decisions such as preserving dedicated strict data roots, separating neutral-recovery data collection, and building reusable evaluation scripts increase short-term development effort but reduce the long-term cost of ambiguous results and repeated ad hoc testing."),
        p("[Add the final budget-control reflection and tradeoff discussion here.]"),
        p("Reflection on Life-Long Learning and Professional Growth", "Heading1"),
        p("This project required development across data acquisition, signal processing, neural network training, realtime systems, simulator integration, and evaluation design. One major professional takeaway is the importance of tightening the experimental contract before making broad performance claims. Another is that deployment quality often depends as much on data discipline and runtime post-processing as on the base classifier architecture itself."),
        p("[Add the final personal learning reflection here.]"),
        p("Conclusion and Recommendations", "Heading1"),
        p("The capstone delivered a technically coherent EMG control pipeline centered on strict sensor placement, CNN-based gesture classification, realtime flicker suppression, and scenario-based CARLA evaluation. The most important engineering outcome was not just a working classifier, but a more reproducible system whose data roots, layout contract, runtime behavior, and downstream metrics are explicitly controlled."),
        p("The clearest next steps are to freeze the final reported experiment set, insert the finalized offline, latency, and CARLA results, and then decide whether the next iteration should prioritize broader gesture vocabulary or deeper tuning and data collection around the already successful three-gesture deployment path."),
        p("References", "Heading1"),
        p("[Insert the final bibliography here, including the Basnet et al. 2025 paper and any additional references.]"),
        p("Appendix A. Commands Used for Metric Collection", "Heading1"),
        p("Offline metrics: python eval_metrics/harvest_model_metrics.py --models-root models/strict"),
        p("Latency runs: python carla_integration/manual_control_emg.py --scenario lane_keep_5min --eval-log-dir eval_metrics/out/run_01"),
        p("Latency summary: python eval_metrics/analyze_latency.py --realtime-log <realtime_csv> --carla-log <carla_csv> --output-json <latency_json> --output-csv <latency_joined_csv>"),
        p("Drive summary: python eval_metrics/analyze_drive_metrics.py --log <carla_csv> --output-json <drive_summary_json>"),
        p("Appendix B. Additional Figures and Tables", "Heading1"),
        p("[Insert supplemental confusion matrices, scenario screenshots, architecture diagrams, or extra tables here.]"),
    ]


def _research_paper_tex() -> str:
    # The paper draft is emitted as TeX so later edits stay diff-friendly and
    # can be moved into a venue template with minimal manual cleanup.
    return r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage{enumitem}

\title{[Paper Title]}
\author{[Author names and affiliations]}
\date{\today}

\begin{document}
\maketitle

\begin{abstract}
This draft presents the current capstone continuation of an EMG-based driving interface built around Delsys Trigno sensing, a strict sensor-placement workflow, convolutional neural network gesture classification, and CARLA-based downstream evaluation. The active implementation collects strict-layout EMG data through a custom Python GUI, resamples heterogeneous sensor streams to a common 2000 Hz time grid, filters the EMG signal with fixed notch and bandpass filters, and trains a residual 1D CNN with channel attention and an explicit energy bypass to preserve rest-versus-active amplitude cues. The deployed realtime path currently focuses on the three-gesture subset of \texttt{neutral}, \texttt{left\_turn}, and \texttt{right\_turn}, and applies smoothing, confidence gating, hysteresis, and ambiguity rejection to reduce neutral flicker. Evaluation is organized around three main groups of metrics: offline classification performance, end-to-end latency, and CARLA scenario performance. Final numerical results will be inserted once the experiment set is frozen.
\end{abstract}

\section{Introduction}
EMG-based human-machine interfaces remain attractive for driving-related control because they offer a potential pathway for non-traditional or accessibility-oriented control without requiring conventional manual interaction. In practice, however, an EMG driving interface is only useful if it is stable enough to move beyond isolated offline classification and into reproducible runtime behavior. This makes system discipline as important as raw classifier accuracy.

The current work continues earlier EMG-for-driving research by tightening the full experimental contract rather than simply increasing the number of gesture classes. Three engineering issues shaped the present design. First, inconsistent sensor placement made cross-session channel meaning unstable. Second, heterogeneous Delsys sensor sampling rates required explicit preprocessing before training and deployment. Third, neutral-class instability in realtime caused downstream steering flicker even when the base classifier appeared promising offline.

This paper therefore centers on a narrower and more defensible story: a strict-layout three-gesture EMG pipeline for driving-related control. The main contributions of the current system are:
\begin{itemize}[leftmargin=1.5em]
\item a fixed pair-number-specific strict sensor-placement workflow for both training and inference;
\item a preprocessing path that resamples multi-sensor EMG to a common time base and applies a mirrored offline/realtime filtering stack;
\item a residual 1D CNN with channel attention and an explicit pre-normalization energy bypass for improved neutral-versus-active separation;
\item a deployment path that combines classifier output with post-processing tuned to reduce neutral flicker; and
\item a CARLA evaluation harness with logged latency and scenario-level driving metrics.
\end{itemize}

At the time of writing, the cleanest reported scope is the strict-layout three-gesture setting consisting of \texttt{neutral}, \texttt{left\_turn}, and \texttt{right\_turn}. Broader gesture sets remain possible, but the three-gesture configuration is the narrowest version of the pipeline that aligns with the current deployed runtime and evaluation emphasis.

\section{Related Work}
This project builds most directly on the prior EMG driving-interface work by Basnet et al.~\cite{basnet2025}. The relationship to that work should be framed as continuation rather than replacement. The key difference in the current capstone branch is the shift toward a more controlled sensing and evaluation contract: fixed sensor placement, strict-layout dataset roots, explicit runtime stabilization, and downstream scenario logging.

The project also overlaps more broadly with literature on EMG gesture recognition, assistive control, and realtime biosignal classification. Within that space, the present work is intentionally focused on a practical engineering question: what system changes are required to make a previously fragile EMG-control prototype reproducible enough to support credible offline, latency, and simulator-level evaluation?

\section{System Overview}
The active system pipeline is collection $\rightarrow$ resampling $\rightarrow$ filtering $\rightarrow$ optional recalibration $\rightarrow$ windowing and label generation $\rightarrow$ CNN training $\rightarrow$ realtime post-processing $\rightarrow$ CARLA control evaluation. Data collection is performed with a custom Delsys GUI, preprocessing and training are implemented in Python and PyTorch, and downstream control experiments are conducted in CARLA through a simulator client that launches realtime inference internally.

The active top-level entrypoints are \texttt{DelsysPythonGUI.py} for collection, \texttt{tools/resample\_raw\_dataset.py} and \texttt{emg/filtering.py} for preprocessing, \texttt{train\_per\_subject.py} and \texttt{train\_cross\_subject.py} for model training, \texttt{realtime\_gesture\_cnn.py} for standalone inference, and \texttt{carla\_integration/manual\_control\_emg.py} for simulator runs. Strict raw collections are stored under \texttt{data\_strict/}, resampled strict data under \texttt{data\_resampled\_strict/}, and trained bundles under \texttt{models/strict/}.

\section{Methods}
\subsection{Data acquisition}
EMG acquisition is performed with Delsys Trigno hardware through a custom Python GUI that supports explicit sensor pairing, scanning of paired devices, live EMG viewing, and scripted labeled collection protocols. The standard collection protocol includes the labels \texttt{left\_turn}, \texttt{right\_turn}, \texttt{neutral}, \texttt{signal\_left}, \texttt{signal\_right}, and \texttt{horn}. In the current implementation, each active gesture lasts 5.0~s, each neutral segment lasts 5.0~s, five repetitions are recorded, and calibration is enabled with 5.0~s neutral-rest and 5.0~s MVC segments.

In addition to the standard protocol, the current branch includes a dedicated neutral-recovery protocol designed to capture turn-to-neutral transitions without modifying the balanced baseline collection workflow. This protocol records repeated \texttt{left\_turn} to \texttt{neutral} and \texttt{right\_turn} to \texttt{neutral} sequences using 3.0~s active holds and 5.0~s neutral holds. The protocol omits an explicit visible \texttt{neutral\_buffer} stage but still trims the leading release region internally before assigning settled neutral labels. This addition was motivated by deployment observations showing that the most important failure mode was instability immediately after returning to neutral.

If the reported experiments only use the three-gesture deployment subset, that should be stated explicitly as a training and evaluation restriction rather than a collection restriction. The collection pipeline itself still supports the broader active label set.

\subsection{Strict sensor-placement policy}
The strict sensor-placement workflow is central to the current method. Earlier mixed-layout assumptions were replaced because they allowed the semantic meaning of channel indices to drift between sessions. The active strict helper resolves channels by fixed pair identity and fails closed if required pair numbers are missing, duplicated, or inconsistent with the arm-specific layout contract.

For the right arm, pairs 1, 2, and 3 are assigned to the three Avanti slots, pair 7 to Maize, pair 9 to Galileo, and pair 11 to Mini. For the left arm, pairs 4, 5, and 6 are assigned to the three Avanti slots, pair 8 to Maize, and pair 10 to Galileo. This yields 17 strict channels on the right arm and 16 on the left. Channels are reordered by fixed pair identity during both training and inference.

\subsection{Preprocessing}
Different Delsys sensor types can stream at different effective sample rates, so raw multi-sensor sessions are first resampled to a common time base. The active resampler estimates per-channel sample rates from timestamp differences, finds the overlapping valid time interval across channels, constructs a uniform 2000~Hz grid, linearly interpolates each channel onto that grid, and transfers labels to the new grid by nearest-neighbor remapping.

After resampling, the active filtering stack applies a 60~Hz notch, a 120~Hz notch, and a 20--450~Hz sixth-order bandpass filter. The same filter family is mirrored in the realtime path so that deployment does not operate on a materially different signal than the one used during training. A recalibration utility is also available for sessions whose explicit MVC calibration is too weak; when the MVC-to-neutral quality ratio is insufficient, the tool can derive replacement neutral and scale terms from the labeled filtered session.

\subsection{Windowing and label generation}
The filtered EMG stream is segmented into overlapping 200-sample windows with a 100-sample step. At the 2000~Hz resampled rate, this corresponds to 100~ms windows with 50~ms stride. Each window is labeled by the majority class of its constituent samples. The inter-gesture rest label \texttt{neutral\_buffer} is retained during collection but excluded from model training. Windows with insufficient label purity can also be dropped through a configurable minimum label-confidence threshold.

\subsection{Model architecture}
The active classifier is \texttt{GestureCNNv2}, which consumes EMG windows shaped as $(\mathrm{channels}, \mathrm{time})$. The network applies input \texttt{InstanceNorm1d} followed by a residual 1D convolutional backbone with three stages at 32, 64, and 128 channels. Each stage includes squeeze-and-excitation channel attention, and the final embedding is global-average pooled before classification.

A key design feature is an explicit pre-normalization energy bypass. Before input normalization, the model computes the mean squared energy of the raw window and concatenates this scalar to the learned representation before the final linear classifier. The purpose of this bypass is to preserve amplitude information that would otherwise be attenuated by per-window normalization, particularly for separating rest-like neutral windows from active gesture windows.

\subsection{Training configuration}
The repository maintains both per-subject and cross-subject training paths. The per-subject configuration currently uses 60 epochs, batch size 512, Adam optimization with learning rate $10^{-4}$, dropout 0.25, label smoothing 0.05, and calibration-aware normalization when calibration quality passes threshold checks. Augmentation includes amplitude scaling, additive Gaussian noise, temporal shift, channel dropout, and temporal stretch. The active per-subject minimum label-confidence threshold is 0.85.

The cross-subject path uses the same general architecture but extends training to 80 epochs and uses subject-balanced sampling via \texttt{WeightedRandomSampler}. This distinction is important because subject-specific and pooled models answer different questions about personalization and generalization.

\subsection{Deployment and runtime post-processing}
The current deployment path uses \texttt{realtime\_gesture\_cnn.py} in dual-arm mode and restricts active output labels to the three-gesture subset \{\texttt{neutral}, \texttt{left\_turn}, \texttt{right\_turn}\}. Incoming streams are resampled to 2000~Hz, filtered using the same notch and bandpass stack as the offline path, and optionally normalized using saved neutral and MVC calibration terms.

The active runtime tuning preset is \texttt{flicker\_mild\_margin}. This preset applies smoothing of 3 frames, a minimum confidence threshold of 0.80, a dual-arm agreement threshold of 0.55, output hysteresis with two-frame confirmation, and softmax ambiguity rejection with minimum confidence 0.80 and minimum margin 0.10. These controls were added to address neutral flicker observed during live use. The paper should describe only the preset actually used for the reported experiments.

\section{Experimental Setup}
\subsection{Hardware and software}
The active software environment targets Python~3.10 with NumPy, SciPy, scikit-learn, libemg, PyTorch, PySide6, pygame, and the CARLA Python API. Live data collection expects Windows with the Delsys SDK bridge through \texttt{pythonnet}. The simulator client is implemented in CARLA through a modified manual-control interface that launches realtime inference internally.

\subsection{Dataset scope}
The narrowest defensible reported configuration on the current branch is the strict-layout three-gesture setting. In that configuration, collection may still use the broader label set, but training and deployment are restricted to \texttt{neutral}, \texttt{left\_turn}, and \texttt{right\_turn}. The final manuscript should explicitly state which subjects, arms, sessions, and strict dataset roots were used for the reported experiments.

\subsection{CARLA scenarios}
To support repeatable downstream evaluation, the simulator layer includes two scenario presets. The first is a checkpoint-based lane-keeping route that starts timing once the vehicle crosses a start checkpoint placed slightly ahead of spawn and finishes when the final visible checkpoint is reached. The second is an overtake scenario in which a slower lead vehicle is spawned ahead of the ego vehicle; success requires both satisfying the overtake objective and crossing the finish checkpoint. Both scenarios use a fixed sedan blueprint and write per-tick state to CSV logs, including checkpoint progress and completion state.

The current branch also exposes visible checkpoint markers and explicit finish conditions so that scenario completion is both machine-detectable and human-interpretable during a run. The final paper should report the exact frozen scenario configuration used for the final results.

\section{Evaluation Protocol}
The main reported metric set for this paper should match the capstone report. Prompted realtime behavior metrics can still be used as supplementary analysis, but the core evaluation should remain centered on offline model quality, latency, and downstream simulator behavior.

\subsection{Offline metrics}
The primary offline metrics are balanced accuracy and macro F1. Supporting metrics include accuracy, weighted F1, per-class recall, worst-class recall, the confusion matrix, confusion-to-neutral rate, and neutral prediction false-positive rate. These metrics are harvested directly from saved model bundles.

\subsection{Latency metrics}
Latency analysis joins realtime prediction logs and CARLA drive logs on a shared prediction identifier. The reported latency families are classifier latency, publish latency, control latency, and end-to-end latency. Each family should be summarized with mean, median, p90, p95, and maximum.

\subsection{CARLA scenario metrics}
The primary CARLA metrics are mean lane error, lane error RMSE, lane invasions, scenario success or failure, and scenario completion time. Lane error is defined as the distance from the vehicle to the nearest driving-lane centerline. Completion time is valid only for explicit scenarios and is measured from the start checkpoint to the scenario finish condition.

Metrics such as steering smoothness, command success rate, and collision count are deliberately omitted from the main metric set because they are either weakly grounded in the current pipeline or not central to the reportable experimental story.

\section{Results}
\subsection{Offline classification results}
[Insert the main offline results table here. The expected headline metrics are balanced accuracy and macro F1, with supporting confusion analysis.]

\subsection{Latency results}
[Insert the latency summary table here. Include classifier, publish, control, and end-to-end latency with mean, median, p90, p95, and max.]

\subsection{CARLA scenario results}
[Insert the lane-keeping and overtake scenario table here. Include mean lane error, lane error RMSE, lane invasions, success/failure, and completion time.]

\subsection{Comparison to prior work}
This section should compare only on metrics that are genuinely comparable to the prior paper. Eye-tracker-specific or otherwise unmatched measures should be excluded. The comparison should also make it clear which changes in the current work reflect a stricter sensing and evaluation contract rather than a direct architectural replacement.

\section{Discussion}
The main technical discussion should focus on whether stricter data discipline and runtime stabilization materially improved system credibility. In particular, the current branch suggests that layout control, preprocessing consistency, and post-processing behavior are at least as important to deployment quality as the underlying network architecture.

One recurring design tradeoff is between control breadth and reliability. The repository still supports a broader collection label set, but the deployed three-gesture subset remains the most stable control story. Another is the tradeoff between classifier flexibility and downstream interpretability: by freezing explicit scenarios, visible checkpoints, and shared metrics, the system becomes easier to evaluate even if some broader behaviors remain out of scope.

The final paper should also state the current limitations plainly. These may include limited subject count, the focus on simulator evaluation rather than real vehicles, the present emphasis on three-gesture deployment, and any scenario configurations that were still being tuned when experiments were collected.

\section{Conclusion}
This draft paper documents a strict-layout EMG control pipeline that connects Delsys-based collection, a reproducible preprocessing workflow, a residual 1D CNN with an energy bypass, runtime flicker suppression, and CARLA-based downstream evaluation. The clearest current contribution is not merely a working classifier, but a more defensible experimental system whose sensing contract, deployment behavior, and evaluation outputs are explicitly controlled.

The next step is to freeze the final experiment set, insert the corresponding offline, latency, and CARLA results, and then tighten the claims so that every reported conclusion maps directly to collected evidence.

\section*{Acknowledgments}
[Insert sponsor, supervisor, lab, and support acknowledgments here.]

\begin{thebibliography}{9}
\bibitem{basnet2025}
Basnet et al.
\newblock Evaluating the Feasibility of EMG-Based Human-Machine Interfaces for Driving.
\newblock 2025.
\newblock [Complete venue, volume, pages, and DOI to be inserted.]
\end{thebibliography}

\end{document}
"""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    final_docx = OUT_DIR / "final_report_draft_v2.docx"
    paper_tex = OUT_DIR / "research_paper_draft.tex"
    _docx_write(final_docx, "Final Design Report Draft", _final_report_draft())
    paper_tex.write_text(_research_paper_tex(), encoding="utf-8")
    print(final_docx)
    print(paper_tex)


if __name__ == "__main__":
    main()
