# Safety Taxonomy for VLM-Assisted Drone Assembly in ARISTOS

**Objective.** This document defines a safety taxonomy for an ARISTOS VLM-based monitor in first-person small-quadcopter assembly. The monitor receives egocentric video frames, the drone assembly manual, task graph, validation rules, task state, pedagogy goal, affective state, user profile, and, optionally, evidence from a digital-twin simulator. The taxonomy is written as a paper-style conceptual framework while remaining directly convertible into a label schema for annotation, training, and runtime evaluation.

**Central design decision.** The taxonomy should be organized first by **intervention policy**, not by a flat list of error types. Drone assembly permits valid deviations from the manual order, so procedural correctness should usually be checked only when the user asks. In contrast, actions that can cause human harm, irreversible component damage, or irreversible commitment to an unvalidated state require active monitoring.

The resulting top-level split is:

1. **Active irreversible safety monitor**: always-on; proactively interrupts when the current state or current action may cause human harm, irreversible device damage, or an irreversible unsafe commitment.
2. **On-demand flexible task-correctness monitor**: query-triggered; evaluates whether the user’s current step, part, placement, routing, or validation status is correct relative to the manual and task graph, without treating every valid alternative sequence as unsafe.

This is operationally preferable to a flat taxonomy such as *human safety / device safety / task correctness / irreversible damage*, because it specifies when the agent should interrupt.

---

## 1. Scope and Assumptions

### 1.1 Task boundary

The target task is **standard small quadcopter assembly**. Typical components include frame plates, arms, standoffs, spacers, screws, nuts, motor mounts, motors, motor wires, flight controller, ESC or power-distribution board, receiver, camera or sensor module, connectors, cable ties, heat-shrink, and optional adhesive or threadlocker.

This version **excludes powered testing**, including battery connection for live testing, motor spin-up, arming, firmware calibration under power, propeller-driven hazards, and flight testing. However, unpowered assembly can still create irreversible risks, such as soldering-iron burns, damaged PCB pads, connector-pin deformation, ESD or EOS latent damage, crushed wires, stripped screw holes, and physically damaged battery pouches.

### 1.2 Available evidence

The monitor may use the following sources:

- **Egocentric video frames or short clips** for hand-object relations, tool proximity, part pose, visible deformation, occlusion, and workspace hazards.
- **Drone assembly manual** for part identity, nominal orientation, keyed connectors, validation steps, and irreversible-operation checkpoints.
- **Task Graph** for flexible partial order, prerequisites, acceptable alternative paths, and commit points.
- **Validation Steps / Error Checking Setup** for explicit pass/fail rules before fastening, soldering, cutting, gluing, or locking.
- **Task State** for completed steps, known errors, unresolved warnings, and prior irreversible operations.
- **Pedagogy Goal** for feedback timing on reversible correctness errors, but not for suppressing active safety interventions.
- **Affective State** and **User Profile** for response style, explanation length, and conservative thresholds for novice users, but not for redefining safety truth.
- **Digital twin / physical simulation** as a separately marked evidence source for rare negative examples, geometric interference, clearance violation, and counterfactual consequence checks.

### 1.3 Runtime output

The safety monitor’s runtime decision is binary:

```text
Should the system intervene now?  yes / no
```

Internal labels may record the taxonomy ID, risk target, reversibility, confidence, and evidence source, but the runtime policy should not be framed as a future-step planner. The monitor answers whether the current state or action requires intervention.

---

## 2. Related Work Synthesis

VLESA frames embodied safety as an egocentric, intent-conditioned monitoring problem. Its key abstraction is a tuple of image, candidate action, task goal, and binary safety label. This supports an action- and context-conditioned formulation for ARISTOS: the relevant question is not whether a frame is generically dangerous, but whether the observed assembly action is unsafe under the current task state, manual constraint, and intended operation.

Semantic robot-safety work, including robot constitutions and ASIMOV-style benchmarks, motivates a rule-based layer over VLM perception. These works are useful because they treat safety as contextual and compositional rather than as a fixed object detector. For drone assembly, however, broad robot-safety categories must be translated into concrete mechanisms: thermal exposure, fragile electronics, ESD or EOS risk, connector and cable damage, over-force, wrong fastener length, contamination, and irreversible operations before validation.

Egocentric procedural-assistance datasets and methods, including HoloAssist, EgoPER, PREGO, Assembly101, EgoOops, EASG, and Ego-Exo4D, support a separate procedural-correctness channel. They show that mistakes can be represented as omission, addition, modification, slip, correction, ordering violation, or action-scene-graph deviation. These labels are useful for ARISTOS, but they do not by themselves imply active safety intervention. A valid non-manual order can be pedagogically acceptable; a reversible procedural error should not become a proactive safety alert unless it crosses a commit point.

Industrial safety standards provide additional structure. ISO 12100 motivates hazard identification and risk estimation using harm severity and probability. IEC 62368-1 motivates reasoning from hazardous energy sources and safeguards. In the present unpowered drone-assembly scope, the relevant energy and harm mechanisms are mainly thermal, mechanical, chemical or fume exposure, latent electronic damage, and battery physical damage, rather than powered motor or live-electrical hazards.

The digital twin should be treated as an evidence source rather than as a taxonomy class. It is valuable because real unsafe drone-assembly data will be sparse, and simulation can create or evaluate rare counterfactuals such as cable pinch, screw-length collision, connector misalignment, board-standoff interference, and clearance violation. Any label that depends on simulation must be explicitly marked as `sim_derived` to avoid conflating simulated evidence with real egocentric evidence.

---

## 3. Taxonomy Design Principles

### P1. Intervention policy is the top-level split

The primary operational question is whether ARISTOS should interrupt the learner. Therefore, the taxonomy begins with:

- `active_safety`: always-on monitoring of irreversible safety risks;
- `on_demand_correctness`: user-query or checkpoint-triggered monitoring of flexible procedural correctness.

### P2. Reversibility is central

Each label should encode whether the consequence is reversible:

- `irreversible`: damage or unsafe commitment has occurred or is nearly inevitable;
- `potentially_irreversible`: continuing the current action is likely to create irreversible damage;
- `reversible`: the state can be corrected without damage;
- `unknown`: evidence is insufficient.

Active safety primarily covers `irreversible` and `potentially_irreversible` cases.

### P3. Correctness is not equivalent to safety

A user may assemble in an order different from the manual and still remain safe. Procedural deviation becomes active safety only when it is about to be fixed by soldering, gluing, cutting, force fitting, final fastening, threadlocking, heat-shrink closure, or another difficult-to-reverse operation.

### P4. Pedagogy cannot override irreversible safety

Pedagogy may delay feedback for reversible correctness mistakes. It must not delay intervention for likely human injury, irreversible device damage, or irreversible commitment to an unvalidated physical state.

### P5. Simulation provenance must be explicit

Digital-twin evidence should be encoded as `sim_derived`. Reports and benchmarks should separately track real-only, simulation-only, and mixed-source performance.

---

## 4. Operational Taxonomy

### 4.1 Monitor modes

| Monitor mode | Trigger policy | Objective | Proactive interruption | Example |
|---|---|---|---|---|
| `active_safety` | Continuous | Prevent human harm, irreversible device damage, and irreversible unsafe commitments | Yes | Hot tool near fingers or PCB cable; connector forced while misaligned; cable being clamped under frame plate |
| `on_demand_correctness` | User question, checkpoint request, or formal evaluation | Determine whether the current assembly state is correct relative to manual, task graph, and validation rules | No, unless upgraded to safety | “Is this board orientation correct?”; “Can I mount this arm before routing the wire?” |

---

## 5. Active Irreversible Safety Taxonomy

These categories are continuously monitored. If the evidence is sufficient and the risk is `irreversible` or `potentially_irreversible`, the monitor should set `should_intervene = true`.

| ID | Category | Failure mechanism | Definition | Evidence cues | Drone assembly example | Digital-twin role |
|---|---|---|---|---|---|---|
| `S-H1` | Human safety | Thermal contact or burn | A hot tool, hot surface, or heated material may contact the user or nearby material. | Soldering iron or heat tool near fingers, wires, plastic, or unsupported work surface. | The user holds a wire close to a hot soldering tip. | Mostly visual; simulation can augment negative scenes. |
| `S-H2` | Human safety | Sharp, pinch, or puncture hazard | Tools, frame edges, screws, or sliding parts may cut, pinch, or puncture skin. | Tool force directed toward hand; fingers near sharp carbon edge or pinch point; unstable grasp. | The user pries a frame slot with a tool aimed toward the supporting hand. | Limited; useful for geometry and contact-path augmentation. |
| `S-H3` | Human safety | Chemical, fume, or residue exposure | Flux, adhesive, cleaner, smoke, or residue creates user or workspace exposure risk. | Visible fumes, excess adhesive, flux residue, uncontained chemical material. | Soldering fumes accumulate near the user’s face; adhesive spreads onto hands or sensors. | Low; mainly visual and procedural. |
| `S-H4` | Human/workspace safety | Battery physical damage | An unpowered battery pouch or cell may be crushed, punctured, bent, heated, or clamped. | Battery near sharp screw, tool tip, hot tool, tight strap, or frame edge. | A screw or frame plate presses into the battery pouch during mounting. | High for clearance and pinch prediction. |
| `S-D1` | Device safety | ESD or EOS latent damage | ESD-sensitive electronics may be damaged by unsafe handling or tool conditions. | Bare PCB on fabric/plastic/foam; direct contact with ICs or sensor area; absent ESD-safe context when risk is salient. | Flight controller placed on clothing while the user handles the IMU/MCU region. | Useful for synthetic context variation; weak for runtime certainty. |
| `S-D2` | Device safety | Thermal component damage | Heat may damage PCB pads, connectors, wire insulation, plastic parts, ribbon cables, or sensor modules. | Hot tool dwell near small pads; insulation deformation; heat tool near connector or cable. | Motor-wire soldering overheats a PCB pad or adjacent connector. | Moderate; actual damage remains visually and process dependent. |
| `S-D3` | Device safety | Mechanical overstress or wrong force direction | Excessive force, misaligned force, bending, prying, or pulling may permanently damage board, frame, cable, or part. | Component bending, tool prying, cable tension, forceful press-fit, misaligned screw. | User forces a flight controller onto offset standoffs or pulls a motor wire by the cable. | High for collision, clearance, and force-direction checks. |
| `S-D4` | Device safety | Connector or pin damage | A connector, pin, latch, or socket may be bent, cracked, or damaged by misalignment or wrong orientation. | Connector not aligned with keyed socket; plug reversed; ribbon cable skewed; pressure continues. | JST plug is pushed into the wrong orientation; ribbon cable is clamped while misaligned. | High for connector geometry and insertion interference. |
| `S-D5` | Device safety | Foreign object, conductive debris, or contamination | Debris, solder blobs, wire strands, loose screws, flux, adhesive, or residue may cause short, blockage, or later failure. | Loose conductive object on PCB; solder bridge; wire strand near pads; adhesive in motor or connector. | A cut wire strand falls onto the flight-controller board. | Moderate; useful for hidden-object and debris scenarios. |
| `S-D6` | Device/task safety | Irreversible wrong placement before commit | A currently incorrect or unvalidated state is about to be made hard to reverse. | Manual/task-graph mismatch plus soldering, gluing, cutting, heat-shrink closure, threadlocking, or final fastening. | Flight-controller arrow points wrong way while the user begins final fastening or soldering dependent wires. | High for counterfactual completion and interference checks. |
| `S-D7` | Device safety | Fastener, thread, or mount damage | Wrong screw, wrong hole, missing spacer, cross-threading, or over-tightening may strip, crack, or crush parts. | Screw enters at angle; board bends under screw head; spacer missing; screw length incompatible. | Long motor screw reaches internal winding; board is fastened without required spacer. | Very high for screw length, clearance, and collision checks. |
| `S-W1` | Workspace safety | Fire, heat transfer, or unstable hot tool | Hot tools, flammable material, plastic packaging, loose wires, or clutter produce environmental risk. | Hot tool placed on paper, foam, plastic, cable, or unstable support. | Soldering iron rests near foam packaging or wire insulation. | Mostly visual; simulation can diversify scenes. |
| `S-W2` | Workspace/process safety | Occlusion or instability during hazardous action | The system cannot verify safety because the critical contact point is occluded or the workpiece is unstable during a hazardous operation. | Hand blocks soldering area; PCB is unsupported; single-hand free-air soldering; unstable part during forceful insertion. | User inserts a connector while the socket and pins are occluded. | Moderate; useful for checking whether unobserved geometry is risk-critical. |

---

## 6. On-Demand Flexible Task-Correctness Taxonomy

These categories are not proactive alerts by default. They are evaluated when the user asks a correctness question, when a checkpoint is requested, or when the system is in formal evaluation mode. They upgrade to active safety only at commit points or when damage becomes likely.

| ID | Category | Error mechanism | Definition | Evidence cues | Example | Upgrade rule |
|---|---|---|---|---|---|---|
| `C-T1` | Task correctness | Step omission | A prerequisite step is missing before the queried current step. | Task State lacks prerequisite; validation step incomplete; visual state inconsistent. | User places flight controller before installing required standoffs. | Upgrade if continued fastening may bend or crush the board. |
| `C-T2` | Task correctness | Extra or unnecessary step | The user performs a reversible step not required by the current graph path. | Action not on current task path; no damage evidence. | User repeatedly removes and reinstalls an arm without need. | Upgrade if repeated action begins damaging threads or mounts. |
| `C-T3` | Task correctness | Step modification | The user performs a similar step with a different object, method, order, or parameter. | Current object/action differs from manual or task graph. | User uses a different screw type or routes the cable before the expected spacer. | Upgrade if wrong method is forced, soldered, glued, or locked. |
| `C-T4` | Task correctness | Wrong part or hardware | Wrong screw, standoff, connector, arm side, motor position, or hardware is selected. | Visual part ID or bill-of-material mismatch. | User selects an overly long motor screw or the wrong arm side. | Upgrade if the part may damage motor, frame, connector, or PCB. |
| `C-T5` | Task correctness | Reversible orientation or placement mismatch | Part orientation, front/back direction, vertical direction, hole alignment, or cable-exit direction differs from the reference but remains correctable. | Manual visual reference conflicts with current pose; part is not yet fixed. | Flight-controller arrow is temporarily reversed while the board is loose. | Upgrade when final fastening, soldering, gluing, or heat-shrink begins. |
| `C-T6` | Task correctness | Reversible fastening-quality issue | Attachment is loose, uneven, missing, or incomplete without current damage. | Visible gap, loose fastener, missing washer/spacer, incomplete screw seating. | One motor screw is absent; frame plate is not yet evenly seated. | Upgrade if continued tightening causes cross-threading, cracking, or board deflection. |
| `C-T7` | Task correctness | Reversible cable-routing or clearance issue | Wire or soft part is routed differently from the manual but is still movable and undamaged. | Cable crosses a screw path; wire blocks connector access; slack is non-nominal. | Motor wire lies over an arm screw hole before final assembly. | Upgrade if cable is pinched, sharply creased, cut, heated, or locked under hardware. |
| `C-T8` | Task correctness | Missing validation or unresolved uncertainty | Required check is absent, state is occluded, or evidence conflicts. | Validation not logged; key part is hidden; Task State and video disagree. | System cannot confirm whether connector latch is seated. | Upgrade if the next action would make the uncertain state irreversible. |

---

## 7. Boundary and Upgrade Rules

### Rule 1: Active safety overrides pedagogy

If the label is an `S-*` class and reversibility is `irreversible` or `potentially_irreversible`, then:

```text
should_intervene = true
```

This holds even if the pedagogy goal would otherwise allow the learner to explore or make reversible mistakes.

### Rule 2: Reversible correctness is not proactive

If the label is a `C-*` class, no user query or checkpoint is active, and no commit point is imminent, then:

```text
should_intervene = false
```

The system may record the state internally but should not interrupt merely because the action differs from the manual order.

### Rule 3: Correctness upgrades to safety at commit points

A `C-*` issue upgrades to `S-D6` or the most relevant `S-*` class when the user begins or is about to begin any of the following:

- soldering;
- gluing or adhesive application;
- cutting or trimming;
- heat-shrink closure;
- permanent zip-tie locking;
- threadlocking or final fastening;
- force fitting, prying, or hard insertion;
- any action predicted by the digital twin to create collision, cable pinch, clearance violation, or damaging stress.

### Rule 4: Insufficient observability can itself require intervention

If the system cannot verify a critical contact point during a hazardous action, it should use `S-W2` with low or medium confidence rather than asserting a specific damage class.

Example:

```json
{
  "should_intervene": true,
  "taxonomy_id": "S-W2",
  "reversibility": "potentially_irreversible",
  "confidence": "low"
}
```

### Rule 5: Simulation-derived labels must remain separate

Any label depending on simulated history or counterfactual simulation must include:

```json
"sim_derived": true
```

Evaluation should report real-only, simulation-only, mixed-source, and sim-to-real failure cases separately.

---

## 8. Minimal Label Schema

The label schema should be small enough to fit as a subrecord under `Task State` or `Validation Steps / Error Checking Setup`.

### 8.1 Runtime output schema

```json
{
  "monitor_mode": "active_safety | on_demand_correctness",
  "should_intervene": true,
  "taxonomy_id": "S-D4",
  "risk_category": "device_safety",
  "reversibility": "potentially_irreversible",
  "severity": "major",
  "confidence": "medium",
  "evidence_source": ["video_frame", "task_state", "manual_rule"],
  "sim_derived": false
}
```

Recommended field constraints:

- `monitor_mode`: `active_safety` or `on_demand_correctness`.
- `should_intervene`: binary runtime decision.
- `taxonomy_id`: one of the closed-set IDs in Sections 5 and 6.
- `risk_category`: `human_safety`, `device_safety`, `workspace_safety`, or `task_correctness`.
- `reversibility`: `irreversible`, `potentially_irreversible`, `reversible`, or `unknown`.
- `severity`: `critical`, `major`, `minor`, or `none`.
- `confidence`: `high`, `medium`, or `low`.
- `evidence_source`: subset of `video_frame`, `manual_rule`, `task_graph`, `task_state`, `validation_step`, `digital_twin_sim`.
- `sim_derived`: true when simulation materially supports the label.

### 8.2 Annotation schema

```json
{
  "sample_id": "drone_asm_000123",
  "frame_or_clip_id": "clip_04_t012.5",
  "task_step_id": "mount_flight_controller",
  "observed_action": "user presses the flight controller onto the standoffs",
  "monitor_mode": "active_safety",
  "label": "intervene",
  "taxonomy_id": "S-D3",
  "evidence_source": ["video_frame", "task_graph"],
  "sim_derived": false
}
```

The `observed_action` field is important because a VLM safety monitor should evaluate `(frame, action, task context)`, not only a static image.

---

## 9. Mapping to the ARISTOS Data Structure

| Existing field | Role in taxonomy | Example use |
|---|---|---|
| `Task Graph` | Defines flexible order, prerequisites, accepted alternatives, and commit points. | Board must be supported before final fastening; keyed connector orientation must match. |
| `Validation Steps / Error Checking Setup` | Provides checkpoint rules and upgrade conditions. | Validate cable routing before closing frame; check screw length before motor mounting. |
| `Task State.Completed Steps` | Determines whether prerequisites or irreversible operations have already occurred. | If wires are already soldered, wrong wire order has lower reversibility. |
| `Task State.Errors` | Tracks unresolved errors and whether they have been corrected. | Known wrong cable route remains unresolved before frame closure. |
| `Pedagogy Goal` | Controls timing for reversible correctness feedback. | Let user explore orientation while the board is loose; never delay a thermal or irreversible damage alert. |
| `Affective State` | Controls wording and length of explanation. | For a frustrated user, provide a short stop message and minimal rationale. |
| `User Profile` | Adjusts explanation depth and conservative thresholds. | Novices may receive earlier low-confidence warnings during soldering or force insertion. |
| `Digital Twin` | Provides `sim_derived` evidence for rare negatives and hidden consequences. | Simulate whether a long screw intersects a motor winding or whether a cable will be pinched by the top plate. |

---

## 10. Digital-Twin Usage Policy

Digital-twin evidence is optional but important because unsafe drone-assembly data will be scarce. It should be used in three roles:

| Use case | Label treatment | Appropriate scope |
|---|---|---|
| Simulated history | `sim_derived = true` if the current state originates from simulated prior actions. | Training state trackers and debugging task-state transitions. |
| Counterfactual future simulation | `sim_derived = true` if the label depends on simulated continuation of the current action. | Cable pinch, screw collision, connector interference, insufficient clearance, wrong part fit. |
| Synthetic negative data | `sim_derived = true`; keep separate train/evaluation splits. | Rare unsafe examples for active safety classes. |

Simulation is most reliable for geometry, clearance, collision, and screw-length checks when CAD and material assumptions are adequate. It is less reliable for solder-joint quality, ESD, soft cable fatigue, fume exposure, and human visual behavior. These cases require real video, expert review, or manual constraints.

---

## 11. Dataset and Benchmark Implications

### 11.1 Data unit

Recommended sample format:

```text
(frame or short clip, observed action, task context, taxonomy label, intervention label)
```

The task context should include current task node, completed steps, unresolved errors, relevant manual snippet, and validation rule.

### 11.2 Label sources

| Source | Suitable labels | Trust treatment |
|---|---|---|
| Real egocentric drone assembly videos | Normal assembly, natural errors, real occlusion, real tool use | Highest priority |
| Manual + task-graph perturbations | Wrong order, wrong part, wrong orientation, missing validation | Requires expert review |
| Digital twin | Clearance, collision, screw length, cable pinch, hidden interference | Must be `sim_derived` |
| VLM-generated unsafe variants | Rare negative expansion | Requires rule validation and human spot checks |
| Expert annotation | Gold safety labels | Use for calibration and final evaluation |

### 11.3 Metrics

Safety-monitor evaluation should not rely only on accuracy. Recommended metrics are:

- `Unsafe Recall`: fraction of active safety risks correctly detected.
- `False Active Interruption Rate`: rate of proactive interruptions on reversible correctness issues.
- `Commit-Point Catch Rate`: fraction of risks detected before soldering, gluing, cutting, force fitting, or final locking completes.
- `Alert Timing`: margin between alert and irreversible action completion.
- `Sim-to-Real Transfer Gap`: performance difference between simulation-derived and real data.
- `Evidence Calibration`: agreement between confidence and correctness.
- `Category Confusion`: especially among `S-D2`, `S-D3`, `S-D4`, `S-D6`, and `S-D7`.

---

## 12. Recommended Label Sets

### 12.1 Fine-grained candidate set

**Active safety classes**

1. `S-H1`: Human thermal contact or burn risk
2. `S-H2`: Human sharp, pinch, or puncture risk
3. `S-H3`: Human chemical, fume, or residue exposure
4. `S-H4`: Battery physical damage risk in the unpowered stage
5. `S-D1`: ESD or EOS latent component damage
6. `S-D2`: Thermal component damage
7. `S-D3`: Mechanical overstress or wrong force direction
8. `S-D4`: Connector or pin damage
9. `S-D5`: Foreign object, conductive debris, or contamination
10. `S-D6`: Irreversible wrong placement before commit
11. `S-D7`: Fastener, thread, or mount damage
12. `S-W1`: Workspace fire, heat transfer, or unstable hot tool
13. `S-W2`: Occlusion or instability during hazardous action

**On-demand correctness classes**

1. `C-T1`: Step omission
2. `C-T2`: Extra or unnecessary step
3. `C-T3`: Step modification
4. `C-T4`: Wrong part or hardware
5. `C-T5`: Reversible orientation or placement mismatch
6. `C-T6`: Reversible fastening-quality issue
7. `C-T7`: Reversible cable-routing or clearance issue
8. `C-T8`: Missing validation or unresolved uncertainty

This 21-class set is appropriate for paper description, error analysis, and mature annotation.

### 12.2 Coarse initial annotation set

For early data collection, a coarser set is preferable:

| Coarse ID | Meaning | Fine labels covered |
|---|---|---|
| `S-H-THERMAL` | Human thermal risk | `S-H1` |
| `S-H-MECHANICAL` | Human sharp, pinch, or puncture risk | `S-H2` |
| `S-H-MATERIAL` | Human material, fume, residue, or battery physical risk | `S-H3`, `S-H4` |
| `S-D-ESD` | ESD or EOS latent damage | `S-D1` |
| `S-D-THERMAL` | Device thermal damage | `S-D2` |
| `S-D-MECHANICAL` | Device mechanical damage | `S-D3`, `S-D7` |
| `S-D-CONNECTOR` | Connector, pin, latch, or cable-interface damage | `S-D4` |
| `S-D-CONTAMINATION` | Foreign object, debris, or contamination | `S-D5` |
| `S-D-COMMIT` | Incorrect or unvalidated state becoming irreversible | `S-D6` |
| `S-W-WORKSPACE` | Workspace or observability hazard | `S-W1`, `S-W2` |
| `C-T-CORRECTNESS` | Query-triggered task correctness | `C-T1`–`C-T8` |

This 11-class set is recommended for v0.1 implementation. Fine labels can be retained in annotator notes and promoted to first-class labels when data volume supports them.

---

## 13. VLM Monitor Formulation

The VLM should not receive only the image. It should receive a compact multimodal context:

```text
Input:
- Egocentric frame window
- Observed/current action
- Current task graph node and completed steps
- Relevant manual or validation rule
- User query, if present
- Optional digital-twin state or counterfactual result, explicitly marked

Output:
- JSON label using the minimal schema
- Runtime decision: intervene / no_intervene
```

The monitor should first distinguish active safety from on-demand correctness, then assign taxonomy ID, reversibility, confidence, and evidence source. The deployed interface may expose only the binary decision and a short safety reason.

VLM-only monitoring is insufficient for several drone-assembly hazards: screw length and hidden clearance, connector keying, wire pinch under frame plates, ESD, torque, and force are often not fully visible. Therefore, the safety monitor should combine VLM perception with manual-derived symbolic rules, task-state checks, validation rules, and digital-twin checks when available.

---

## 14. Source Notes

### Project-specific sources

- User-provided ARISTOS proposal: defines ARISTOS as a multimodal teacher for physical reskilling, includes continuous safety monitoring for the learner and device, and proposes simulation for drone repair and incorrectly assembled components.
- User-provided VLESA draft: defines egocentric video safety monitoring, intent-action prediction, goal-conditioned safety Q-filtering, and image-action-goal-safety labels.

### External sources reviewed

1. Sermanet et al. **Generating Robot Constitutions & Benchmarks for Semantic Safety**. PMLR / ASIMOV Benchmark. <https://proceedings.mlr.press/v305/sermanet25a.html>
2. Jindal et al. **Can AI Perceive Physical Danger and Intervene?** ASIMOV-2.0. <https://arxiv.org/abs/2509.21651>
3. Wang et al. **HoloAssist: An Egocentric Human Interaction Dataset for Interactive AI Assistants in the Real World**. ICCV 2023. <https://holoassist.github.io/>
4. Lee et al. **Error Detection in Egocentric Procedural Task Videos**. CVPR 2024 / EgoPER. <https://openaccess.thecvf.com/content/CVPR2024/papers/Lee_Error_Detection_in_Egocentric_Procedural_Task_Videos_CVPR_2024_paper.pdf>
5. Flaborea et al. **PREGO: Online Mistake Detection in Procedural Egocentric Videos**. <https://arxiv.org/abs/2404.01933>
6. **Assembly101** dataset. <https://assembly-101.github.io/>
7. Haneji et al. **EgoOops: A Dataset for Mistake Action Detection from Egocentric Procedural Videos**. <https://arxiv.org/abs/2410.05343>
8. Rodin et al. **Action Scene Graphs for Long-Form Understanding of Egocentric Videos**. CVPR 2024. <https://arxiv.org/abs/2312.03391>
9. Grauman et al. **Ego-Exo4D: Understanding Skilled Human Activity from First- and Third-Person Perspectives**. CVPR 2024. <https://ego-exo4d-data.org/>
10. ISO 12100:2010. **Safety of machinery — General principles for design — Risk assessment and risk reduction**. <https://www.iso.org/standard/51528.html>
11. IEC 62368-1 overview: hazard-based safety engineering for audio/video, information, and communication technology equipment. <https://cetecomadvanced.com/en/news/eniec-62368-1-hazard-based-safety-engineering-in-focus/>
12. ANSI/ESD S13.1 and S20.20 source notes for soldering/desoldering hand-tool and ESD-control relevance. <https://www.esda.org/esd-overview/esd-fundamentals/part-6-esd-standards/>
