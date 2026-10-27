# Plan: reducing false positives per scan

## Goal

Move the full pipeline from today's operating point toward the range used by the official LUNA16 FROC metric
(0.125 to 8 false positives per scan) while keeping sensitivity high.

| | Now (50 scans of subset 9) | Milestone 1 | Milestone 2 | Target |
|---|---|---|---|---|
| False positives per scan | 140.7 | below 50 | below 20 | 8 or fewer |
| Sensitivity at that rate | 90.0% | at least 90% | at least 88% | at least 85% |

The milestones are goals, not results; each phase below is measured before the next one starts.

## Where the false positives come from

1. **Many Stage 1 candidates:** 639 per scan at threshold 0.10 with stride-16 windows. Stage 2 can only filter what Stage 1
   proposes, so every extra candidate costs Stage 2 precision.
2. **Single round of hard negatives:** Stage 2 learns from the false positives of one Stage 1 run, so it has not seen the
   structures a stronger Stage 1 would still confuse with nodules (vessels, airway walls, lymph nodes, scars).
3. **One patch scale:** a 64x64x64 patch at 1 mm shows 3 mm and 30 mm nodules at very different relative sizes.
4. **Duplicate detections:** one nodule can be hit by several overlapping windows; NMS at 10 voxels may leave neighbours.
5. **Small evaluation set:** 50 scans from one subset gives noisy rates; the official metric needs all 10 subsets.

## Phase 0: measure properly (1 to 2 days)

- Add FROC evaluation (`evaluation/froc.py`): sensitivity at 0.125, 0.25, 0.5, 1, 2, 4, 8 FP/scan and the CPM average,
  using the LUNA16 matching rule (a detection counts when it lies within the annotated nodule radius).
- 10-fold cross-validation by subset (train on 8, validate on 1, test on 1) so every scan is scored once.
- Record candidates per scan, FP per scan by size bucket and by structure type (vessel, airway, pleura, other), to see which
  false positives dominate.
- **Done when:** a baseline FROC table for the current models exists and is saved in `reports/`.

## Phase 1: cheaper wins at inference (2 to 3 days)

- Tune thresholds on validation data: Stage 1 threshold (0.10 now) and Stage 2 threshold (0.30 now) from the FROC curve
  instead of fixed values.
- Smarter NMS: merge detections with a radius proportional to the predicted nodule size, keep the highest probability, and
  average the probabilities of merged windows.
- Lung-mask validation: drop candidates outside the mask with a small margin; drop candidates under 3 mm equivalent diameter.
- Test-time augmentation for Stage 2 (flips and 90-degree rotations, average the probabilities).
- **Expected effect:** fewer duplicates and edge detections with no retraining; measure the change in FP/scan at equal sensitivity.

## Phase 2: better hard-negative mining (3 to 5 days)

- Iterate: Stage 1 -> mine false positives -> train Stage 2 -> run Stage 2 on the training scans -> add its remaining false
  positives to the negatives -> retrain (2 to 3 rounds).
- Stratify negatives by structure type and balance them against positives (start at 5:1 to 10:1) so the model does not
  become over-conservative.
- Keep patient-level splits for every round.

## Phase 3: richer input (4 to 6 days)

- Multi-scale patches (for example 64 at 1 mm, 32 at 2 mm, 20 at 4 mm) fed through parallel encoders and fused before the
  classifier head.
- Stronger 3D augmentation: elastic deformation, random affine, HU window jitter, Gaussian noise.
- Optional second view: 2.5D slices (three orthogonal planes) as an auxiliary input.

## Phase 4: model ensemble (3 to 4 days)

- Train 3 to 5 Stage 2 models (different seeds, folds or architectures such as 3D ResNet-18 or SE-ResNet) and average their
  probabilities.
- Calibrate the averaged probability (temperature scaling) before choosing the final threshold.

## Phase 5: improve the candidate generator (optional, 1 week)

- Replace or complement the classifier-on-windows Stage 1 with a detector that proposes fewer, better located candidates
  (for example a 3D U-Net or RetinaNet-style detector), so Stage 2 starts from tens of candidates per scan instead of hundreds.

## Order and decision rule

Run phases in order. After each phase recompute the FROC table; keep a change only if FP/scan drops at equal or better
sensitivity on the cross-validated scans. Stop when the target is reached.

## Deliverables

- `evaluation/froc.py`, the cross-validation driver and a results table per phase in `reports/`.
- Updated `scripts/06_full_pipeline_test.py` reading thresholds and NMS settings from one config.
- README results section refreshed after each milestone.

## Risks

- Hard-negative mining can overfit to one Stage 1 model; use cross-validated Stage 1 predictions for the mined negatives.
- Lowering FP too aggressively hurts sensitivity for small (3 to 6 mm) nodules; always read the FROC curve by nodule size.
- Compute: multi-scale and ensembles multiply training time; run Phase 1 and 2 first because they are the cheapest.
