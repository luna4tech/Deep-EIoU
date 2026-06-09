# Priority 2 — Batch Detection & Re-ID Across Frames

## Problem

The demo processes the video **one frame at a time**. The whole loop in
`Deep-EIoU/tools/demo.py` (the `while True` block, line 199) runs detection,
Re-ID, and tracking for a single frame before reading the next one.

The detector is hardcoded to batch size 1. In `Predictor.inference`
(`demo.py:168`):

```python
img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)  # batch = 1
```

Re-ID is the same — one frame's crops per call (`demo.py:212`).

This wastes the GPU:

- Each frame pays **fixed per-call overhead** (kernel launches, Python loop,
  CPU↔GPU copies). At batch 1 that overhead is paid ~30 times per second of
  video instead of being shared.
- The GPU is **idle most of the time** (low SM utilization) because one frame
  is not enough work to keep it busy.
- Only **2–3 GB of the 16 GB** VRAM is used. There is large unused headroom.

## Proposed Solution

Process frames in **batches of N** (start with N = 8):

1. Read and preprocess N frames into one tensor of shape `(N, 3, 800, 1440)`.
2. Run **one** YOLOX forward on the whole batch.
3. Collect crops from all N frames and run **one** Re-ID forward on them.
4. Run the tracker over the N frames **in original order** (see constraint).

### Why this is safe to do

- `postprocess()` already returns **a list with one detection set per image**,
  so YOLOX naturally supports batched input — the demo just never uses it.
- Every frame in the video has the **same resolution**, so all frames
  preprocess to the same tensor shape and stack cleanly (same `ratio`).

### Hard constraint: the tracker stays sequential

The tracker is **causal** — `tracker.update(det, embs)` (`demo.py:214`) depends
on the state left by the previous frame (Kalman predictions, track IDs, lost
track buffer). It **cannot** be batched. Only detection and Re-ID — which are
independent per frame — get batched. The tracker still runs frame by frame,
in order. Tracking output is unchanged.

### Sketch

```python
# 1. accumulate N frames
batch = torch.stack([preproc(f, test_size, mean, std)[0] for f in frames]).to(device)

# 2. one detector forward for all N frames
with torch.no_grad():
    dets = postprocess(model(batch), num_classes, conf, nms)   # list of N results

# 3. one Re-ID forward for all crops across the N frames
embs = extractor(all_crops)                                    # split back per frame

# 4. tracker runs in order — NOT batched
for det_i, emb_i in zip(dets, embs_per_frame):
    tracker.update(det_i, emb_i)
```

## Expected Impact

- Higher SM utilization — the GPU gets enough work to stay busy.
- Per-frame overhead amortized across N frames.
- Best combined with Priority 3 (so the Re-ID crops are also prepared on the
  GPU instead of one at a time on the CPU).

## Accuracy

**No change.** Batching does not alter the per-frame math; each frame produces
the same detections and features it would at batch 1. The tracker sees the same
inputs in the same order.

## Notes

- This adds latency (you wait to collect N frames), which is irrelevant here —
  the demo writes to a file offline, not live.
- VRAM at batch 1 is 2–3 GB, so batch 8 fits comfortably. Raise N until VRAM is
  well used or throughput stops improving.
