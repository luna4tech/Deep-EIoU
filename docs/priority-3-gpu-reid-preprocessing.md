# Priority 3 — Move Re-ID Preprocessing to the GPU

## Problem

For every frame, the Re-ID extractor prepares each player crop **one at a time
on the CPU**, inside a Python loop.

In `FeatureExtractor.__call__`
(`Deep-EIoU/reid/torchreid/utils/feature_extractor.py:113`):

```python
for element in input:                 # one crop at a time
    image = self.to_pil(element)      # numpy array -> PIL image   (CPU)
    image = self.preprocess(image)    # PIL resize + ToTensor + Normalize (CPU)
    images.append(image)
images = torch.stack(images, dim=0)   # only now moved to GPU
```

A crowded sports frame has ~20+ players, so this does ~20+ serial
`numpy → PIL → resize → normalize` conversions **on the CPU** every frame.

While the CPU grinds through that loop, the **GPU sits idle**. This is a main
cause of the low SM utilization: the GPU is waiting on CPU preprocessing.

## Proposed Solution

Replace the per-crop PIL loop with **batched operations on the GPU**. No PIL, no
per-crop Python work on the critical path:

1. Move the raw crops to the GPU.
2. Resize each to 256×128 with `F.interpolate` (GPU).
3. Stack, then normalize the whole batch as one tensor (GPU).
4. Run OSNet on the batch.

### Sketch

```python
import torch.nn.functional as F

mean = torch.tensor([0.485, 0.456, 0.406], device='cuda').view(1, 3, 1, 1)
std  = torch.tensor([0.229, 0.224, 0.225], device='cuda').view(1, 3, 1, 1)

# crops: list of HxWxC numpy arrays (varying sizes)
tensors = [
    F.interpolate(
        torch.from_numpy(c).to('cuda').permute(2, 0, 1)[None].float(),
        size=(256, 128), mode='bilinear', align_corners=False
    )
    for c in crops
]
batch = (torch.cat(tensors) / 255.0 - mean) / std    # one normalized GPU batch

with torch.no_grad():
    feats = model(batch)
```

### Why a list comprehension is still fine here

Crops have **different sizes**, so they can't be stacked into one tensor before
resizing. Each is resized individually — but now on the **GPU**, not via CPU
PIL. The expensive resize/normalize/model work all runs on the GPU in one batch.
The remaining loop only issues lightweight GPU calls.

## Expected Impact

- Removes the serial CPU preprocessing that starves the GPU.
- Raises SM utilization — the GPU stops waiting on the CPU each frame.
- Unlocks the full benefit of Priority 2: batched Re-ID is only worthwhile if the
  crops are also prepared on the GPU, not one at a time on the CPU.

## Accuracy

**Negligible change.** `F.interpolate` bilinear resize differs from PIL's resize
only by sub-pixel rounding, well within feature noise. Cosine similarity between
embeddings — what the tracker actually uses — is unaffected. (Optionally run the
model in FP16 for extra speed; feature distances stay within matching tolerance.)

## Notes

- Keep crop extraction (`frame[y1:y2, x1:x2]`) as is; only the
  `numpy → PIL → CPU transforms` path changes.
- This change lives inside `FeatureExtractor`, so the demo call site
  (`demo.py:212`) does not need to change.
