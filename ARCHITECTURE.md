# ARCHITECTURE.md

## 1. Model Selection Defense

### Choice: Qwen2-VL-2B-Instruct

Qwen2-VL-2B was selected over LLaVA-NeXT-Video-7B and VideoLLaMA2-7B for the following reasons:

| Model | Parameters | Training VRAM (4-bit QLoRA) | Fits Kaggle T4 (16GB)? | Notes |
|---|---|---|---|---|
| **Qwen2-VL-2B** | 2B | ~5–7 GB | ✅ Yes | Selected |
| LLaVA-NeXT-Video-7B | 7B | ~10–14 GB | ⚠️ Tight | Risk of OOM |
| VideoLLaMA2-7B | 7B | ~12–15 GB | ❌ Likely OOM | Less documented |

**Why Qwen2-VL-2B specifically:**
- Smallest VRAM footprint — comfortably fits on a single T4 with headroom for activations
- Native multi-image input support via `qwen_vl_utils`, which maps cleanly to the multi-frame clip format required by OpenPack
- Well-documented fine-tuning ecosystem (HuggingFace `transformers` + `peft` + `trl`)
- Unsloth compatibility exists as an alternative quantization path — I chose BitsAndBytes (`bnb_4bit_quant_type="nf4"`) directly since it integrates cleanly with HuggingFace `Trainer` without additional dependencies

**Why not Unsloth:**
Unsloth would have reduced VRAM further (~30% savings) and sped up training, but adds a dependency that complicates reproducibility on vanilla Kaggle kernels. BitsAndBytes QLoRA with `bnb_4bit_compute_dtype=bfloat16` and double quantization achieved acceptable VRAM usage without it.

---

## 2. Frame Sampling Strategy

### Strategy: Entropy-Based Keyframe Selection with Uniform Fallback

Implemented in `data_pipeline.py` → `entropy_keyframe_indices()` (line 215).

The clip (125 frames at 25fps over 5 seconds) is divided into `N=8` equal windows. Within each window, the frame with the highest pixel-level entropy is selected:

```python
hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 255))
p = hist[hist > 0].astype(float) / hist.sum()
e = float(-np.sum(p * np.log2(p)))   # Shannon entropy
```

The frame with the highest `e` per window is selected. Entropy spikes naturally at operation boundaries because the pixel distribution changes — hands repositioning, objects moving, background shifting — producing a more uniform histogram than a static mid-operation frame.

**Why entropy beats uniform sampling for this task:**

Uniform sampling at 25fps over 5 seconds picks one frame every ~16 frames. For a clip straddling a boundary (e.g., "Tape" → "Label"), the transition moment may fall between two uniformly-spaced samples entirely. Entropy-based selection is attracted to these high-variation frames rather than ignoring them.

```
Clip:    [=====Tape==========|=========Label=====]
                              ↑ boundary (high entropy)

Uniform:  ^    ^    ^    ^    ^    ^    ^    ^
                              (may or may not land here)

Entropy:  ^  ^      ^      ^ ^^      ^       ^
                              (pulled toward boundary spike)
```

**Clips are centered on boundaries, not mid-operation:**

`build_clip_from_boundary()` (line 251) extracts a 5-second window centered on the detected label-change frame (`boundary_frame ± half`). This means every training sample straddles a transition, forcing the model to see both the ending and beginning operation in the same context window — which is precisely what trains the `anticipated_next_operation` head.

**Fallback behaviour:**

When RGB frames are not available locally (e.g., running annotation-only validation without the 48GB video files), `entropy_keyframe_indices()` falls back to uniform stride automatically:

```python
if n_total <= n_select or frame_loader_fn is None:
    step = max(1, n_total // n_select)
    return [min(i * step, n_total - 1) for i in range(n_select)]
```

This means the pipeline runs correctly in both modes — full entropy on Kaggle/GCP with RGB frames, uniform stride for annotation-only dry runs locally.

---

## 3. Failure Mode Analysis

### Root Cause: Dataset Access Constraint

The most significant failure in this pipeline is not a model architecture choice — it is a data access failure.

**What was intended:**
- Download OpenPack annotation metadata from Zenodo (manageable size)
- Download Kinect frontal RGB videos from Google Drive (subjects U0101–U0108)
- Run `data_pipeline.py` to extract boundary-centered clips, sample frames via entropy, and generate training pairs
- Fine-tune on thousands of real annotated clips across 6 subjects
- Evaluate on 30 held-out clips from subject U0108

**What actually happened:**
The Kinect RGB video files for even the smallest subject session (U0209/S0500) are approximately **48GB**. No free-tier compute platform — Kaggle, Google Colab, Lightning AI, Paperspace — provides sufficient persistent storage or download bandwidth to handle this within a session. Kaggle datasets have a 20GB upload limit. Colab sessions reset before a 48GB download completes. The annotation CSVs from Zenodo loaded and parsed correctly; the blocker was exclusively the RGB video files.

**Workaround:**
A small pre-sampled dataset of ~20 clips was constructed from available frames and used for both fine-tuning and evaluation. This means:

- The fine-tuned model trained on the same samples it was evaluated on
- OCA, tIoU@0.5, and AA@1 metrics in `results.json` reflect behaviour on a toy dataset, not generalisation
- The metrics **do not represent real model capability** on OpenPack — they are included to satisfy the submission format and demonstrate the pipeline runs end-to-end

**Most confused operation class (expected, based on visual similarity):**
Had the full dataset been available, the most likely confusion would be between **"Tape"** and **"Pack"**. Both operations involve the worker leaning over the box with hands making contact with the box surface. The distinguishing signal — a tape dispenser vs. a folding/pressing motion — is subtle and occupies only a narrow frame window. A model without sufficient boundary examples would default to the higher-frequency class ("Pack") when uncertain, since mid-operation "Pack" frames dominate the training distribution.

**What would fix this in production:**
- Pre-extract all RGB frames to JPEG at 336×336 using ffmpeg before uploading to Kaggle, reducing 48GB of video to ~4–6GB of images — within Kaggle's 20GB dataset limit
- Or mount a GCS/S3 bucket with the full RGB files to a Vertex AI training VM
- The `data_pipeline.py` entropy sampling and boundary extraction logic is correct and would run without modification once frames are accessible at the expected paths (`frames/{subject}/{session}/frame_{idx:06d}.jpg`)
