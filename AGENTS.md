# AGENTS.md — AI Development Log

## Phase 1 — Inference API & GPU Optimization [Hour 4]

**Tool:** Claude (claude.ai)
**Request:** Add dynamic batching in FastAPI so GPU doesn't sit idle, and add inference mode to predictor.py
**Output:** `main.py` — async queue-based dynamic batcher with 0.1s wait window and future-based result mapping | `predictor.py` — model.eval() at load time, torch.inference_mode() wrapping GPU ops
**Accepted:** Dynamic batching logic, inference mode context manager, model.eval() placement
**Modified:** None — accepted as is
**Time saved:** ~45 minutes
**Time of this prompt:** 18:30


## Phase 1 — Docker & Deployment [Hour 4]

**Tool:** Claude (claude.ai)
**Request:** Build Dockerfile and docker-compose.yml for FastAPI + Qwen2-VL-2B with GPU support
**Output:** `Dockerfile`, `docker-compose.yml`, `requirements.txt`
**Accepted:** Yes — as is
**Modified:** None
**Time saved:** ~20 minutes
**Time of this prompt:** 18:41


## Phase 1 — Structured Prompt Engineering [Hour 4]

**Tool:** Claude (claude.ai)
**Request:** Design inference prompt for Qwen2-VL-2B to return consistent JSON output for warehouse operation classification
**Output:** Few-shot prompt with concrete JSON example, operation class constraints, no-markdown instruction
**Accepted:** Yes
**Modified:** None
**Time saved:** ~15 minutes
**Time of this prompt:** 18:49


## Phase 2 — Data Pipeline [Hour 12]

**Tool:** Claude (claude.ai)
**Request:** Build OpenPack temporal data pipeline with entropy-based keyframe sampling, WebDataset sharding, and VLM training pair generation
**Output:** `data_pipeline.py` — CSV annotation parser, boundary clip builder, entropy-based frame sampler, WebDataset shard writer, 20-sample exporter
**Accepted:** Full pipeline structure, entropy sampling strategy, WebDataset sharding logic, training pair JSON schema
**Modified:** Debugged multiple issues during development:
- Fixed `webdataset` version conflict (downgraded to 0.2.5 to avoid PyTorch dependency on Windows)
- Fixed `openpack-toolkit` missing `load_operation_labels` API — replaced with direct `pandas` CSV parsing
- Fixed annotation CSV column mismatch — real format uses ISO datetime strings (`start`, `end`, `operation`) not unix timestamps
- Fixed 0 clips bug — sample data has only 15 frames vs expected 125; changed minimum check from `CLIP_FRAMES` to `FRAMES_PER_CLIP`
- Fixed frame path resolution for nested timestamp-based folder structure in sample data
**Time saved:** ~3 hours
**Time of this prompt:** 22:24


## Phase 2 — Dataset Research [Hour 12]

**Tool:** ChatGPT Deep Research
**Request:** Find any publicly available pre-processed or pre-clipped versions of the OpenPack dataset with RGB frames and operation annotations, ready for VLM fine-tuning without downloading the full 400GB+ raw data. Searched HuggingFace Datasets, Kaggle Datasets, Zenodo, GitHub, and academic paper supplementary materials.
**Output:** No usable pre-processed OpenPack RGB dataset found anywhere — HuggingFace: no entries | Kaggle: no entries | Zenodo: annotations only, RGB on separate Google Drive | Google Drive: download quota exceeded | GitHub openpack-dataset: U0209 sample only, 15 frames, no full video
**Accepted:** Research conclusion — full RGB inaccessible within free-tier constraints
**Modified:** N/A — research task, no code output
**Time saved:** None (wasted 2 hours figuring out the dataset issue , decided to move on with Sample dataset creating sample scripts while trying to locally download one big file that i can clip and use for training)
**Time of this prompt:** 03:48

## Phase 3: PEFT Fine-Tuning

### Interaction 1 — DataLoader / Dataset Class
- **Tool used:** Claude (claude.ai)
- **Request:** "Build a PyTorch Dataset class for OpenPack VLM fine-tuning that loads
  JSON sample pairs, builds Qwen2-VL message format with image frames, applies chat
  template, and returns tokenized inputs with labels"
- **Code accepted:** `OpenPackDataset.__getitem__()`, `data_collator()` function,
  `image_grid_thw` cat logic in collator
- **Code modified by me:** Frame path resolution (Windows backslash fix), added
  `.squeeze(0)` for batch dimension
- **What I wrote myself:** LoRA config, QLoRA BitsAndBytes config, TrainingArguments,
  W&B init and login, VRAM math cell
- **Estimated time saved:** ~45 minutes
- **Time of this prompt:** 8:00 

---
