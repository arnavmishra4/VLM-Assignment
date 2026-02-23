# AGENTS.md — AI Development Log

## Phase 1 — Inference API & GPU Optimization [Hour 1]

**Tool:** Claude (claude.ai)  
**Request:** Add dynamic batching in FastAPI so GPU doesn't sit idle, and add inference mode to predictor.py  
**Output:** `main.py` — async queue-based dynamic batcher with 0.1s wait window and future-based result mapping | `predictor.py` — model.eval() at load time, torch.inference_mode() wrapping GPU ops  
**Accepted:** Dynamic batching logic, inference mode context manager, model.eval() placement  
**Modified:** None — accepted as is  
**Time of this prompt:** 18:30  


## Phase 1 — Docker & Deployment [Hour 1]

**Tool:** Claude (claude.ai)  
**Request:** Build Dockerfile and docker-compose.yml for FastAPI + Qwen2-VL-2B with GPU support  
**Output:** `Dockerfile`, `docker-compose.yml`, `requirements.txt`  
**Accepted:** Yes — as is  
**Modified:** None  
**Time of this prompt:** 18:41  


## Phase 1 — Structured Prompt Engineering [Hour 1]

**Tool:** Claude (claude.ai)  
**Request:** Design inference prompt for Qwen2-VL-2B to return consistent JSON output for warehouse operation classification  
**Output:** Few-shot prompt with concrete JSON example, operation class constraints, no-markdown instruction  
**Accepted:** Yes  
**Modified:** None  
**Time of this prompt:** 18:49 


## Phase 2 — Data Pipeline [Hour 4]

**Tool:** Claude (claude.ai)  
**Request:** Build OpenPack temporal data pipeline with entropy-based keyframe sampling  
**Output:** `data_pipeline.py` — CSV parser, boundary detection, VLM training pair generator  
**Accepted:** CSV parser, boundary detection, training pair schema  
**Modified:** Debugged path issues, fixed CSV format mismatch (real format used ISO timestamps)  
**Time of this prompt:** 22:24  
