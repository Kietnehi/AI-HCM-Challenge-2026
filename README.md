<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:0056D2,50:6C63FF,100:00C9A7&height=180&section=header&text=AI-HCM%20Challenge%202026&fontSize=42&fontColor=ffffff&animation=fadeIn&fontAlignY=35&desc=Multimodal%20Video%20Retrieval&descAlignY=58&descSize=20" alt="AI-HCM Challenge 2026 banner" />
</p>

<p align="center">
  <a href="https://github.com/Kietnehi/AI-HCM-Challenge-2026/stargazers"><img src="https://img.shields.io/github/stars/Kietnehi/AI-HCM-Challenge-2026?style=for-the-badge&color=yellow&logo=github" alt="GitHub stars" /></a>
  <a href="https://github.com/Kietnehi/AI-HCM-Challenge-2026/network/members"><img src="https://img.shields.io/github/forks/Kietnehi/AI-HCM-Challenge-2026?style=for-the-badge&color=orange&logo=github" alt="GitHub forks" /></a>
  <a href="https://github.com/Kietnehi/AI-HCM-Challenge-2026/issues"><img src="https://img.shields.io/github/issues/Kietnehi/AI-HCM-Challenge-2026?style=for-the-badge&color=red&logo=github" alt="GitHub issues" /></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Kaggle-Notebook-20BEFF?style=flat-square&logo=kaggle&logoColor=white" alt="Kaggle Notebook" />
  <img src="https://img.shields.io/badge/FAISS-Vector_Search-0467DF?style=flat-square" alt="FAISS" />
  <img src="https://img.shields.io/badge/Task-Multimodal_Video_Retrieval-6C63FF?style=flat-square" alt="Multimodal Video Retrieval" />
</p>

<p align="center">
  A competition-oriented retrieval system for finding the right <b>video</b>, <b>moment</b>, and <b>frame</b> from natural-language queries.
</p>

> [!IMPORTANT]
> This repository contains source code, notebooks, documentation, and lightweight examples only. Large datasets, internal links, private prompts, generated artifacts, and secrets are intentionally excluded through `.gitignore`.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Workflow](#system-workflow)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Pipeline Guide](#pipeline-guide)
- [API Configuration and Cost Controls](#api-configuration-and-cost-controls)
- [Submission Format](#submission-format)
- [Data and Security](#data-and-security)
- [Project Status](#project-status)
- [Authors & GitHub Accounts](#authors--github-accounts)

## Overview

AI-HCM Challenge 2026 requires a system that can understand a query, search across heterogeneous video metadata, identify promising moments, and export results in the competition format. This repository brings those stages together in a reproducible notebook-based workflow.

The system combines visual embeddings, multilingual text retrieval, OCR, speech transcripts, captions, object detections, metadata, and temporal evidence. Candidates from independent retrieval branches are fused and reranked before fine-grained frame refinement and human review.

Supported submission scenarios include:

- **KIS** — known-item search for a target video moment.
- **Q&A** — question answering grounded in retrieved video content.
- **TRAKE** — temporal retrieval requiring an ordered sequence of frames.

## Key Features

- Multimodal retrieval over visual features, OCR, ASR, captions, metadata, and detected objects.
- Dedicated FAISS indices for different embedding spaces, plus BM25 and object inverted indices.
- Query analysis, translation, and expansion for Vietnamese and English retrieval.
- Weighted reciprocal rank fusion, video priors, reranking, and temporal non-maximum suppression.
- Fine-grained frame refinement by decoding source video within a narrow time window.
- Stage-level and query-level checkpoints for safe resume after notebook interruption.
- Human-review packages with candidate frames and editable decisions.
- Submission validation for KIS, Q&A, and TRAKE formats.
- A safe-by-default API workflow with dry runs, caching, and a hard cost limit.

## System Workflow

```text
Raw videos + metadata
          │
          ▼
Keyframes + OCR + ASR + captions + visual/text features
          │
          ▼
FAISS indices + BM25 index + object index
          │
          ▼
Query analysis + translation + expansion
          │
          ▼
Multibranch retrieval → fusion → reranking → temporal refinement
          │
          ▼
Candidate review + frame selection
          │
          ▼
Validation → submission.zip
```

Each embedding model keeps its own vector space and FAISS index. Scores are combined only at the ranking layer, preventing invalid comparisons between unrelated embeddings.

## Repository Structure

| Path | Purpose |
|---|---|
| [`Code-Extract-Input/`](Code-Extract-Input/) | Feature-extraction notebooks for keyframes, OCR, speech-to-text, image captioning, VLM output, summaries, translations, and embeddings. |
| [`Code-ThuNghiem-AIC/Pipeline-API/`](Code-ThuNghiem-AIC/Pipeline-API/) | Retrieval pipeline that uses external APIs for query processing, text embeddings, and reranking. |
| [`Code-ThuNghiem-AIC/Local/`](Code-ThuNghiem-AIC/Local/) | GPU-based local pipeline with index building, retrieval, review, validation, and submission export. |
| [`Code-ThuNghiem-AIC/Pipeline-Cũ-4.8-Point/`](Code-ThuNghiem-AIC/Pipeline-Cũ-4.8-Point/) | Previous 4.8-point pipeline retained as an experimental reference. |
| [`Planning/`](Planning/) | Development plans, technical notes, and experiment documentation. |
| [`TheLeCuocThi-DeThi/`](TheLeCuocThi-DeThi/) | Competition rules, query sets, and official submission-format notes. |
| [`Information.txt`](Information.txt) | Dataset and resource notes. |
| [`.gitignore`](.gitignore) | Excludes large data, secrets, caches, internal resources, and generated outputs. |

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Kietnehi/AI-HCM-Challenge-2026.git
cd AI-HCM-Challenge-2026
```

### 2. Prepare the environment

Running the notebooks on **Kaggle** is recommended because feature extraction and retrieval require a GPU, substantial memory, and several large models.

Before running a pipeline, make sure that you have:

- A Kaggle account with access to the required datasets.
- A GPU-enabled Kaggle Notebook session.
- Internet access when a notebook must download models or call an external API.
- The source videos, feature dataset, and competition queries mounted under `/kaggle/input`.
- An `OPENROUTER_API_KEY` stored as a Kaggle Secret when using the API pipeline.

Most notebooks automatically discover datasets under `/kaggle/input`. If discovery fails, override the relevant configuration value, such as `feature_root`, `dataset_root`, `query_dir`, `FEATURE_ROOT`, `ART_INPUT`, or `PKG_INPUT`.

### 3. Choose a workflow

| Workflow | Best for | Main trade-off |
|---|---|---|
| **Feature Extraction** | Creating or refreshing features from raw videos. | Compute-intensive and usually run only when inputs or models change. |
| **API Pipeline** | Rapid experiments using hosted embeddings, rerankers, and query analysis. | Requires API access and careful cost control. |
| **Local Pipeline** | GPU-first experiments with more local control and an integrated review flow. | Requires more GPU memory and model setup. |

### 4. Run a smoke test first

Notebook stages produce artifacts consumed by later stages, so run them in the documented order. Before processing the complete dataset, use a small query or video limit to verify dataset paths, schemas, model loading, output locations, and estimated API cost.

## Pipeline Guide

### API Pipeline

See the [API Pipeline README](Code-ThuNghiem-AIC/Pipeline-API/README.md) and [Kaggle execution guide](Code-ThuNghiem-AIC/Pipeline-API/HUONG_DAN_CHAY_KAGGLE.md) for pipeline-specific instructions.

Run the main notebooks in this order:

1. **`01-build-indices-api.ipynb`**
   - Discovers and validates input data.
   - Produces canonical records.
   - Builds FAISS, BM25, object, and text-embedding indices.
   - Writes artifacts and `artifact_manifest.json` to `/kaggle/working/artifacts/`.

2. **`02-retrieve-refine-candidates-api.ipynb`**
   - Analyzes queries and performs multibranch retrieval.
   - Applies fusion, reranking, video priors, temporal NMS, and frame refinement.
   - Writes a review package to `/kaggle/working/review_package/`.
   - Exports the baseline `/kaggle/working/submission.zip` from its submission cell.

3. **`03-time-to-frameindex.ipynb`**
   - Converts and verifies time-based results against frame indices.

> [!NOTE]
> An earlier design referenced a separate human-review notebook for the API workflow. That notebook is not currently present; baseline submission export is integrated into the retrieval notebook instead.

Expected Kaggle datasets include:

- `fatle542/AIC-Dataset`
- `kitnehi1211/feature-AIC-2026`
- `kitnehi1211/dethithunghiem`

Dataset slugs and mount paths may change between versions. Update `CFG` when automatic discovery cannot resolve them.

### Local Pipeline

See the [Local Pipeline README](Code-ThuNghiem-AIC/Local/README.md) and [Local Kaggle guide](Code-ThuNghiem-AIC/Local/KAGGLE_RUN_GUIDE.md) for full instructions.

The local workflow is designed as a three-notebook sequence:

1. **`01_build_indices_local.ipynb`** — validates schema and coverage, creates canonical records, and builds separate SigLIP2, BGE-M3, and CLIP FAISS indices alongside multifield BM25 and an object inverted index.
2. **`02_retrieve_refine_candidates_local.ipynb`** — translates and expands queries, performs bilingual retrieval, applies weighted RRF and reranking, refines frames, and produces a review package.
3. **`03_human_review_submit_local.ipynb`** *(documented, but not currently included)* — provides an `ipywidgets` review interface, saves decisions, validates outputs, and creates `submission.zip`.

> [!WARNING]
> The third local notebook is referenced by the local documentation but is not present in the current repository snapshot. The checked-in local workflow therefore ends with the review package produced by NB02. Use the API pipeline's integrated exporter, or restore the reviewed local submission notebook, before relying on this path for a final archive.

Important outputs include:

- **NB01:** `/kaggle/working/artifacts/` and `manifest.json`.
- **NB02:** `artifacts02_mimo/review_package/`, including `candidates.parquet`, `frame_catalog.csv`, `frames/`, `queries_parsed.json`, and `sheets/`.
- **NB03 (when available):** review decisions, the validation report, and the final `submission.zip`.

The local workflow releases GPU resources between stages and treats missing modalities gracefully: a candidate loses only the score contribution from the unavailable branch rather than being discarded entirely.

## API Configuration and Cost Controls

Store the OpenRouter key in the `OPENROUTER_API_KEY` environment variable or a Kaggle Secret with the same name:

```text
OPENROUTER_API_KEY=<your-key>
```

Never paste a real key into a notebook, source file, README, log, or generated artifact.

The API notebooks include the following safeguards:

- `CFG["DRY_RUN"] = True` by default, estimating tokens and cost without making paid requests.
- `MAX_TOTAL_COST_USD = 2.00` as a shared hard cap.
- Content-addressed caching for embeddings, reranking, and LLM responses.
- Explicit opt-in through `DRY_RUN = False` after inputs and estimates have been reviewed.

The current API workflow uses:

| Component | Model or method |
|---|---|
| Visual representation | `google/siglip2-giant-opt-patch16-384` |
| Text embeddings | `openai/text-embedding-3-small` through OpenRouter |
| Text reranking | `voyageai/rerank-2.5-lite` |
| Query analysis, translation, and expansion | `xiaomi/mimo-v2.5` |
| Lexical retrieval | BM25-Okapi on `scipy.sparse` |

## Submission Format

Always verify generated outputs against the competition rules in [`TheLeCuocThi-DeThi/sotuyenAIC.md`](TheLeCuocThi-DeThi/sotuyenAIC.md). The validator checks the core constraints below:

- One CSV file per query.
- UTF-8 encoding, comma-separated values, and no header row.
- No more than 100 rows per file.
- A `video_id` without the `.mp4` suffix.
- A valid integer `frame_id`.
- The expected number of Q&A columns and an answer of at most 100 characters.
- The required number of TRAKE frames in ascending order.
- A ZIP archive containing a top-level `submission/` directory.

For Q&A tasks that require a reviewer-provided answer, the pipeline can consume a JSON mapping such as:

```json
{
  "query-p1-15-qa": "12"
}
```

After review, rerun the export and validation stage. Do not submit an archive while the validation report contains any P0 error.

## Data and Security

The following local resources are excluded from Git through `.gitignore`:

- `Feature_Dataset/` and `Feature_Dataset.zip`
- `Kiet-Prompt/`
- `Link.txt`
- `THUNGHIEM-bo-de-thi/` and related experimental datasets
- `.env` and other secret-bearing files
- Caches, virtual environments, generated artifacts, and temporary outputs

These resources may still exist on a developer machine or in a private Kaggle environment. A fresh clone requires them to be provisioned separately with the appropriate access rights.

Security checklist:

- Never commit API keys, passwords, tokens, cookies, or private paths.
- Run `git status` and `git check-ignore` before adding new data.
- Revoke and rotate any credential that has appeared in Git history; deleting it from the latest file is not sufficient.
- Do not publish competition datasets or internal documents without distribution permission.

## Project Status

This repository is actively used for research, experimentation, and submission preparation for AI-HCM Challenge 2026. To keep experiments reproducible, record model versions, dataset versions, parameters, random seeds, and validation results whenever a pipeline configuration changes.

## Authors & GitHub Accounts

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&height=120&section=header" alt="Header" />
</p>

| | | | |
| :---: | :---: | :---: | :---: |
| <a href="https://github.com/Kietnehi"><img src="https://github-readme-stats.vercel.app/api?username=Kietnehi&show_icons=true&hide_title=true&hide=issues,contribs,prs&rank_icon=github&hide_border=true" alt="Kietnehi's GitHub stats" /></a> | <a href="https://github.com/ductoanoxo"><img src="https://github-readme-stats.vercel.app/api?username=ductoanoxo&show_icons=true&hide_title=true&hide=issues,contribs,prs&rank_icon=github&hide_border=true" alt="ductoanoxo's GitHub stats" /></a> | <a href="https://github.com/phatle224"><img src="https://github-readme-stats.vercel.app/api?username=phatle224&show_icons=true&hide_title=true&hide=issues,contribs,prs&rank_icon=github&hide_border=true" alt="phatle224's GitHub stats" /></a> | <a href="https://github.com/nhdotvn"><img src="https://github-readme-stats.vercel.app/api?username=nhdotvn&show_icons=true&hide_title=true&hide=issues,contribs,prs&rank_icon=github&hide_border=true" alt="nhdotvn's GitHub stats" /></a> |
| <img src="https://github.com/Kietnehi.png" width="80" alt="Trương Phú Kiệt" /> | <img src="https://github.com/ductoanoxo.png" width="80" alt="Đức Toàn" /> | <img src="https://github.com/phatle224.png" width="80" alt="Phát Lê" /> | <img src="https://github.com/nhdotvn.png" width="80" alt="Lê Ngọc Hiệp" /> |
| <b><a href="https://github.com/Kietnehi">Trương Phú Kiệt</a></b> | <b><a href="https://github.com/ductoanoxo">Đức Toàn</a></b> | <b><a href="https://github.com/phatle224">Phát Lê</a></b> | <b><a href="https://github.com/nhdotvn">Lê Ngọc Hiệp</a></b> |
| AI Engineer | NLP Engineer | Data Engineer | ML Engineer |
| <p align="center"><img src="https://img.shields.io/github/followers/Kietnehi?style=for-the-badge" alt="Kietnehi followers" /> <img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.github-star-counter.workers.dev%2Fuser%2FKietnehi&query=%24.stars&style=for-the-badge&color=yellow&label=Stars&logo=github" alt="Kietnehi stars" /> <a href="https://github.com/Kietnehi"><img src="https://img.shields.io/badge/Profile-GitHub-181717?style=for-the-badge&logo=github" alt="Kietnehi GitHub profile" /></a></p> | <p align="center"><img src="https://img.shields.io/github/followers/ductoanoxo?style=for-the-badge" alt="ductoanoxo followers" /> <img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.github-star-counter.workers.dev%2Fuser%2Fductoanoxo&query=%24.stars&style=for-the-badge&color=yellow&label=Stars&logo=github" alt="ductoanoxo stars" /> <a href="https://github.com/ductoanoxo"><img src="https://img.shields.io/badge/Profile-GitHub-181717?style=for-the-badge&logo=github" alt="ductoanoxo GitHub profile" /></a></p> | <p align="center"><img src="https://img.shields.io/github/followers/phatle224?style=for-the-badge" alt="phatle224 followers" /> <img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.github-star-counter.workers.dev%2Fuser%2Fphatle224&query=%24.stars&style=for-the-badge&color=yellow&label=Stars&logo=github" alt="phatle224 stars" /> <a href="https://github.com/phatle224"><img src="https://img.shields.io/badge/Profile-GitHub-181717?style=for-the-badge&logo=github" alt="phatle224 GitHub profile" /></a></p> | <p align="center"><img src="https://img.shields.io/github/followers/nhdotvn?style=for-the-badge" alt="nhdotvn followers" /> <img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.github-star-counter.workers.dev%2Fuser%2Fnhdotvn&query=%24.stars&style=for-the-badge&color=yellow&label=Stars&logo=github" alt="nhdotvn stars" /> <a href="https://github.com/nhdotvn"><img src="https://img.shields.io/badge/Profile-GitHub-181717?style=for-the-badge&logo=github" alt="nhdotvn GitHub profile" /></a></p> |

<p align="center">
  <a href="https://github.com/Kietnehi/AI-HCM-Challenge-2026">
    <img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&pause=1000&color=236AD3&center=true&vCenter=true&width=600&lines=AI-HCM+Challenge+2026;Multimodal+Video+Retrieval;Search%2C+Rerank+%26+Submission" alt="AI-HCM Challenge 2026" />
  </a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/AI--HCM-Challenge_2026-0056D2?style=flat-square" alt="AI-HCM Challenge 2026" />
  <img src="https://img.shields.io/badge/Task-Multimodal_Video_Retrieval-FF4B4B?style=flat-square" alt="Multimodal Video Retrieval" />
</p>

### Tech Stack

<p align="center">
  <img src="https://skillicons.dev/icons?i=python,pytorch,sklearn,docker,git" alt="Python, PyTorch, scikit-learn, Docker, and Git" />
</p>

### AI-HCM CHALLENGE 2026

<p align="center">
  <a href="https://github.com/Kietnehi/AI-HCM-Challenge-2026">
    <img src="https://img.shields.io/github/stars/Kietnehi/AI-HCM-Challenge-2026?style=for-the-badge&color=yellow" alt="Stars" />
    <img src="https://img.shields.io/github/forks/Kietnehi/AI-HCM-Challenge-2026?style=for-the-badge&color=orange" alt="Forks" />
    <img src="https://img.shields.io/github/issues/Kietnehi/AI-HCM-Challenge-2026?style=for-the-badge&color=red" alt="Issues" />
  </a>
</p>

<!-- Dynamic quote -->
<p align="center">
  <img src="https://quotes-github-readme.vercel.app/api?type=horizontal&theme=dark" alt="Daily Quote" />
</p>

<p align="center">
  <i>Thank you for stopping by! Don't forget to give this repository a <b>⭐ Star</b> if you find it useful.</i>
</p>

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&height=80&section=footer" alt="Footer" />
</p>
