# Large-Scale Two-Stage Recommendation System

An end-to-end personalized recommendation system built on Amazon Reviews 2023. The pipeline combines a PyTorch Two-Tower retriever, FAISS vector search, and LightGBM LambdaRank to turn a 26K-item catalog into 20 ranked recommendations, served through FastAPI.

The project explores two practical questions: how much reranking improves recommendation quality, and how retrieval accuracy and serving throughput change with index choice and request concurrency.

## Results at a glance

- **857,505 interactions**, **98,906 users**, and **26,354 items** after filtering 4.6M raw reviews.
- **118% higher NDCG@20** and **103% higher HitRate@20** with LambdaRank compared with retrieval-only ordering.
- **665 requests/sec at 7.84 ms P95 latency** with four concurrent clients in a local CPU benchmark.
- **180,000 measured HTTP requests** across 100 users, three concurrency levels, and three trials per level.

## Architecture

```text
Amazon Reviews + Item Metadata
              │
     Filtering & Temporal Split
              │
     PyTorch Two-Tower Model
     User ID + History / Item ID + Category
              │
       FAISS IVF-PQ Search
              │
        Top-200 Candidates
              │
     LightGBM LambdaRank
     Retrieval + User / Item / Cross Features
              │
      Top-20 Recommendations
              │
          FastAPI API
```

**Retrieval:** User and item towers produce embeddings trained with InfoNCE and in-batch negatives. The user tower incorporates interaction history. FAISS indexes item embeddings for candidate generation.

**Ranking:** LambdaRank learns within user-level candidate groups using retrieval similarity, user activity and rating statistics, item popularity and rating statistics, and user-category affinity. Training and serving share the same feature builder. Scores express relative ranking preference, not calibrated click probabilities.

**Serving:** FastAPI loads the retrieval checkpoint, FAISS index, and native LightGBM model. `POST /recommend` returns ranked items and server-side timing; `GET /health` provides a health check.

## Offline evaluation

### Recommendation quality

Evaluated on **77,141 held-out positive test interactions** (rating ≥ 4), using the same retrieved candidates for both ordering methods.

| Metric | Retrieval-only ordering | LambdaRank reranking |
|---|---:|---:|
| NDCG@10 | 0.01240 | 0.02884 |
| NDCG@20 | 0.01590 | **0.03467** |
| HitRate@10 | 2.34% | **5.29%** |
| HitRate@20 | 3.75% | **7.60%** |
| MRR over retrieved candidates | 0.01173 | **0.02471** |

Candidate retrieval achieves **6.56% Recall@50**, **9.96% Recall@100**, and **15.02% Recall@200** on this positive-only test subset. Reranking improves candidate order; it cannot recover relevant items absent from the candidate set. Relative improvements above compare against retrieval-only ordering, not an MLP baseline.

### Exact vs. approximate retrieval

The index comparison uses **all 98,906 test interactions**, including lower-rated reviews, so its recall differs from the positive-only evaluation above.

| Index | Recall@200 | Amortized search time per query |
|---|---:|---:|
| FlatIP (exact) | 14.46% | 0.021 ms |
| IVF256,PQ32 (`nprobe=32`) | 14.39% | 0.011 ms |

IVF-PQ reduces measured search time by approximately 48%, with a 0.07 percentage-point recall decrease. These rounded timings are total batched search time divided by query count; they exclude user encoding, reranking, and HTTP overhead and are not single-request latency measurements.

## API benchmark

Local CPU serving on macOS ARM64, Python 3.14.6. Client and server run on the same machine. Requests use HTTP/1.1 persistent connections, with one connection per client worker and `top_k=20`.

Each concurrency level runs **three trials of 20,000 requests**, preceded by **100 warm-up requests per trial**, cycling through **100 reproducibly sampled users**. Each row below is the actual middle trial by QPS, preserving throughput and latency from the same run.

| Concurrent clients | Requests/sec | P50 | P95 | P99 |
|---|---:|---:|---:|---:|
| 1 | 541.48 | 1.79 ms | 2.12 ms | 2.41 ms |
| 4 | **665.13** | 5.86 ms | **7.84 ms** | 9.35 ms |
| 8 | 578.24 | 13.64 ms | 18.68 ms | 21.95 ms |

Four concurrent clients achieved the highest throughput among the tested settings, ranging from 659.69 to 668.43 requests/sec across three trials. Higher concurrency increased latency without increasing throughput. These measurements describe the local workload, not production capacity or a cloud deployment.

## Reproduce the pipeline

Run commands from the repository root. Hyperparameters and paths are configured in `configs/base.yaml`.

### 1. Set up the environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

If using Conda, activate the desired environment before creating or activating `.venv`. The active `.venv` supplies the Python interpreter and project dependencies.

### 2. Download and preprocess

```bash
python scripts/download_data.py
python scripts/preprocess.py
```

The downloader reads official JSONL files directly and converts the required columns to Parquet in bounded batches. Existing raw Parquet files are skipped; no Hugging Face remote loading script is required.

The Video Games dataset contains 4,624,615 raw reviews and 137,269 metadata records. Iterative 5-core filtering yields 857,505 interactions, split into 659,693 training, 98,906 validation, and 98,906 test interactions.

### 3. Train and evaluate

```bash
python scripts/run_macos.py scripts/train_retriever.py
python scripts/run_macos.py scripts/train_ranker.py
python scripts/run_macos.py scripts/evaluate.py
python scripts/run_macos.py scripts/compare_index.py
```

The launcher resolves the bundled OpenMP runtime conflict on macOS; on other platforms it forwards the command unchanged. To rebuild the index from an existing compatible retrieval checkpoint without retraining:

```bash
python scripts/run_macos.py scripts/train_retriever.py --index-only
```

### 4. Start the API

```bash
python scripts/run_macos.py -m uvicorn src.api.app:app --port 8000
```

In a second terminal, activate `.venv` and send a request using a user ID from the processed dataset:

```bash
source .venv/bin/activate
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/recommend \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"REPLACE_WITH_KNOWN_USER","top_k":20}'
```

Unknown users receive HTTP 404. `top_k` accepts values from 1 to 200.

### 5. Run the full benchmark

Keep the API running and execute in the second terminal:

```bash
python scripts/benchmark_api.py --output artifacts/api_benchmark_keepalive.json
```

The default workload runs all nine trials. Override `--requests`, `--repeats`, `--concurrency`, or `--num-users` for smaller experiments. Each completed trial is saved incrementally; request failures stop the benchmark and are recorded rather than counted as successful traffic.

### 6. Run tests

```bash
python scripts/run_macos.py -m pytest -q
```

## Evaluation protocol and limitations

- **Temporal split:** For each user, the final interaction is test, the penultimate is validation, and earlier interactions form training. This is a per-user split, not a global time cutoff.
- **Feature provenance:** Rating statistics and user histories use training interactions only and remain frozen for validation and test. The ID vocabulary covers the full filtered dataset, making the catalog transductive.
- **Ranker fitting:** Up to 20,000 validation users are sampled and partitioned 80/20 for LambdaRank fitting and early stopping. Retrieval model selection also uses validation data; final quality is reported on the untouched test labels.
- **Candidate labels:** Only actual retrieved candidates are used; held-out positives are never injected. Unobserved candidates are treated as implicit negatives, not confirmed dislikes. Previously seen items remain eligible for repeat recommendations.
- **Metric interpretation:** The final evaluator includes retrieval misses as zero-quality outcomes. LightGBM's training-time validation NDCG uses a different convention and query population and should not be presented as end-to-end test quality.
- **Scope:** Redis caching, cloud deployment, and MLflow tracking are not part of the measured pipeline. The repository includes a Dockerfile, but the reported serving results come from the local Python service.

## Project structure

```text
configs/base.yaml          Pipeline configuration
scripts/                   Download, preprocess, train, evaluate, benchmark
src/data/                  Filtering, splits, vocabularies, features, datasets
src/retrieval/             Two-Tower model, trainer, FAISS index
src/ranking/               LambdaRank features and reference MLP modules
src/eval/                  Ranking metrics and evaluation utilities
src/api/                   FastAPI application and recommendation engine
tests/                     Data, model, serving, and benchmark tests
```

Generated outputs include `artifacts/retriever_best.pt`, `artifacts/faiss.index` and its ID mapping, `artifacts/ranker.txt`, `artifacts/ranker_training.json`, `artifacts/evaluation_results.json`, and `artifacts/api_benchmark_keepalive.json`. The reference MLP/ONNX export path is not required for the LambdaRank pipeline.

## Acknowledgments

Data comes from [McAuley Lab's Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023).
