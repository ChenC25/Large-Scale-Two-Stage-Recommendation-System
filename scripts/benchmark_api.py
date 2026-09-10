"""Repeated multi-user HTTP benchmark; start the API server before running."""
import argparse
import json
import platform
import time
import http.client
import threading
from urllib.parse import urlsplit
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_trial(url, users, requests, concurrency, warmup):
    payloads = [json.dumps({"user_id": user, "top_k": 20}).encode() for user in users]

    endpoint = urlsplit(url)
    if endpoint.scheme not in ("http", "https") or not endpoint.hostname:
        raise ValueError("URL must be an HTTP(S) endpoint")
    path = endpoint.path or "/"
    if endpoint.query:
        path += "?" + endpoint.query
    local = threading.local()
    connections = []
    lock = threading.Lock()

    def request(i):
        start = time.perf_counter()
        if not hasattr(local, "connection"):
            cls = http.client.HTTPSConnection if endpoint.scheme == "https" else http.client.HTTPConnection
            local.connection = cls(endpoint.hostname, endpoint.port, timeout=60)
            with lock:
                connections.append(local.connection)
        connection = local.connection
        connection.request("POST", path, body=payloads[i % len(payloads)],
                           headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        body = response.read()  # Fully consume the response before reusing the socket.
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {body[:200]!r}")
        result = json.loads(body)
        if not isinstance(result.get("recommendations"), list):
            raise ValueError("Invalid recommendation response")
        return (time.perf_counter() - start) * 1000

    try:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            list(pool.map(request, range(warmup)))
            start = time.perf_counter()
            latencies = list(pool.map(request, range(requests)))
            elapsed = time.perf_counter() - start
    finally:
        for connection in connections:
            connection.close()
    return {"requests": requests, "concurrency": concurrency,
            "qps": requests / elapsed, "elapsed_seconds": elapsed,
            **{f"p{p}_ms": float(np.percentile(latencies, p)) for p in (50, 95, 99)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000/recommend")
    parser.add_argument("--user-id", help="Optional single-user benchmark override")
    parser.add_argument("--users-file", type=Path, help="Parquet file with user_id column; default: configured test split")
    parser.add_argument("--num-users", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--requests", type=int, default=20000, help="Measured requests per trial")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=100, help="Unmeasured requests before each trial")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts/api_benchmark.json")
    args = parser.parse_args()
    if min(args.requests, args.repeats, args.num_users, *args.concurrency) < 1 or args.warmup < 0:
        parser.error("requests, repeats, num-users and concurrency must be positive; warmup must be nonnegative")
    if len(set(args.concurrency)) != len(args.concurrency):
        parser.error("concurrency values must be unique")
    if args.user_id:
        users = [args.user_id]
    else:
        if args.users_file is None:
            cfg = yaml.safe_load((PROJECT_ROOT / "configs/base.yaml").read_text())
            args.users_file = PROJECT_ROOT / cfg["data"]["processed_dir"] / "test.parquet"
        available = sorted(pd.read_parquet(args.users_file, columns=["user_id"])
                           .user_id.dropna().astype(str).unique())
        if not available:
            parser.error("No users found in users-file")
        users = np.random.default_rng(args.seed).choice(
            available, size=min(args.num_users, len(available)), replace=False).tolist()
    results = {"transport": "HTTP/1.1 persistent connection per worker", "url": args.url, "platform": platform.platform(), "python": platform.python_version(),
               "user_count": len(users), "seed": args.seed, "user_ids": users,
               "requests_per_trial": args.requests, "repeats": args.repeats,
               "warmup_per_trial": args.warmup, "runs": [], "summary": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = args.output.with_suffix(".json.partial")
        temporary.write_text(json.dumps(results, indent=2))
        temporary.replace(args.output)

    save()
    for concurrency in args.concurrency:
        trials = []
        for repeat in range(1, args.repeats + 1):
            print(f"Concurrency={concurrency}, trial={repeat}/{args.repeats}, "
                  f"requests={args.requests}, users={len(users)}", flush=True)
            try:
                result = run_trial(args.url, users, args.requests, concurrency, args.warmup)
            except Exception as exc:
                results["failure"] = {"concurrency": concurrency, "repeat": repeat, "error": str(exc)}
                save()
                raise
            result["repeat"] = repeat
            trials.append(result)
            results["runs"].append(result)
            save()
            print(json.dumps(result, indent=2), flush=True)
        # Preserve QPS and latency from the same actual trial for resume reporting.
        representative = sorted(trials, key=lambda run: run["qps"])[len(trials)//2]
        results["summary"].append({
            "concurrency": concurrency,
            "representative_run": representative,
            "selection": "middle trial by QPS (upper middle for even repeat counts)",
            "qps_min": min(run["qps"] for run in trials),
            "qps_max": max(run["qps"] for run in trials),
        })
        save()
    print("Summary:", json.dumps(results["summary"], indent=2))
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
