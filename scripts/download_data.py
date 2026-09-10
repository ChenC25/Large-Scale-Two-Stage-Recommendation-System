"""Download raw JSONL directly; no Hugging Face dataset loading scripts."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from src.utils import load_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def convert_jsonl(source, destination, *, metadata=False, batch_size=10000):
    """Project required scalar columns in bounded batches, then publish atomically."""
    schema = (pa.schema([("item_id", pa.string()), ("category", pa.string()),
                         ("price", pa.string())]) if metadata else
              pa.schema([("user_id", pa.string()), ("item_id", pa.string()),
                         ("rating", pa.float64()), ("timestamp", pa.int64())]))
    destination = Path(destination)
    temporary = destination.with_suffix(".parquet.partial")
    count = 0
    try:
        with pq.ParquetWriter(temporary, schema) as writer, open(source, encoding="utf-8") as stream:
            rows = []
            for line in tqdm(stream, desc=f"Converting {destination.name}", unit=" rows"):
                if not line.strip():
                    continue
                record = json.loads(line)
                if metadata:
                    row = {"item_id": str(record["parent_asin"]),
                           "category": str(record.get("main_category") or "unknown"),
                           "price": str(record["price"]) if record.get("price") is not None else None}
                else:
                    row = {"user_id": str(record["user_id"]),
                           "item_id": str(record["parent_asin"]),
                           "rating": float(record["rating"]), "timestamp": int(record["timestamp"])}
                rows.append(row)
                count += 1
                if len(rows) >= batch_size:
                    writer.write_table(pa.Table.from_pylist(rows, schema=schema))
                    rows.clear()
            if rows:
                writer.write_table(pa.Table.from_pylist(rows, schema=schema))
        if count == 0:
            raise ValueError(f"No records in {source}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return count


def main():
    data_cfg = load_config()["data"]
    raw_dir = PROJECT_ROOT / data_cfg["raw_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    category = data_cfg["dataset_name"]
    files = [(f"raw/review_categories/{category}.jsonl", "reviews.parquet", False),
             (f"raw/meta_categories/meta_{category}.jsonl", "meta.parquet", True)]
    for filename, output, metadata in files:
        destination = raw_dir / output
        if destination.exists():
            print(f"Already exists: {destination}; skipping")
            continue
        print(f"Downloading {filename} ...", flush=True)
        source = hf_hub_download(repo_id=data_cfg["hf_dataset"], filename=filename,
                                 repo_type="dataset", local_dir=raw_dir / "downloads")
        count = convert_jsonl(source, destination, metadata=metadata)
        print(f"Saved {destination} ({count:,} rows)")


if __name__ == "__main__":
    main()
