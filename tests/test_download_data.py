import json
import pandas as pd
import pytest
from scripts.download_data import convert_jsonl


def test_review_and_metadata_projection(tmp_path):
    source = tmp_path / "sample.jsonl"
    source.write_text(json.dumps({"user_id": "u", "parent_asin": "a", "rating": 5,
        "timestamp": 123, "main_category": None, "price": "None", "images": [{"anything": []}]}) + "\n")
    reviews, meta = tmp_path / "reviews.parquet", tmp_path / "meta.parquet"
    assert convert_jsonl(source, reviews, batch_size=1) == 1
    assert convert_jsonl(source, meta, metadata=True) == 1
    assert pd.read_parquet(reviews).iloc[0].to_dict() == {
        "user_id": "u", "item_id": "a", "rating": 5., "timestamp": 123}
    assert pd.read_parquet(meta).iloc[0].to_dict() == {
        "item_id": "a", "category": "unknown", "price": "None"}


def test_conversion_failure_preserves_existing_output(tmp_path):
    output = tmp_path / "reviews.parquet"
    output.write_bytes(b"existing")
    source = tmp_path / "bad.jsonl"
    source.write_text('{bad json}\n')
    with pytest.raises(json.JSONDecodeError):
        convert_jsonl(source, output)
    assert output.read_bytes() == b"existing"
    assert not output.with_suffix(".parquet.partial").exists()
