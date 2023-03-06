import json
from pathlib import Path

WINOGROUND_ROOT = Path("/net/acadia4a/data/zkhan/winoground")

with open(WINOGROUND_ROOT / "annotations.jsonl", "r") as f:
    raw_annotations = [json.loads(line) for line in f]


def convert_record(record):
    a = {
        "caption": [record["caption_0"]],
        "image": record["image_0"] + ".png",
    }

    b = {
        "caption": [record["caption_1"]],
        "image": record["image_1"] + ".png",
    }

    return a, b


testing_records = []
for _ in raw_annotations:
    testing_records.extend(convert_record(_))

with open(WINOGROUND_ROOT / "test_pairs.json", "w") as f:
    json.dump(testing_records, f)
