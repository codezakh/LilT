import json
from pathlib import Path

CONCADIA_ROOT = Path("/net/acadia4a/data/zkhan/concadia")


with open(CONCADIA_ROOT / "wiki_split.json", "r") as f:
    raw_annotations = json.load(f)["images"]


test_split = [_ for _ in raw_annotations if _["split"] == "test"]


def convert_record(record):
    converted = {"image": record["filename"], "caption": [record["description"]["raw"]]}
    return converted


with open(CONCADIA_ROOT / "test_pairs_description.json", "w") as f:
    json.dump([convert_record(_) for _ in test_split], f)
