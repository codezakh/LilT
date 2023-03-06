from pathlib import Path
import json

VIZWIZ_ROOT = Path("/net/acadia4a/data/zkhan/vizwiz-captions")

with open(VIZWIZ_ROOT / "annotations" / "val.json", "r") as f:
    raw_annotations = json.load(f)


image_id_to_annotation = {_["image_id"]: _ for _ in raw_annotations["annotations"]}

records = list(
    zip(
        raw_annotations["images"],
        raw_annotations["annotations"],
    )
)


def convert_record(image_record, annotation_record):
    assert image_record["id"] == annotation_record["image_id"]
    return {
        "caption": [annotation_record["caption"]],
        "image": image_record["file_name"],
    }


testing_records = [
    convert_record(i, image_id_to_annotation[i["id"]])
    for i in raw_annotations["images"]
]

with open(VIZWIZ_ROOT / "val_pairs.json", "w") as f:
    json.dump(testing_records, f)
