from doctest import OutputChecker
import enum
from pathlib import Path
from typing import List
import json
from collections import defaultdict

XTD10_DIR = Path("/home/zkhan/Cross-lingual-Test-Dataset-XTD10/XTD10")
COCO_TRAIN_DIR = Path("./storage/10/coco2014/train2014")
OUTPUT_DIR = Path("./storage/10/multilingual_coco2014_xtd10")
LANGUAGES = ("es", "it", "ko", "pl", "ru", "tr", "zh")

OUTPUT_DIR.mkdir(exist_ok=True)

with open(XTD10_DIR / "test_image_names.txt", "r") as f:
    image_names = f.readlines()
    image_names = [x.strip() for x in image_names]

language_to_captions = {}
for language in LANGUAGES:
    with open(XTD10_DIR / f"test_1kcaptions_{language}.txt", "r") as f:
        captions_in_language = f.readlines()
        captions_in_language = [x.strip() for x in captions_in_language]
        language_to_captions[language] = captions_in_language


def make_val_record(image_path: str, caption: List[str]):
    return {
        "image": f"val2014/{image_path}",
        "caption": caption,
    }


def make_train_record(image_path: str, caption: str):
    return {
        "image": image_path,
        "caption": caption,
    }


def make_path_absolute(image_path: str):
    return str(COCO_TRAIN_DIR / image_path)


train_records = []
val_records = []
for sample_idx, image_path in enumerate(image_names):
    if "train" in image_path:
        records = [
            make_train_record(
                make_path_absolute(image_path),
                language_to_captions[language][sample_idx],
            )
            for language in LANGUAGES
        ]
        train_records.extend(records)
    if "val" in image_path:
        records = [
            make_val_record(
                image_path,
                [language_to_captions[language][sample_idx] for language in LANGUAGES],
            )
        ]
        val_records.extend(records)

with open(OUTPUT_DIR / "train.json", "w") as f:
    json.dump(train_records, f)


with open(OUTPUT_DIR / "val.json", "w") as f:
    json.dump(val_records, f)

print(f"Wrote {len(train_records)} training records")
print(f"Wrote {len(val_records)} validation records")

for language in LANGUAGES:
    pairs = list(zip(image_names, language_to_captions[language]))

    # The validation records.
    records_for_language = [
        make_val_record(image_path, [caption])
        for image_path, caption in pairs
        if "val" in image_path
    ]
    print(f"Wrote {len(records_for_language)} validation records for {language}")
    with open(OUTPUT_DIR / f"val_{language}.json", "w") as f:
        json.dump(records_for_language, f)

    # The training records.
    records_for_language = [
        make_train_record(make_path_absolute(image_path), caption)
        for image_path, caption in pairs
        if "train" in image_path
    ]
    print(f"Wrote {len(records_for_language)} training records for {language}")
    with open(OUTPUT_DIR / f"train_{language}.json", "w") as f:
        json.dump(records_for_language, f)
