"""
01_prepare_data.py
------------------
Builds a small Food-101 subset (only our chosen classes) split into
train / val / test, and creates an empty calories.csv template.

Run from the folder that contains the extracted `food-101` folder:
    python 01_prepare_data.py

Output:
    food101_subset/train/<class>/*.jpg
    food101_subset/val/<class>/*.jpg
    food101_subset/test/<class>/*.jpg
    splits.csv        (which image is in which split, for reproducibility)
    calories.csv      (template: fill kcal_per_100g from USDA by hand)
"""

import csv
import random
import shutil
from collections import Counter
from pathlib import Path

# ----------------------------------------------------------------------
# SETTINGS  (edit these)
# ----------------------------------------------------------------------
SRC = Path("food-101")  # folder that directly contains images/ and meta/
DST = Path("food101_subset")  # output folder that will be created
VAL_FRACTION = 0.15  # share of the official train set used for validation
SEED = 42  # fixed seed: every teammate gets the SAME split

# Our classes. Names must match Food-101 exactly (the script checks this).
# This is a starter list: replace it with your team's final choice.
MY_CLASSES = [
    "dumplings",
    "pizza",
    "caesar_salad",
    "ice_cream",
    "sushi",
    "hamburger",
    "fried_rice",
    "steak",
    "chocolate_cake",
    "french_fries",
    "donuts",
    "cheesecake",
    "chicken_wings",
    "tiramisu",
    "bibimbap",
    "omelette",
    "pancakes",
    "hot_dog",
    "waffles",
    "tacos",
]


# ----------------------------------------------------------------------
# STEP 0: locate the dataset folder and validate the class names
# ----------------------------------------------------------------------
# Zip files often extract into a double folder (food-101/food-101/...).
if not (SRC / "images").exists() and (SRC / "food-101" / "images").exists():
    SRC = SRC / "food-101"

assert (SRC / "images").exists(), f"Cannot find images/ inside {SRC.resolve()}"
assert (SRC / "meta" / "train.txt").exists(), "meta/train.txt is missing"

all_classes = (SRC / "meta" / "classes.txt").read_text().split()
missing = [c for c in MY_CLASSES if c not in all_classes]
assert not missing, f"These names are not in Food-101 (check spelling): {missing}"
assert len(set(MY_CLASSES)) == len(MY_CLASSES), "Duplicate class in MY_CLASSES"


# ----------------------------------------------------------------------
# STEP 1: read the official split files, keep only our classes
# ----------------------------------------------------------------------
def read_split(name):
    """Return entries like 'pizza/1001116' from meta/<name>.txt for our classes."""
    lines = (SRC / "meta" / f"{name}.txt").read_text().split()
    return [line for line in lines if line.split("/")[0] in MY_CLASSES]


train_all = read_split("train")  # official train: 750 images per class
test = read_split("test")  # official test: 250 per class, left untouched


# ----------------------------------------------------------------------
# STEP 2: cut a validation set out of the official train set
# ----------------------------------------------------------------------
# Sort first so the result does not depend on file order, then shuffle
# with a fixed seed. Same seed -> same split on every computer.
train_all.sort()
random.Random(SEED).shuffle(train_all)
n_val = int(len(train_all) * VAL_FRACTION)
val, train = train_all[:n_val], train_all[n_val:]


# ----------------------------------------------------------------------
# STEP 3: copy images into food101_subset/<split>/<class>/
# ----------------------------------------------------------------------
def copy_files(items, split_name):
    """Copy every image in `items` to DST/split_name/class/."""
    for item in items:
        cls = item.split("/")[0]
        out_dir = DST / split_name / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(SRC / "images" / f"{item}.jpg", out_dir)


print("Copying images (this takes a few minutes)...")
for items, name in [(train, "train"), (val, "val"), (test, "test")]:
    copy_files(items, name)
    print(f"  {name}: {len(items)} images")


# ----------------------------------------------------------------------
# STEP 4: save the split lists (proof that the split is fixed and shareable)
# ----------------------------------------------------------------------
with open("splits.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image", "class", "split"])
    for items, name in [(train, "train"), (val, "val"), (test, "test")]:
        for item in items:
            writer.writerow([item, item.split("/")[0], name])


# ----------------------------------------------------------------------
# STEP 6: sanity check, count images per class in each split
# ----------------------------------------------------------------------
print("\nImages per class (should be about 637 / 113 / 250):")
for items, name in [(train, "train"), (val, "val"), (test, "test")]:
    counts = Counter(item.split("/")[0] for item in items)
    print(f"  {name}: min={min(counts.values())}, max={max(counts.values())}")

print(f"\nDone. {len(MY_CLASSES)} classes, subset saved in '{DST}/'.")
