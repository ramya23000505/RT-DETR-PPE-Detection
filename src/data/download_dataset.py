"""Download and normalize a real PPE dataset into this project's three-class YOLO taxonomy.

Default source: Kaggle's public "Construction Site Safety Image Dataset" by
snehilsanyal.  It is a real-image construction dataset with YOLO labels for
Hardhat, Person, and Safety Vest, plus additional classes that this script
intentionally drops.  Review the source licence and dataset terms before use.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

import yaml

DEFAULT_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "snehilsanyal/construction-site-safety-image-dataset-roboflow"
)
DEFAULT_SOURCE = "Kaggle: snehilsanyal/construction-site-safety-image-dataset-roboflow"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
TARGET_NAMES = {0: "person", 1: "hard-hat", 2: "safety-vest"}
CANONICAL_TARGET_IDS = {
    "person": 0,
    "people": 0,
    "hardhat": 1,
    "hardhats": 1,
    "helmet": 1,
    "helmets": 1,
    "safetyvest": 2,
    "safetyvests": 2,
    "vest": 2,
    "vests": 2,
}

LOGGER = logging.getLogger("ppe_dataset_download")


def normalise_class_name(name: object) -> str:
    return "".join(character for character in str(name).lower() if character.isalnum())


def download_archive(url: str, destination: Path) -> int:
    """Download a ZIP with bounded streaming and clear network errors."""
    request = urllib.request.Request(url, headers={"User-Agent": "rt-detr-ppe-dataset-downloader/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response, destination.open("wb") as output:
            downloaded = 0
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                downloaded += len(chunk)
        if downloaded == 0:
            raise RuntimeError("The dataset server returned an empty archive.")
        return downloaded
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"Dataset download failed from {url}: {exc}") from exc


def safe_extract(archive: Path, destination: Path) -> int:
    """Extract a ZIP without allowing archive entries to escape the staging directory."""
    try:
        with zipfile.ZipFile(archive) as zipped:
            for member in zipped.infolist():
                target = (destination / member.filename).resolve()
                if destination.resolve() not in target.parents and target != destination.resolve():
                    raise RuntimeError(f"Unsafe path in archive: {member.filename}")
            zipped.extractall(destination)
            return sum(not member.is_dir() for member in zipped.infolist())
    except zipfile.BadZipFile as exc:
        raise RuntimeError("Downloaded file is not a valid ZIP archive.") from exc


def find_data_yaml(extracted_root: Path) -> Path:
    candidates = sorted((*extracted_root.rglob("data.yaml"), *extracted_root.rglob("data.yml")))
    if not candidates:
        raise RuntimeError("No YOLO data.yaml was found in the downloaded archive.")
    return candidates[0]


def load_class_mapping(data_yaml: Path) -> dict[int, int]:
    try:
        metadata = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
        names = metadata["names"]
    except (OSError, KeyError, TypeError, yaml.YAMLError) as exc:
        raise RuntimeError(f"Could not read class names from {data_yaml}.") from exc

    if isinstance(names, list):
        names = dict(enumerate(names))
    if not isinstance(names, dict):
        raise RuntimeError("The source data.yaml 'names' field must be a list or mapping.")

    mapping: dict[int, int] = {}
    for source_id, source_name in names.items():
        target_id = CANONICAL_TARGET_IDS.get(normalise_class_name(source_name))
        if target_id is not None:
            mapping[int(source_id)] = target_id

    missing = set(TARGET_NAMES) - set(mapping.values())
    if missing:
        missing_names = [TARGET_NAMES[class_id] for class_id in sorted(missing)]
        raise RuntimeError(
            f"Source taxonomy does not provide all required classes: {missing_names}. "
            f"Found names: {list(names.values())}"
        )
    LOGGER.info("Source-to-target class mapping: %s", mapping)
    return mapping


def image_label_path(image_path: Path, all_labels_by_stem: dict[str, list[Path]]) -> Path | None:
    parts = list(image_path.parts)
    for index, part in enumerate(parts):
        if part.lower() == "images":
            candidate = Path(*parts[:index], "labels", *parts[index + 1:]).with_suffix(".txt")
            if candidate.is_file():
                return candidate
    sibling = image_path.with_suffix(".txt")
    if sibling.is_file():
        return sibling
    matches = all_labels_by_stem.get(image_path.stem, [])
    return matches[0] if len(matches) == 1 else None


def remap_label(label_path: Path, class_mapping: dict[int, int]) -> list[str]:
    remapped: list[str] = []
    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RuntimeError(f"Cannot read label file {label_path}.") from exc

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise RuntimeError(f"Malformed YOLO annotation in {label_path}:{line_number}.")
        try:
            source_id = int(fields[0])
            coordinates = [float(value) for value in fields[1:]]
        except ValueError as exc:
            raise RuntimeError(f"Non-numeric YOLO annotation in {label_path}:{line_number}.") from exc
        x_center, y_center, width, height = coordinates
        if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < width <= 1 and 0 < height <= 1):
            raise RuntimeError(f"Out-of-range YOLO annotation in {label_path}:{line_number}.")
        if source_id in class_mapping:
            remapped.append(f"{class_mapping[source_id]} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    return remapped


def iter_images(root: Path) -> Iterable[Path]:
    return (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def reset_destination(images_dir: Path, labels_dir: Path, replace: bool) -> None:
    existing = [path for directory in (images_dir, labels_dir) if directory.exists() for path in directory.iterdir()]
    if existing and not replace:
        raise RuntimeError(
            "Destination already contains files. Re-run with --replace only after backing up data you need."
        )
    if replace:
        for directory in (images_dir, labels_dir):
            if directory.exists():
                shutil.rmtree(directory)
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)


def clear_generated_splits(dataset_root: Path, replace: bool) -> None:
    """Avoid stale train/validation/test files after an explicitly requested replacement."""
    if not replace:
        return
    for kind in ("images", "labels"):
        for split in ("train", "val", "test"):
            split_dir = dataset_root / kind / split
            if split_dir.exists():
                shutil.rmtree(split_dir)
    for cache_file in (dataset_root / "labels").glob("*.cache"):
        cache_file.unlink()


def write_dataset_yaml(dataset_root: Path, source: str) -> None:
    payload = {
        "path": str(dataset_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": TARGET_NAMES,
        "source": source,
        "note": "Source labels remapped by src/data/download_dataset.py; validate and split before training.",
    }
    (dataset_root / "dataset.yaml").write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def download_and_extract(url: str, source: str, dataset_root: Path, replace: bool) -> dict[str, int]:
    images_dir = dataset_root / "images" / "all"
    labels_dir = dataset_root / "labels" / "all"

    with tempfile.TemporaryDirectory(prefix="ppe_download_") as temporary:
        temporary_root = Path(temporary)
        archive = temporary_root / "source.zip"
        byte_count = download_archive(url, archive)
        extracted_root = temporary_root / "extracted"
        extracted_root.mkdir()
        archive_file_count = safe_extract(archive, extracted_root)
        class_mapping = load_class_mapping(find_data_yaml(extracted_root))
        staged_images = temporary_root / "staged_images"
        staged_labels = temporary_root / "staged_labels"
        staged_images.mkdir()
        staged_labels.mkdir()
        labels_by_stem: dict[str, list[Path]] = {}
        for label in extracted_root.rglob("*.txt"):
            labels_by_stem.setdefault(label.stem, []).append(label)

        copied_images = 0
        copied_annotations = 0
        seen_destination_names: set[str] = set()
        for image in sorted(iter_images(extracted_root)):
            label = image_label_path(image, labels_by_stem)
            if label is None:
                raise RuntimeError(f"Image has no unique matching YOLO label file: {image}")
            remapped_lines = remap_label(label, class_mapping)
            # Prefix split/parent segments to avoid collisions while flattening to images/all.
            relative_key = "__".join(image.relative_to(extracted_root).with_suffix("").parts)
            destination_name = f"{relative_key}{image.suffix.lower()}"
            if destination_name in seen_destination_names:
                raise RuntimeError(f"Duplicate destination filename generated for {image}.")
            seen_destination_names.add(destination_name)
            shutil.copy2(image, staged_images / destination_name)
            (staged_labels / f"{Path(destination_name).stem}.txt").write_text("".join(remapped_lines), encoding="utf-8")
            copied_images += 1
            copied_annotations += len(remapped_lines)

        if copied_images == 0:
            raise RuntimeError("No supported images were extracted from the source archive.")
        # Existing data is untouched until download, archive validation, and label remapping all succeed.
        reset_destination(images_dir, labels_dir, replace)
        clear_generated_splits(dataset_root, replace)
        for staged_image in staged_images.iterdir():
            shutil.copy2(staged_image, images_dir / staged_image.name)
        for staged_label in staged_labels.iterdir():
            shutil.copy2(staged_label, labels_dir / staged_label.name)

    manifest = {
        "source": source,
        "download_url": url,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "downloaded_bytes": byte_count,
        "archive_files": archive_file_count,
        "images": copied_images,
        "target_annotations": copied_annotations,
        "target_taxonomy": TARGET_NAMES,
    }
    (dataset_root / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_dataset_yaml(dataset_root, source)
    return {
        "downloaded_bytes": byte_count,
        "archive_files": archive_file_count,
        "images": copied_images,
        "annotations": copied_annotations,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and remap a real PPE YOLO dataset.")
    parser.add_argument("--url", default=DEFAULT_URL, help="Direct ZIP download URL.")
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="Human-readable dataset provenance.")
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"), help="Destination dataset directory.")
    parser.add_argument("--replace", action="store_true", help="Delete existing dataset/images/all and dataset/labels/all first.")
    arguments = parser.parse_args()
    try:
        result = download_and_extract(arguments.url, arguments.source, arguments.dataset_root, arguments.replace)
    except RuntimeError as exc:
        LOGGER.error("Dataset preparation failed: %s", exc)
        return 1
    LOGGER.info(
        "Download complete: %d archive files, %d images, %d retained annotations, %.2f MB downloaded.",
        result["archive_files"], result["images"], result["annotations"], result["downloaded_bytes"] / (1024 * 1024),
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    sys.exit(main())
