"""Format conversion helpers for exporting annotation data.

Provides :func:`to_coco` and :func:`to_yolo` conversion functions that
take lists of :class:`Image` and :class:`Annotation` objects and produce
dictionaries ready for serialization (e.g. as JSON / text files).
"""

from __future__ import annotations

from .types import Annotation, Image


def _invert_label_mapping(label_mapping: dict) -> dict[int, str]:
    """Extract ``{class_id: class_name}`` from a project label mapping."""
    result: dict[int, str] = {}

    labels = label_mapping.get("labels") if isinstance(label_mapping.get("labels"), dict) else None
    source = labels if labels is not None else label_mapping

    for name, value in source.items():
        if isinstance(value, dict):
            class_id = value.get("id") if "id" in value else value.get("class_id")
            if isinstance(class_id, int):
                result[class_id] = name
        elif isinstance(value, int):
            result[value] = name

    return result


def _category_metadata(label_mapping: dict) -> dict[int, dict]:
    """Extract optional supercategory and inherited keypoint metadata by id."""
    labels = label_mapping.get("labels") if isinstance(label_mapping.get("labels"), dict) else None
    source = labels if labels is not None else label_mapping
    supercategories = label_mapping.get("supercategories", {})
    result: dict[int, dict] = {}
    for value in source.values():
        if not isinstance(value, dict):
            continue
        class_id = value.get("id") if "id" in value else value.get("class_id")
        if not isinstance(class_id, int):
            continue
        supercategory = value.get("supercategory")
        keypoints = value.get("keypoints")
        if not keypoints and isinstance(supercategory, str):
            parent = supercategories.get(supercategory, {})
            if isinstance(parent, dict):
                keypoints = parent.get("keypoints")
        metadata: dict = {}
        if isinstance(supercategory, str):
            metadata["supercategory"] = supercategory
        if isinstance(keypoints, list) and all(isinstance(name, str) for name in keypoints):
            metadata["keypoints"] = keypoints
        result[class_id] = metadata
    return result


# ---------------------------------------------------------------------------
# COCO
# ---------------------------------------------------------------------------


def to_coco(
    images: list[Image],
    annotations_by_image: dict[int, list[Annotation]],
    label_mapping: dict,
) -> dict:
    """Build a COCO-format dictionary.

    Parameters:
        images: List of :class:`Image` objects to include.
        annotations_by_image: Annotations keyed by ``image.id``.
        label_mapping: Project label mapping dict (from :meth:`Client.get_meta`).

    Returns:
        A dict with ``"images"``, ``"annotations"``, and ``"categories"`` keys
        following the COCO format specification.
    """
    label_names = _invert_label_mapping(label_mapping)
    category_metadata = _category_metadata(label_mapping)

    coco_images: list[dict] = []
    coco_annotations: list[dict] = []
    coco_categories: list[dict] = []

    for image in images:
        coco_images.append(
            {
                "id": image.id,
                "file_name": image.file_name,
                "width": image.width,
                "height": image.height,
            }
        )

    for cat_id in sorted(label_names.keys()):
        metadata = category_metadata.get(cat_id, {})
        category = {
            "id": cat_id,
            "name": label_names[cat_id],
            "supercategory": metadata.get("supercategory", ""),
        }
        if "keypoints" in metadata:
            category["keypoints"] = metadata["keypoints"]
        coco_categories.append(category)

    annotation_id = 1
    for image in images:
        for annotation in annotations_by_image.get(image.id, []):
            bbox = None
            seg = None
            area = 0.0
            keypoints = None
            num_keypoints = None
            geometry = annotation.geometry

            if geometry.annotation_type == "polygon":
                pts = geometry.points
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                bbox = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
                seg = [[c for p in pts for c in p]]
                n = len(xs)
                area = 0.5 * abs(
                    sum(xs[i] * ys[(i + 1) % n] - xs[(i + 1) % n] * ys[i] for i in range(n))
                )

            elif geometry.annotation_type == "box":
                corners = geometry.to_corners()
                xs = [c[0] for c in corners]
                ys = [c[1] for c in corners]
                bbox = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
                area = geometry.width * geometry.height
                if getattr(geometry, "rotation", 0) != 0:
                    seg = [[c for corner in corners for c in corner]]

            elif geometry.annotation_type == "keypoint":
                pts = geometry.points
                labelled = [p for p in pts if p[2] > 0]
                if labelled:
                    xs = [p[0] for p in labelled]
                    ys = [p[1] for p in labelled]
                    bbox = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
                    area = bbox[2] * bbox[3]
                else:
                    bbox = [0, 0, 0, 0]
                keypoints = [coordinate for kp in pts for coordinate in kp]
                num_keypoints = len(labelled)

            coco_ann = {
                "id": annotation_id,
                "image_id": image.id,
                "category_id": annotation.label,
                "bbox": bbox or [0, 0, 0, 0],
                "area": area,
                "iscrowd": 0,
            }
            if seg:
                coco_ann["segmentation"] = seg
            if keypoints is not None:
                coco_ann["keypoints"] = keypoints
                coco_ann["num_keypoints"] = num_keypoints

            coco_annotations.append(coco_ann)
            annotation_id += 1

    return {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": coco_categories,
    }


# ---------------------------------------------------------------------------
# YOLO
# ---------------------------------------------------------------------------


def to_yolo(
    images: list[Image],
    annotations_by_image: dict[int, list[Annotation]],
    label_mapping: dict,
) -> dict[str, str]:
    """Build YOLO-format label files.

    Parameters:
        images: List of :class:`Image` objects to include.
        annotations_by_image: Annotations keyed by ``image.id``.
        label_mapping: Project label mapping dict (from :meth:`Client.get_meta`).

    Returns:
        A dict mapping filename to file content:

        * ``"classes.txt"`` — one class name per line
        * ``"labels/{image_stem}.txt"`` — one annotation per line for each image

        Axis-aligned boxes are written as ``class cx cy w h`` (center-normalized).
        Rotated boxes are written as ``class x1 y1 x2 y2 x3 y3 x4 y4`` (OBB eight-point,
        corners from :meth:`RotatedBox2D.to_corners`).
        Polygons are written as ``class x1 y1 x2 y2 ...`` (normalized vertices).
        Keypoints are **skipped**.
    """
    label_names = _invert_label_mapping(label_mapping)

    image_lines: dict[int, list[str]] = {}
    for image in images:
        image_lines[image.id] = []

    for image in images:
        for annotation in annotations_by_image.get(image.id, []):
            label = annotation.label
            if label is None:
                continue
            if image.width is None or image.height is None:
                continue
            W, H = image.width, image.height
            geometry = annotation.geometry

            if geometry.annotation_type == "polygon":
                pts = geometry.points
                parts = [str(label)]
                for p in pts:
                    parts.append(f"{p[0] / W:.6f}")
                    parts.append(f"{p[1] / H:.6f}")
                image_lines[image.id].append(" ".join(parts))

            elif geometry.annotation_type == "box":
                corners = geometry.to_corners()
                if getattr(geometry, "rotation", 0) == 0:
                    cx = (geometry.x + geometry.width / 2) / W
                    cy = (geometry.y + geometry.height / 2) / H
                    nw = geometry.width / W
                    nh = geometry.height / H
                    image_lines[image.id].append(f"{label} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                else:
                    parts = [str(label)]
                    for c in corners:
                        parts.append(f"{c[0] / W:.6f}")
                        parts.append(f"{c[1] / H:.6f}")
                    image_lines[image.id].append(" ".join(parts))

            elif geometry.annotation_type == "keypoint":
                continue

    result: dict[str, str] = {}
    result["classes.txt"] = "".join(
        f"{label_names[cat_id]}\n" for cat_id in sorted(label_names.keys())
    )

    for image in images:
        stem = _file_stem(image.file_name)
        lines = image_lines.get(image.id, [])
        result[f"labels/{stem}.txt"] = "\n".join(lines) + ("\n" if lines else "")

    return result


def _file_stem(file_name: str) -> str:
    """Return the filename without extension (e.g. ``"image"`` from ``"image.jpg"``)."""
    idx = file_name.rfind(".")
    return file_name if idx == -1 else file_name[:idx]
