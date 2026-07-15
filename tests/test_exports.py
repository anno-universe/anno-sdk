"""Tests for COCO / YOLO export helpers."""

from __future__ import annotations

import math

from anno_sdk import (
    Annotation,
    Box2D,
    Image,
    Keypoint2D,
    Polygon2D,
    RotatedBox2D,
    to_coco,
    to_yolo,
)

# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

SAMPLE_IMAGES = [
    Image(id=1, file_name="img_001.jpg", width=640, height=480),
    Image(id=2, file_name="img_002.png", width=800, height=600),
]

LABEL_MAPPING = {
    "labels": {
        "cat": {"id": 0},
        "dog": {"id": 1},
        "bird": {"id": 2},
    }
}

# ---------------------------------------------------------------------------
# to_coco
# ---------------------------------------------------------------------------


class TestToCoco:
    def test_basic_structure(self) -> None:
        annotations = {
            1: [Annotation(label=0, geometry=Box2D(10, 20, 100, 50))],
        }
        result = to_coco(SAMPLE_IMAGES[:1], annotations, LABEL_MAPPING)
        assert "images" in result
        assert "annotations" in result
        assert "categories" in result
        assert len(result["images"]) == 1
        assert len(result["annotations"]) == 1
        assert len(result["categories"]) == 3

    def test_image_fields(self) -> None:
        result = to_coco(SAMPLE_IMAGES[:1], {}, LABEL_MAPPING)
        img = result["images"][0]
        assert img["id"] == 1
        assert img["file_name"] == "img_001.jpg"
        assert img["width"] == 640
        assert img["height"] == 480

    def test_axis_aligned_box(self) -> None:
        annotations = {
            1: [Annotation(label=0, geometry=Box2D(10, 20, 100, 50))],
        }
        result = to_coco(SAMPLE_IMAGES[:1], annotations, LABEL_MAPPING)
        ann = result["annotations"][0]
        assert ann["bbox"] == [10, 20, 100, 50]
        assert ann["area"] == 5000.0
        assert ann["category_id"] == 0
        assert ann["image_id"] == 1
        assert "segmentation" not in ann

    def test_rotated_box(self) -> None:
        annotations = {
            1: [Annotation(label=1, geometry=RotatedBox2D(10, 20, 100, 50, rotation=45))],
        }
        result = to_coco(SAMPLE_IMAGES[:1], annotations, LABEL_MAPPING)
        ann = result["annotations"][0]
        assert ann["category_id"] == 1
        # Rotated box should have segmentation (corners as polygon)
        assert "segmentation" in ann
        seg = ann["segmentation"][0]
        assert len(seg) == 8  # 4 corners * 2 coords

    def test_polygon(self) -> None:
        annotations = {
            2: [Annotation(label=2, geometry=Polygon2D([[0, 0], [100, 0], [100, 100], [0, 100]]))],
        }
        result = to_coco(SAMPLE_IMAGES[1:], annotations, LABEL_MAPPING)
        ann = result["annotations"][0]
        assert ann["bbox"] == [0, 0, 100, 100]
        assert ann["area"] == 10000.0  # 0.5 * |400 + 400| = 400... wait: shoelace formula
        # Shoelace: sum(x[i]*y[i+1] - x[i+1]*y[i]) =
        # 0*0 + 100*100 + 100*100 + 0*0 - (0*100 + 0*100 + 100*0 + 100*0) =
        # 0 + 10000 + 10000 + 0 - 0 = 20000, area = 10000
        assert "segmentation" in ann

    def test_keypoint(self) -> None:
        annotations = {
            1: [Annotation(label=2, geometry=Keypoint2D([[100, 200], [300, 400]]))],
        }
        result = to_coco(SAMPLE_IMAGES[:1], annotations, LABEL_MAPPING)
        ann = result["annotations"][0]
        assert ann["keypoints"] == [100, 200, 2, 300, 400, 2]
        assert ann["num_keypoints"] == 2

    def test_annotation_ids_sequential(self) -> None:
        annotations = {
            1: [
                Annotation(label=0, geometry=Box2D(0, 0, 10, 10)),
                Annotation(label=1, geometry=Box2D(10, 10, 20, 20)),
            ],
        }
        result = to_coco(SAMPLE_IMAGES[:1], annotations, LABEL_MAPPING)
        assert result["annotations"][0]["id"] == 1
        assert result["annotations"][1]["id"] == 2

    def test_empty_annotations(self) -> None:
        result = to_coco(SAMPLE_IMAGES, {}, LABEL_MAPPING)
        assert result["annotations"] == []
        assert len(result["images"]) == 2

    def test_categories_sorted_by_id(self) -> None:
        result = to_coco(SAMPLE_IMAGES[:1], {}, LABEL_MAPPING)
        category_ids = [c["id"] for c in result["categories"]]
        assert category_ids == [0, 1, 2]

    def test_flat_label_mapping(self) -> None:
        flat_mapping = {"cat": 0, "dog": 1}
        result = to_coco(SAMPLE_IMAGES[:1], {}, flat_mapping)
        assert len(result["categories"]) == 2
        assert result["categories"][0]["name"] == "cat"

    def test_label_none_is_preserved(self) -> None:
        annotations = {
            1: [Annotation(label=None, geometry=Box2D(10, 10, 50, 50))],
        }
        result = to_coco(SAMPLE_IMAGES[:1], annotations, LABEL_MAPPING)
        assert result["annotations"][0]["category_id"] is None


# ---------------------------------------------------------------------------
# to_yolo
# ---------------------------------------------------------------------------


class TestToYolo:
    def test_basic_structure(self) -> None:
        annotations = {
            1: [Annotation(label=0, geometry=Box2D(0, 0, 100, 100))],
        }
        result = to_yolo(SAMPLE_IMAGES[:1], annotations, LABEL_MAPPING)
        assert "classes.txt" in result
        assert "labels/img_001.txt" in result

    def test_classes_file(self) -> None:
        result = to_yolo(SAMPLE_IMAGES[:1], {}, LABEL_MAPPING)
        lines = result["classes.txt"].strip().split("\n")
        assert lines == ["cat", "dog", "bird"]

    def test_axis_aligned_box(self) -> None:
        annotations = {
            1: [
                Annotation(label=0, geometry=Box2D(10, 20, 100, 50)),
                Annotation(label=1, geometry=Box2D(300, 200, 200, 100)),
            ],
        }
        result = to_yolo(SAMPLE_IMAGES[:1], annotations, LABEL_MAPPING)
        lines = result["labels/img_001.txt"].strip().split("\n")
        assert len(lines) == 2
        assert lines[0] == "0 0.093750 0.093750 0.156250 0.104167"
        assert lines[1] == "1 0.625000 0.520833 0.312500 0.208333"

    def test_rotated_box(self) -> None:
        annotations = {
            1: [Annotation(label=0, geometry=RotatedBox2D(0, 0, 100, 100, rotation=45))],
        }
        result = to_yolo(SAMPLE_IMAGES[:1], annotations, LABEL_MAPPING)
        line = result["labels/img_001.txt"].strip()
        parts = line.split()
        assert parts[0] == "0"
        # Should have 9 values: class + 8 coords (4 corners)
        assert len(parts) == 9

    def test_polygon(self) -> None:
        annotations = {
            2: [Annotation(label=2, geometry=Polygon2D([[0, 0], [800, 0], [800, 600]]))],
        }
        result = to_yolo(SAMPLE_IMAGES[1:], annotations, LABEL_MAPPING)
        line = result["labels/img_002.txt"].strip()
        parts = line.split()
        assert parts[0] == "2"
        assert len(parts) == 7  # class + 3 points * 2 coords

    def test_keypoint_skipped(self) -> None:
        annotations = {
            1: [Annotation(label=2, geometry=Keypoint2D([[100, 200]]))],
        }
        result = to_yolo(SAMPLE_IMAGES[:1], annotations, LABEL_MAPPING)
        assert result["labels/img_001.txt"] == ""

    def test_label_none_skipped(self) -> None:
        annotations = {
            1: [Annotation(label=None, geometry=Box2D(10, 10, 50, 50))],
        }
        result = to_yolo(SAMPLE_IMAGES[:1], annotations, LABEL_MAPPING)
        assert result["labels/img_001.txt"] == ""

    def test_skip_image_without_dimensions(self) -> None:
        img = Image(id=3, file_name="no_dims.jpg", width=None, height=None)
        annotations = {
            3: [Annotation(label=0, geometry=Box2D(0, 0, 100, 100))],
        }
        result = to_yolo([img], annotations, LABEL_MAPPING)
        assert result["labels/no_dims.txt"] == ""

    def test_multiple_images(self) -> None:
        annotations = {
            1: [Annotation(label=0, geometry=Box2D(0, 0, 100, 100))],
            2: [Annotation(label=1, geometry=Box2D(0, 0, 200, 200))],
        }
        result = to_yolo(SAMPLE_IMAGES, annotations, LABEL_MAPPING)
        assert "labels/img_001.txt" in result
        assert "labels/img_002.txt" in result
        assert result["labels/img_001.txt"].strip()
        assert result["labels/img_002.txt"].strip()

    def test_file_stem_with_multiple_dots(self) -> None:
        img = Image(id=99, file_name="my.image.name.jpg", width=100, height=100)
        annotations = {
            99: [Annotation(label=0, geometry=Box2D(0, 0, 10, 10))],
        }
        result = to_yolo([img], annotations, LABEL_MAPPING)
        assert "labels/my.image.name.txt" in result

    def test_flat_label_mapping(self) -> None:
        flat_mapping = {"car": 5, "truck": 7}
        result = to_yolo(SAMPLE_IMAGES[:1], {}, flat_mapping)
        lines = result["classes.txt"].strip().split("\n")
        assert lines == ["car", "truck"]


# ---------------------------------------------------------------------------
# to_corners (via types)
# ---------------------------------------------------------------------------


class TestToCorners:
    def test_box2d_axis_aligned(self) -> None:
        box = Box2D(10, 20, 100, 50)
        corners = box.to_corners()
        assert corners == [[10, 20], [110, 20], [110, 70], [10, 70]]

    def test_rotated_box_zero_rotation(self) -> None:
        box = RotatedBox2D(10, 20, 100, 50, rotation=0)
        corners = box.to_corners()
        assert corners == [[10, 20], [110, 20], [110, 70], [10, 70]]

    def test_rotated_box_90deg(self) -> None:
        box = RotatedBox2D(x=10, y=20, width=100, height=50, rotation=90)
        corners = box.to_corners()
        assert len(corners) == 4
        # 90deg CCW: TL at (10,20) new position is:
        # cx=60, cy=45, dx=-50, dy=-25
        # rx=60 + (-50*0 - (-25)*1) = 85, ry=45 + (-50*1 + (-25)*0) = -5
        assert abs(corners[0][0] - 85) < 1e-9
        assert abs(corners[0][1] - -5) < 1e-9

    def test_rotated_box_square_45deg(self) -> None:
        box = RotatedBox2D(x=0, y=0, width=100, height=100, rotation=45)
        corners = box.to_corners()
        # All corners should be same distance from center (50, 50)
        distances = [math.hypot(c[0] - 50, c[1] - 50) for c in corners]
        expected = math.sqrt(2) * 50  # ~70.71
        for d in distances:
            assert abs(d - expected) < 1e-6

    def test_rotated_box_small_angle_is_negligible(self) -> None:
        box = RotatedBox2D(x=10, y=20, width=100, height=50, rotation=1e-10)
        corners = box.to_corners()
        # Effectively axis-aligned
        assert abs(corners[0][0] - 10) < 1e-6
        assert abs(corners[0][1] - 20) < 1e-6
