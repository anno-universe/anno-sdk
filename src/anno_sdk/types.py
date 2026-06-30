"""Data-object classes for the Anno inference SDK.

Geometry DOs
------------
Each geometry class knows its own ``annotation_type`` string and can
serialize itself to the backend wire format via :meth:`to_dict`.

* :class:`Mask2D` — polygon mask, backend type ``"polygon"``
* :class:`Box2D` — axis-aligned bounding box, backend type ``"box"``
* :class:`RotatedBox2D` — rotated bounding box, backend type ``"box"``
* :class:`Keypoint2D` — keypoint set, backend type ``"keypoint"``

Payload wrapper
---------------
:class:`Annotation` bundles a geometry DO with a class label and an
optional client-side reference.  Its :meth:`Annotation.to_dict` produces
the exact payload the backend expects for a single annotation item.

Response / result DOs
---------------------
:class:`Image`, :class:`ProjectMeta`, :class:`PaginatedResponse`,
:class:`AnnotationBatchResult`, :class:`AnnotationResultItem`,
:class:`AnnotationModifyResult`.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import TypeVar

# ---------------------------------------------------------------------------
# Geometry data objects
# ---------------------------------------------------------------------------


@dataclass
class Box2D:
    """Axis-aligned bounding box.

    Serializes to the backend ``"box"`` geometry with ``rotation=0.0``.
    """

    x: float
    y: float
    width: float
    height: float

    @property
    def annotation_type(self) -> str:
        return "box"

    def to_dict(self) -> dict:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "rotation": 0.0,
        }


@dataclass
class RotatedBox2D:
    """Rotated bounding box.

    Serializes to the backend ``"box"`` geometry including the rotation
    angle in degrees clockwise.
    """

    x: float
    y: float
    width: float
    height: float
    rotation: float

    @property
    def annotation_type(self) -> str:
        return "box"

    def to_dict(self) -> dict:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "rotation": self.rotation,
        }


@dataclass
class Mask2D:
    """Polygon mask defined by a closed list of ``[x, y]`` points."""

    points: list[list[float]]

    @property
    def annotation_type(self) -> str:
        return "polygon"

    def to_dict(self) -> dict:
        return {"points": self.points}


@dataclass
class Keypoint2D:
    """Keypoint set defined by a list of ``[x, y]`` points."""

    points: list[list[float]]

    @property
    def annotation_type(self) -> str:
        return "keypoint"

    def to_dict(self) -> dict:
        return {"points": self.points}


# ---------------------------------------------------------------------------
# Union type for annotation geometry
# ---------------------------------------------------------------------------

GeometryDO = Box2D | RotatedBox2D | Mask2D | Keypoint2D

# ---------------------------------------------------------------------------
# Annotation payload
# ---------------------------------------------------------------------------


@dataclass
class Annotation:
    """A single annotation to be uploaded or used as a modify payload.

    Bundles a geometry DO (which determines ``annotation_type``) with an
    optional integer class label and an optional client-side reference.
    """

    label: int | None
    geometry: GeometryDO
    client_ref: str | None = None

    def to_dict(self) -> dict:
        """Serialize to the backend wire format for a single annotation item."""
        payload: dict = {
            "annotation_type": self.geometry.annotation_type,
            "label": self.label,
        }
        payload[self.geometry.annotation_type] = self.geometry.to_dict()
        if self.client_ref is not None:
            payload["client_ref"] = self.client_ref
        return payload

    @classmethod
    def from_geometry(
        cls,
        geometry: GeometryDO,
        label: int | None = None,
        client_ref: str | None = None,
    ) -> Annotation:
        """Convenience constructor: ``Annotation.from_geometry(Box2D(0,0,10,10), label=1)``."""
        return cls(label=label, geometry=geometry, client_ref=client_ref)

    @classmethod
    def from_dict(cls, data: dict) -> Annotation:
        """Inverse of :meth:`to_dict`.

        Reconstructs an :class:`Annotation` (and its geometry DO) from the
        backend wire format. A ``"box"`` payload becomes a :class:`Box2D` when
        its rotation is zero/absent, otherwise a :class:`RotatedBox2D`.
        """
        return cls(
            label=data.get("label"),
            geometry=_geometry_from_dict(data["annotation_type"], data[data["annotation_type"]]),
            client_ref=data.get("client_ref"),
        )


def _geometry_from_dict(annotation_type: str, data: dict) -> GeometryDO:
    """Build a geometry DO from a wire-format geometry payload."""
    if annotation_type == "polygon":
        return Mask2D(points=data["points"])
    if annotation_type == "keypoint":
        return Keypoint2D(points=data["points"])
    if annotation_type == "box":
        rotation = data.get("rotation") or 0.0
        if rotation:
            return RotatedBox2D(
                x=data["x"],
                y=data["y"],
                width=data["width"],
                height=data["height"],
                rotation=rotation,
            )
        return Box2D(x=data["x"], y=data["y"], width=data["width"], height=data["height"])
    raise ValueError(f"Unknown annotation_type: {annotation_type!r}")


# Module-level alias kept for callers that prefer a free function over the
# classmethod (e.g. the server parsing inference responses).
_annotation_from_dict = Annotation.from_dict


# ---------------------------------------------------------------------------
# Response / result data objects
# ---------------------------------------------------------------------------

T = TypeVar("T")


@dataclass
class Image:
    """An image in a project, as returned by the inference API."""

    id: int
    file_name: str
    width: int | None
    height: int | None
    file_url: str

    @classmethod
    def from_dict(cls, data: dict) -> Image:
        return cls(
            id=data["id"],
            file_name=data["file_name"],
            width=data.get("width"),
            height=data.get("height"),
            file_url=data["file_url"],
        )


@dataclass
class ProjectMeta:
    """Project metadata, including the label mapping."""

    id: int
    name: str
    description: str
    meta_info: dict
    label_mapping: dict
    created_at: datetime.datetime
    updated_at: datetime.datetime

    @classmethod
    def from_dict(cls, data: dict) -> ProjectMeta:
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            meta_info=data["meta_info"],
            label_mapping=data["label_mapping"],
            created_at=_parse_datetime(data["created_at"]),
            updated_at=_parse_datetime(data["updated_at"]),
        )


@dataclass
class PaginatedResponse[T]:
    """Generic offset/limit paginated response."""

    count: int
    limit: int
    offset: int
    items: list[T]

    @classmethod
    def from_dict(
        cls,
        data: dict,
        item_factory: callable = lambda d: d,
    ) -> PaginatedResponse[T]:
        return cls(
            count=data["count"],
            limit=data["limit"],
            offset=data["offset"],
            items=[item_factory(item) for item in data["items"]],
        )


@dataclass
class AnnotationResultItem:
    """Per-annotation result within a batch submission."""

    client_ref: str | None
    image_id: int
    annotation_id: int | None
    status: str  # "created" | "error"
    error: str | None = None

    @classmethod
    def from_dict(cls, data: dict) -> AnnotationResultItem:
        return cls(
            client_ref=data.get("client_ref"),
            image_id=data["image_id"],
            annotation_id=data.get("annotation_id"),
            status=data["status"],
            error=data.get("error"),
        )

    @property
    def is_success(self) -> bool:
        return self.status == "created"


@dataclass
class AnnotationBatchResult:
    """Result of a batch annotation upload."""

    created: int
    failed: int
    results: list[AnnotationResultItem]

    @classmethod
    def from_dict(cls, data: dict) -> AnnotationBatchResult:
        return cls(
            created=data["created"],
            failed=data["failed"],
            results=[AnnotationResultItem.from_dict(r) for r in data["results"]],
        )


@dataclass
class AnnotationModifyResult:
    """Result of modifying an existing annotation."""

    id: int
    image_id: int
    annotation_type: str
    label: int | None
    data: dict
    is_active: bool
    created_at: datetime.datetime
    modified_at: datetime.datetime

    @classmethod
    def from_dict(cls, data: dict) -> AnnotationModifyResult:
        return cls(
            id=data["id"],
            image_id=data["image_id"],
            annotation_type=data["annotation_type"],
            label=data.get("label"),
            data=data["data"],
            is_active=data["is_active"],
            created_at=_parse_datetime(data["created_at"]),
            modified_at=_parse_datetime(data["modified_at"]),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_datetime(value: str) -> datetime.datetime:
    """Parse an ISO-8601 datetime string, handling both ``Z`` and offset suffixes."""
    return datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
