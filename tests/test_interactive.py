"""Tests for the interactive inference contract (interactive.py)."""

from __future__ import annotations

from anno_sdk import (
    Annotation,
    InteractiveInferenceRequestMeta,
    InteractiveInferenceResponse,
    Polygon2D,
)


class TestInteractiveInferenceRequestMeta:
    def test_roundtrip(self) -> None:
        meta = InteractiveInferenceRequestMeta(
            image_id=42,
            session_id=7,
            step_index=3,
            prompts=[
                {"type": "box", "x": 10, "y": 20, "width": 100, "height": 50},
                {"type": "positive_point", "x": 55, "y": 40},
                {"type": "negative_point", "x": 5, "y": 5},
                {"type": "text", "text": "the cat"},
            ],
            label_mapping={"cat": 0},
            requested_types=["polygon"],
            width=1920,
            height=1080,
            client_ref="ref-1",
        )
        restored = InteractiveInferenceRequestMeta.from_dict(meta.to_dict())
        assert restored == meta

    def test_from_dict_tolerates_missing_optionals(self) -> None:
        meta = InteractiveInferenceRequestMeta.from_dict(
            {"image_id": 1, "session_id": 2, "step_index": 1}
        )
        assert meta.prompts == []
        assert meta.label_mapping == {}
        assert meta.requested_types == []
        assert meta.width is None and meta.client_ref is None


class TestInteractiveInferenceResponse:
    def test_roundtrip_with_candidate(self) -> None:
        resp = InteractiveInferenceResponse(
            annotation=Annotation(label=0, geometry=Polygon2D([[0, 0], [1, 1], [2, 0]])),
            score=0.97,
            model_version="sam2-hiera-large",
            raw={"logits": [1, 2, 3]},
        )
        restored = InteractiveInferenceResponse.from_dict(resp.to_dict())
        assert restored.score == 0.97
        assert restored.model_version == "sam2-hiera-large"
        assert restored.raw == {"logits": [1, 2, 3]}
        assert restored.annotation is not None
        assert restored.annotation.to_dict() == resp.annotation.to_dict()

    def test_roundtrip_empty_candidate(self) -> None:
        resp = InteractiveInferenceResponse(annotation=None)
        restored = InteractiveInferenceResponse.from_dict(resp.to_dict())
        assert restored.annotation is None
        assert restored.score is None

    def test_from_dict_empty(self) -> None:
        restored = InteractiveInferenceResponse.from_dict({})
        assert restored.annotation is None
        assert restored.raw == {}
