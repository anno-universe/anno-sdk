"""Tests for the Client class with HTTP mocking."""

from __future__ import annotations

import re

import httpx
import pytest
from pytest_httpx import HTTPXMock

from anno_sdk import (
    AnnoAPIError,
    AnnoConnectionError,
    Annotation,
    Box2D,
    Client,
    Image,
    Keypoint2D,
    PaginatedResponse,
    Polygon2D,
    ProjectMeta,
    RotatedBox2D,
)

BASE_URL = "http://anno.example.com"
API_KEY = "ak_deadbeef.secretsecretsecretsecretsecret"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client() -> Client:
    return Client(base_url=BASE_URL, api_key=API_KEY)


# ---------------------------------------------------------------------------
# GET /meta
# ---------------------------------------------------------------------------


META_RESPONSE = {
    "id": 1,
    "name": "Test Project",
    "description": "A test",
    "meta_info": {},
    "label_mapping": {"cat": 0, "dog": 1},
    "created_at": "2025-01-15T10:30:00Z",
    "updated_at": "2025-06-01T12:00:00Z",
}


def test_get_meta(client: Client, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=f"{BASE_URL}/api/project-api/meta",
        json=META_RESPONSE,
    )
    meta = client.get_meta()
    assert isinstance(meta, ProjectMeta)
    assert meta.name == "Test Project"
    assert meta.label_mapping == {"cat": 0, "dog": 1}


def test_get_meta_unauthorized(client: Client, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=f"{BASE_URL}/api/project-api/meta",
        status_code=401,
        text="Unauthorized",
    )
    with pytest.raises(AnnoAPIError) as exc:
        client.get_meta()
    assert exc.value.status_code == 401


# ---------------------------------------------------------------------------
# GET /images
# ---------------------------------------------------------------------------


IMAGE_ITEMS = [
    {"id": 1, "file_name": "a.jpg", "width": 640, "height": 480},
    {"id": 2, "file_name": "b.jpg", "width": 800, "height": 600},
]


def test_paginate_images(client: Client, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=re.compile(rf"^{re.escape(BASE_URL)}/api/project-api/images(\?.*)?$"),
        json={"count": 2, "limit": 100, "offset": 0, "items": IMAGE_ITEMS},
    )
    page = client.paginate_images()
    assert isinstance(page, PaginatedResponse)
    assert page.count == 2
    assert len(page.items) == 2
    assert page.items[0].id == 1
    assert isinstance(page.items[0], Image)


def test_paginate_images_with_filter(client: Client, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=re.compile(rf"^{re.escape(BASE_URL)}/api/project-api/images(\?.*)?$"),
        json={"count": 0, "limit": 50, "offset": 10, "items": []},
    )
    page = client.paginate_images(limit=50, offset=10, has_active_annotations=True)
    assert page.count == 0
    # Verify query params were sent
    req = httpx_mock.get_request()
    assert req is not None
    assert "has_active_annotations=true" in str(req.url)


def test_paginate_images_exclude_annotated(client: Client, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=re.compile(rf"^{re.escape(BASE_URL)}/api/project-api/images(\?.*)?$"),
        json={"count": 0, "limit": 100, "offset": 0, "items": []},
    )
    client.paginate_images(has_active_annotations=False)
    req = httpx_mock.get_request()
    assert "has_active_annotations=false" in str(req.url)


# ---------------------------------------------------------------------------
# iter_images
# ---------------------------------------------------------------------------


_IMAGES_URL = re.compile(rf"^{re.escape(BASE_URL)}/api/project-api/images(\?.*)?$")


def test_iter_images_single_page(client: Client, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=_IMAGES_URL,
        json={"count": 2, "limit": 100, "offset": 0, "items": IMAGE_ITEMS},
    )
    images = list(client.iter_images())
    assert len(images) == 2
    assert images[0].file_name == "a.jpg"
    assert images[1].file_name == "b.jpg"


def test_iter_images_multi_page(client: Client, httpx_mock: HTTPXMock) -> None:
    # Page 1
    httpx_mock.add_response(
        url=_IMAGES_URL,
        json={"count": 3, "limit": 2, "offset": 0, "items": IMAGE_ITEMS},
    )
    # Page 2
    httpx_mock.add_response(
        url=_IMAGES_URL,
        json={
            "count": 3,
            "limit": 2,
            "offset": 2,
            "items": [
                {
                    "id": 3,
                    "file_name": "c.jpg",
                    "width": 100,
                    "height": 100,
                },
            ],
        },
    )
    images = list(client.iter_images(limit=2))
    assert len(images) == 3
    assert [img.file_name for img in images] == ["a.jpg", "b.jpg", "c.jpg"]


def test_iter_images_empty(client: Client, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=_IMAGES_URL,
        json={"count": 0, "limit": 100, "offset": 0, "items": []},
    )
    assert list(client.iter_images()) == []


# ---------------------------------------------------------------------------
# GET /images/{id}
# ---------------------------------------------------------------------------


def test_get_image(client: Client, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=f"{BASE_URL}/api/project-api/images/42",
        json={
            "id": 42,
            "file_name": "cat.png",
            "width": 300,
            "height": 200,
        },
    )
    img = client.get_image(42)
    assert img.id == 42
    assert img.file_name == "cat.png"


def test_get_image_not_found(client: Client, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(url=f"{BASE_URL}/api/project-api/images/999", status_code=404)
    with pytest.raises(AnnoAPIError) as exc:
        client.get_image(999)
    assert exc.value.status_code == 404


# ---------------------------------------------------------------------------
# POST /images
# ---------------------------------------------------------------------------


UPLOAD_IMAGE_RESPONSE = {
    "id": 123,
    "project_id": 1,
    "file_name": "sdk-upload.png",
    "width": 32,
    "height": 24,
    "tags": [],
}


def test_upload_image_from_path(client: Client, httpx_mock: HTTPXMock, tmp_path) -> None:
    image_path = tmp_path / "sdk-upload.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    httpx_mock.add_response(
        url=f"{BASE_URL}/api/project-api/images",
        status_code=201,
        json=UPLOAD_IMAGE_RESPONSE,
    )

    image = client.upload_image(image_path, content_type="image/png")

    assert isinstance(image, Image)
    assert image.id == 123
    assert image.file_name == "sdk-upload.png"
    assert (image.width, image.height) == (32, 24)
    req = httpx_mock.get_request()
    assert req is not None
    assert req.method == "POST"
    assert req.headers["X-API-Key"] == API_KEY
    assert req.headers["content-type"].startswith("multipart/form-data")
    body = req.read()
    assert b'name="file"' in body
    assert b'filename="sdk-upload.png"' in body
    assert b"Content-Type: image/png" in body
    assert b"\x89PNG\r\n\x1a\nfake" in body


def test_upload_image_with_filename_override(
    client: Client, httpx_mock: HTTPXMock, tmp_path
) -> None:
    image_path = tmp_path / "original.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    httpx_mock.add_response(
        url=f"{BASE_URL}/api/project-api/images",
        status_code=201,
        json=UPLOAD_IMAGE_RESPONSE,
    )

    image = client.upload_image(
        image_path,
        filename="sdk-upload.png",
        content_type="image/png",
    )

    assert image.id == 123
    body = httpx_mock.get_request().read()
    assert b'filename="sdk-upload.png"' in body


def test_upload_image_error(client: Client, httpx_mock: HTTPXMock, tmp_path) -> None:
    image_path = tmp_path / "invalid.png"
    image_path.write_bytes(b"not an image")
    httpx_mock.add_response(
        url=f"{BASE_URL}/api/project-api/images",
        status_code=400,
        text="Invalid image file.",
    )

    with pytest.raises(AnnoAPIError) as exc:
        client.upload_image(image_path)
    assert exc.value.status_code == 400


def test_upload_image_network_error(client: Client, httpx_mock: HTTPXMock, tmp_path) -> None:
    image_path = tmp_path / "sdk-upload.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    httpx_mock.add_exception(httpx.ConnectError("Connection refused"))

    with pytest.raises(AnnoConnectionError):
        client.upload_image(image_path)


# ---------------------------------------------------------------------------
# GET /images/{id}/original_file
# ---------------------------------------------------------------------------


def test_get_image_file(client: Client, httpx_mock: HTTPXMock) -> None:
    png_bytes = b"\x89PNG\r\n\x1a\nfake"
    httpx_mock.add_response(
        url=f"{BASE_URL}/api/project-api/images/1/original_file",
        content=png_bytes,
        headers={"content-type": "image/png"},
    )
    data = client.get_image_file(1)
    assert data == png_bytes


def test_get_image_file_error(client: Client, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=f"{BASE_URL}/api/project-api/images/1/original_file",
        status_code=403,
    )
    with pytest.raises(AnnoAPIError):
        client.get_image_file(1)


def test_get_image_file_follows_redirect(client: Client, httpx_mock: HTTPXMock) -> None:
    file_url = "https://files.example.com/image.png"
    httpx_mock.add_response(
        url=f"{BASE_URL}/api/project-api/images/1/original_file",
        status_code=307,
        headers={"location": file_url},
    )
    httpx_mock.add_response(url=file_url, content=b"image bytes")

    assert client.get_image_file(1) == b"image bytes"
    requests = httpx_mock.get_requests()
    assert requests[0].headers["X-API-Key"] == API_KEY
    assert "X-API-Key" not in requests[1].headers


def test_iter_image_file(client: Client, httpx_mock: HTTPXMock) -> None:
    png_bytes = b"\x89PNG\r\n\x1a\nfake"
    httpx_mock.add_response(
        url=f"{BASE_URL}/api/project-api/images/1/original_file",
        content=png_bytes,
        headers={"content-type": "image/png"},
    )
    chunks = client.iter_image_file(1)
    assert b"".join(chunks) == png_bytes


def test_iter_image_file_network_error(client: Client, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_exception(httpx.ReadError("Connection interrupted"))

    with pytest.raises(AnnoConnectionError):
        list(client.iter_image_file(1))


def test_iter_image_file_error(client: Client, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=f"{BASE_URL}/api/project-api/images/1/original_file",
        status_code=403,
    )
    with pytest.raises(AnnoAPIError):
        for _ in client.iter_image_file(1):
            pass


# ---------------------------------------------------------------------------
# POST /images/{id}/annotations
# ---------------------------------------------------------------------------


BATCH_RESPONSE = {
    "created": 2,
    "failed": 1,
    "results": [
        {"client_ref": "r1", "image_id": 1, "annotation_id": 100, "status": "created"},
        {"client_ref": "r2", "image_id": 1, "annotation_id": 101, "status": "created"},
        {
            "client_ref": "r3",
            "image_id": 1,
            "annotation_id": None,
            "status": "error",
            "error": "bad geometry",
        },
    ],
}


def test_upload_annotations(client: Client, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=f"{BASE_URL}/api/project-api/images/1/annotations",
        json=BATCH_RESPONSE,
    )
    annotations = [
        Annotation.from_geometry(Box2D(0, 0, 10, 10), label=0, client_ref="r1"),
        Annotation.from_geometry(Box2D(10, 0, 20, 10), label=1, client_ref="r2"),
        Annotation.from_geometry(Polygon2D([[0, 0]]), label=None, client_ref="r3"),
    ]
    result = client.upload_annotations(image_id=1, annotations=annotations)
    assert result.created == 2
    assert result.failed == 1
    assert len(result.results) == 3
    assert result.results[0].is_success
    assert not result.results[2].is_success

    # Verify the request body was serialized correctly
    body = httpx_mock.get_request().read().decode()
    assert '"annotation_type"' in body
    assert '"box"' in body
    assert '"polygon"' in body


def test_upload_annotations_all_geometry_types(client: Client, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=f"{BASE_URL}/api/project-api/images/1/annotations",
        json={"created": 3, "failed": 0, "results": []},
    )
    annotations = [
        Annotation.from_geometry(Box2D(0, 0, 1, 1), label=0),
        Annotation.from_geometry(RotatedBox2D(0, 0, 1, 1, 30), label=1),
        Annotation.from_geometry(Polygon2D([[0, 0]]), label=2),
        Annotation.from_geometry(Keypoint2D([[1, 1, 2]]), label=3),
    ]
    result = client.upload_annotations(image_id=1, annotations=annotations)
    assert result.created == 3

    # Verify body contains all annotation types
    body = httpx_mock.get_request().read().decode()
    assert '"box"' in body
    assert '"polygon"' in body
    assert '"keypoint"' in body


# ---------------------------------------------------------------------------
# PATCH /images/{id}/annotations/{aid}
# ---------------------------------------------------------------------------


MODIFY_RESPONSE = {
    "id": 200,
    "image_id": 1,
    "annotation_type": "box",
    "label": 1,
    "data": {"x": 5, "y": 5, "width": 50, "height": 50, "rotation": 0.0},
    "is_active": True,
    "created_at": "2025-06-15T08:00:00Z",
    "modified_at": "2025-06-15T08:01:00Z",
}


def test_modify_annotation(client: Client, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=f"{BASE_URL}/api/project-api/images/1/annotations/100",
        json=MODIFY_RESPONSE,
    )
    ann = Annotation.from_geometry(Box2D(5, 5, 50, 50), label=1)
    result = client.modify_annotation(image_id=1, annotation_id=100, annotation=ann)
    assert result.id == 200
    assert result.annotation_type == "box"
    assert result.is_active is True

    # Verify the body
    req = httpx_mock.get_request()
    assert req.method == "PATCH"


def test_modify_annotation_not_found(client: Client, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=f"{BASE_URL}/api/project-api/images/1/annotations/999",
        status_code=404,
    )
    ann = Annotation.from_geometry(Box2D(0, 0, 1, 1), label=0)
    with pytest.raises(AnnoAPIError) as exc:
        client.modify_annotation(image_id=1, annotation_id=999, annotation=ann)
    assert exc.value.status_code == 404


# ---------------------------------------------------------------------------
# GET /images/{id}/annotations
# ---------------------------------------------------------------------------


ANNOTATIONS_LIST = [
    {
        "id": 1,
        "image_id": 42,
        "annotation_type": "box",
        "label": 0,
        "data": {"x": 10, "y": 20, "width": 100, "height": 50, "rotation": 0.0},
        "is_active": True,
        "created_at": "2025-06-01T10:00:00Z",
        "modified_at": "2025-06-01T10:00:00Z",
    },
    {
        "id": 2,
        "image_id": 42,
        "annotation_type": "polygon",
        "label": 1,
        "data": {"points": [[0, 0], [10, 0], [10, 10]]},
        "is_active": True,
        "created_at": "2025-06-01T10:00:00Z",
        "modified_at": "2025-06-01T10:00:00Z",
    },
    {
        "id": 3,
        "image_id": 42,
        "annotation_type": "box",
        "label": None,
        "data": {"x": 5, "y": 5, "width": 20, "height": 30, "rotation": 45.0},
        "is_active": True,
        "created_at": "2025-06-01T10:00:00Z",
        "modified_at": "2025-06-01T10:00:00Z",
    },
    {
        "id": 4,
        "image_id": 42,
        "annotation_type": "keypoint",
        "label": 2,
        "data": {"points": [[1.5, 2.5, 2], [3.0, 4.0, 1]]},
        "is_active": True,
        "created_at": "2025-06-01T10:00:00Z",
        "modified_at": "2025-06-01T10:00:00Z",
    },
]


def test_get_annotations(client: Client, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=f"{BASE_URL}/api/project-api/images/42/annotations",
        json=ANNOTATIONS_LIST,
    )
    annotations = client.get_annotations(image_id=42)
    assert len(annotations) == 4

    # Box2D (rotation=0)
    assert isinstance(annotations[0].geometry, Box2D)
    assert annotations[0].label == 0
    assert annotations[0].geometry.x == 10

    # Polygon2D
    assert isinstance(annotations[1].geometry, Polygon2D)
    assert annotations[1].label == 1
    assert annotations[1].geometry.points == [[0, 0], [10, 0], [10, 10]]

    # RotatedBox2D (rotation=45)
    assert isinstance(annotations[2].geometry, RotatedBox2D)
    assert annotations[2].label is None
    assert annotations[2].geometry.rotation == 45.0

    # Keypoint2D
    assert isinstance(annotations[3].geometry, Keypoint2D)
    assert annotations[3].label == 2


def test_get_annotations_empty(client: Client, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=f"{BASE_URL}/api/project-api/images/99/annotations",
        json=[],
    )
    annotations = client.get_annotations(image_id=99)
    assert annotations == []


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_network_error_triggers_connection_error(client: Client, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_exception(httpx.ConnectError("Connection refused"))
    with pytest.raises(AnnoConnectionError):
        client.get_meta()


def test_timeout_triggers_connection_error(client: Client, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_exception(httpx.ReadTimeout("timeout"))
    with pytest.raises(AnnoConnectionError):
        client.get_meta()


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


def test_context_manager() -> None:
    c = Client(base_url=BASE_URL, api_key=API_KEY)
    assert not c._http.is_closed
    with c:
        pass
    assert c._http.is_closed


def test_explicit_close(client: Client) -> None:
    assert not client._http.is_closed
    client.close()
    assert client._http.is_closed


# ---------------------------------------------------------------------------
# Header / auth
# ---------------------------------------------------------------------------


def test_api_key_sent_in_header(client: Client, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=f"{BASE_URL}/api/project-api/meta",
        json=META_RESPONSE,
    )
    client.get_meta()
    req = httpx_mock.get_request()
    assert req is not None
    assert req.headers["X-API-Key"] == API_KEY
