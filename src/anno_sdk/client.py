"""Anno project API client."""

from __future__ import annotations

from collections.abc import Iterator
from os import PathLike
from pathlib import Path
from typing import Any

import httpx

from .exceptions import AnnoAPIError, AnnoConnectionError
from .types import (
    Annotation,
    AnnotationBatchResult,
    AnnotationModifyResult,
    Image,
    PaginatedResponse,
    ProjectMeta,
)

# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class Client:
    """HTTP client for the Anno project API.

    Authenticates with a per-project API key via the ``X-API-Key`` header.
    All methods return deserialized data-objects, not raw dicts.

    Supports use as a context manager::

        with Client(base_url="http://localhost:8000", api_key="ak_...") as client:
            meta = client.get_meta()
            ...

    Parameters:
        base_url: Root URL of the Anno server (e.g. ``http://localhost:8000``).
        api_key: Plaintext project API key (``ak_XXXXXXXX.yyyy...``).
        timeout: HTTP request timeout in seconds (default 30).
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        api_url = httpx.URL(self.base_url)
        self._api_origin = (api_url.scheme, api_url.host, api_url.port)
        self._http = httpx.Client(
            base_url=self.base_url,
            headers={"X-API-Key": api_key},
            timeout=timeout,
            event_hooks={"request": [self._protect_api_key]},
        )

    # -- context manager ---------------------------------------------------

    def __enter__(self) -> Client:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._http.close()

    # -- helpers -----------------------------------------------------------

    def _protect_api_key(self, request: httpx.Request) -> None:
        """Keep project credentials off cross-origin redirect requests."""
        request_origin = (request.url.scheme, request.url.host, request.url.port)
        if request_origin != self._api_origin:
            request.headers.pop("X-API-Key", None)

    def _request(self, method: str, path: str, **kwargs: Any) -> dict | list:
        """Issue an HTTP request and return the parsed JSON body.

        Raises:
            AnnoAPIError: On HTTP 4xx / 5xx.
            AnnoConnectionError: On network / timeout errors.
        """
        try:
            response = self._http.request(method, path, **kwargs)
        except httpx.RequestError as exc:
            raise AnnoConnectionError(str(exc)) from exc

        if response.status_code >= 400:
            detail = response.text
            raise AnnoAPIError(response.status_code, detail)

        # Some endpoints (e.g. image file) return non-JSON — handled by the
        # calling method, but for JSON endpoints we parse here.
        if response.headers.get("content-type", "").startswith("application/json"):
            return response.json()
        return response.content  # type: ignore[return-value]

    def _get(self, path: str, **params: Any) -> Any:
        return self._request("GET", path, params=params)

    def _post(self, path: str, *, json: dict | list) -> Any:
        return self._request("POST", path, json=json)

    def _patch(self, path: str, *, json: dict) -> Any:
        return self._request("PATCH", path, json=json)

    # -- project meta ------------------------------------------------------

    def get_meta(self) -> ProjectMeta:
        """Return project metadata including the label mapping."""
        data = self._get("/api/project-api/meta")
        return ProjectMeta.from_dict(data)

    # -- images ------------------------------------------------------------

    def paginate_images(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        has_active_annotations: bool | None = None,
    ) -> PaginatedResponse[Image]:
        """List images in the project with offset/limit pagination.

        Parameters:
            limit: Page size (1–500, clamped by the server).
            offset: Number of images to skip.
            has_active_annotations:
                ``True`` — only images that have at least one active annotation.
                ``False`` — only images with zero active annotations.
                ``None`` — all images (default).
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if has_active_annotations is not None:
            params["has_active_annotations"] = str(has_active_annotations).lower()

        data = self._get("/api/project-api/images", **params)
        return PaginatedResponse.from_dict(data, item_factory=Image.from_dict)

    def iter_images(
        self,
        *,
        limit: int = 100,
        has_active_annotations: bool | None = None,
    ) -> Iterator[Image]:
        """Yield every image in the project, auto-advancing offset.

        Parameters:
            limit: Number of images per HTTP request (page size, 1–500).
            has_active_annotations: Optional filter (see :meth:`paginate_images`).

        Yields:
            :class:`Image` instances one at a time.
        """
        offset = 0
        while True:
            page = self.paginate_images(
                limit=limit,
                offset=offset,
                has_active_annotations=has_active_annotations,
            )
            yield from page.items
            offset += limit
            if offset >= page.count:
                break

    def get_image(self, image_id: int) -> Image:
        """Get a single image by ID."""
        data = self._get(f"/api/project-api/images/{image_id}")
        return Image.from_dict(data)

    def upload_image(
        self,
        file: str | PathLike[str],
        *,
        filename: str | None = None,
        content_type: str | None = None,
    ) -> Image:
        """Upload an image file to the project.

        The backend validates that the uploaded bytes are a valid image and
        records the returned filename from the multipart part.

        Parameters:
            file: Local path.
            filename: Optional multipart filename. Defaults to the path basename.
            content_type: Optional MIME type, e.g. ``"image/png"``.
        """
        path = Path(file)
        upload_name = filename or path.name
        file_obj = path.open("rb")
        try:
            files = (
                {"file": (upload_name, file_obj, content_type)}
                if content_type is not None
                else {"file": (upload_name, file_obj)}
            )
            try:
                response = self._http.post("/api/project-api/images", files=files)
            except httpx.RequestError as exc:
                raise AnnoConnectionError(str(exc)) from exc

            if response.status_code >= 400:
                raise AnnoAPIError(response.status_code, response.text)

            return Image.from_dict(response.json())
        finally:
            file_obj.close()

    def get_image_file(self, image_id: int) -> bytes:
        """Download the original image file bytes.

        Loads the entire image into memory. For large files, prefer
        :meth:`iter_image_file` which streams in chunks.
        """
        try:
            response = self._http.get(
                f"/api/project-api/images/{image_id}/original_file",
                follow_redirects=True,
            )
        except httpx.RequestError as exc:
            raise AnnoConnectionError(str(exc)) from exc
        if response.status_code >= 400:
            raise AnnoAPIError(response.status_code, response.text)
        return response.content

    def iter_image_file(
        self,
        image_id: int,
        *,
        chunk_size: int = 10240,
    ) -> Iterator[bytes]:
        """Stream the original image file bytes in chunks.

        Returns an iterator that yields byte chunks as they are received.
        This avoids loading the entire image into memory at once, which is
        useful for large files.

        Parameters:
            image_id: The image ID.
            chunk_size: Bytes per chunk (default 10240).
        """
        try:
            with self._http.stream(
                "GET",
                f"/api/project-api/images/{image_id}/original_file",
                follow_redirects=True,
            ) as response:
                if response.status_code >= 400:
                    response.read()
                    raise AnnoAPIError(response.status_code, response.text)
                yield from response.iter_bytes(chunk_size=chunk_size)
        except httpx.RequestError as exc:
            raise AnnoConnectionError(str(exc)) from exc

    # -- annotations -------------------------------------------------------

    def get_annotations(self, image_id: int) -> list[Annotation]:
        """Retrieve all active annotations for an image.

        Returns a list of :class:`Annotation` objects with geometry data.
        Each annotation's geometry is deserialized into the appropriate DO
        (:class:`Box2D`, :class:`RotatedBox2D`, :class:`Polygon2D`, or
        :class:`Keypoint2D`).
        """
        data = self._get(f"/api/project-api/images/{image_id}/annotations")
        annotations: list[Annotation] = []
        for item in data:
            ann_dict = {
                "annotation_type": item["annotation_type"],
                "label": item.get("label"),
                item["annotation_type"]: item["data"],
            }
            annotations.append(Annotation.from_dict(ann_dict))
        return annotations

    def upload_annotations(
        self,
        image_id: int,
        annotations: list[Annotation],
    ) -> AnnotationBatchResult:
        """Submit a batch of annotations for a single image.

        Each annotation is processed independently by the backend — one
        failure does not affect the others.  Per-item errors are reported in
        the returned ``AnnotationBatchResult.results`` (with ``status="error"``)
        and are **not** raised as exceptions.

        Parameters:
            image_id: The image to annotate.
            annotations: List of :class:`Annotation` payloads.
        """
        body = {
            "annotations": [a.to_dict() for a in annotations],
        }
        data = self._post(
            f"/api/project-api/images/{image_id}/annotations",
            json=body,
        )
        return AnnotationBatchResult.from_dict(data)

    def modify_annotation(
        self,
        image_id: int,
        annotation_id: int,
        annotation: Annotation,
    ) -> AnnotationModifyResult:
        """Modify an existing annotation (immutable pattern).

        The backend creates a **new** annotation row and deactivates the old
        one (``is_active=False``), recording an audit operation linking the two.

        Parameters:
            image_id: The image the annotation belongs to.
            annotation_id: The ID of the annotation to modify (must be active).
            annotation: The replacement annotation payload.
        """
        data = self._patch(
            f"/api/project-api/images/{image_id}/annotations/{annotation_id}",
            json=annotation.to_dict(),
        )
        return AnnotationModifyResult.from_dict(data)
