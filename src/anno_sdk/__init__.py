"""Anno SDK — Python client for the Anno annotation platform inference API.

Usage::

    from anno_sdk import Client, Annotation, Box2D, Mask2D

    client = Client(base_url="http://localhost:8000", api_key="ak_...")
    meta = client.get_meta()

    # Upload a box annotation
    ann = Annotation(label=1, geometry=Box2D(10, 20, 100, 50))
    result = client.upload_annotations(image_id=42, annotations=[ann])

    # Iterate over all images
    for img in client.iter_images():
        print(img.file_name)
"""

from .client import Client
from .exceptions import AnnoAPIError, AnnoConnectionError, AnnoSDKError
from .handler import PredictFn, Predictor, serve_predict
from .inference import InferenceRequestMeta, InferenceResponse
from .types import (
    Annotation,
    AnnotationBatchResult,
    AnnotationModifyResult,
    AnnotationResultItem,
    Box2D,
    GeometryDO,
    Image,
    Keypoint2D,
    Mask2D,
    PaginatedResponse,
    ProjectMeta,
    RotatedBox2D,
)

__all__ = [
    "Client",
    # Geometry
    "Box2D",
    "RotatedBox2D",
    "Mask2D",
    "Keypoint2D",
    "GeometryDO",
    # Payload
    "Annotation",
    # Responses / results
    "Image",
    "ProjectMeta",
    "PaginatedResponse",
    "AnnotationBatchResult",
    "AnnotationResultItem",
    "AnnotationModifyResult",
    # Server-driven inference contract
    "InferenceRequestMeta",
    "InferenceResponse",
    "Predictor",
    "serve_predict",
    "PredictFn",
    # Exceptions
    "AnnoSDKError",
    "AnnoAPIError",
    "AnnoConnectionError",
]
