# anno-sdk

Python SDK for Anno project API and inference integrations.

## Project API client

```python
from anno_sdk import Annotation, Box2D, Client

client = Client(base_url="http://localhost:8000", api_key="ak_...")

image = client.upload_image("cat.png", content_type="image/png")

annotation = Annotation(label=1, geometry=Box2D(10, 20, 100, 50))
result = client.upload_annotations(image_id=image.id, annotations=[annotation])
```

`Client.upload_image()` sends `multipart/form-data` with the `file` field to
`/api/project-api/images` and returns an `Image` object containing the uploaded
image ID, filename, width, and height.
