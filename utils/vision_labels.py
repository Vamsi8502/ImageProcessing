# utils/vision_labels.py

from google.cloud import vision
from google.cloud.vision_v1.types.image_annotator import AnnotateImageResponse

def get_image_labels(image_path: str) -> list:
    client = vision.ImageAnnotatorClient()

    with open(image_path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response: AnnotateImageResponse = client.label_detection(image=image)
    labels = response.label_annotations

    return [label.description for label in labels]
