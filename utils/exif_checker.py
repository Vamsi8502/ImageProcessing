from PIL import Image
from PIL.ExifTags import TAGS
from datetime import datetime

def extract_exif_data(image_path):
    try:
        img = Image.open(image_path)
        exif_data = img._getexif()
        if not exif_data:
            return None
        extracted = {}
        for tag_id, value in exif_data.items():
            tag = TAGS.get(tag_id, tag_id)
            extracted[tag] = value
        return extracted
    except Exception:
        return None

def get_datetime_original(exif):
    try:
        return datetime.strptime(exif["DateTimeOriginal"], "%Y:%m:%d %H:%M:%S")
    except:
        return None

def has_gps_data(exif):
    return 'GPSInfo' in exif if exif else False
