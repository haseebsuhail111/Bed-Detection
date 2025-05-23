# Bed Detection and Cropping using YOLOv8

This project uses the YOLOv8 object detection model from the [Ultralytics](https://github.com/ultralytics/ultralytics) library to detect beds in an image and crop them out. Cropped bed images are saved in a `cropped_images` directory.

## 🖼️ Example Use Case

Given a room image, this script automatically detects any beds and saves cropped versions of those beds for further analysis or classification.

---

## 📦 Requirements

- Python 3.8+
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- OpenCV

Install all dependencies using:

```bash
pip install -r requirements.txt
