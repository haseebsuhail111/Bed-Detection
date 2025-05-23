import os
import cv2
from ultralytics import YOLO

def crop_bed(image_path):
    # Create output directory if it doesn't exist
    output_dir = "cropped_images"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image '{image_path}'.")
        return
    
    # Load the pretrained YOLOv8 model (this uses a model trained on COCO)
    model = YOLO("yolov8s.pt")
    
    # Run inference on the image
    results = model(image_path)
    result = results[0]  # In case of single image, get the first result
    
    # Flag to check if a bed was detected
    bed_found = False
    
    # Iterate over each detection in the results 
    # (each detection box data is: [x1, y1, x2, y2, conf, cls])
    for idx, box in enumerate(result.boxes.data.tolist()):
        x1, y1, x2, y2, conf, cls = box
        
        # Get the class label name using the model's result mapping
        label = result.names[int(cls)]
        
        # Check if the detected object is "bed" (case-insensitive)
        if label.lower() == "bed":
            bed_found = True
            
            # Convert coordinates to integers
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Crop the detected bed from the image
            cropped = image[y1:y2, x1:x2]
            
            # Set base filename and extension
            base_filename = f"bed_{idx}"
            extension = ".jpg"
            output_path = os.path.join(output_dir, base_filename + extension)
            
            # Check if output file exists and, if so, create a unique filename by adding a counter
            counter = 1
            while os.path.exists(output_path):
                output_path = os.path.join(output_dir, f"{base_filename}_{counter}{extension}")
                counter += 1
            
            # Save cropped image
            cv2.imwrite(output_path, cropped)
            print(f"Cropped bed saved at: {output_path}")
    
    if not bed_found:
        print("No bed was detected in the image.")

if __name__ == '__main__':
    # Replace with the actual path to your image file.
    image_path = r"dataset/no_reformado/row_206_image_1.jpg"
    crop_bed(image_path)
