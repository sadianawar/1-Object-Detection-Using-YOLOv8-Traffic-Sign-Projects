import os
from ultralytics import YOLO
import cv2

IMAGES_DIR = os.path.join('TrafficSignLocalizationandDetection/test', 'images')
OUTPUT_DIR = os.path.join(IMAGES_DIR, 'output_images')

# Create the output folder if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

image_path = os.path.join(IMAGES_DIR, '000314_jpg.rf.292060647eb54be88844b1956a97d4c1.jpg')  # Change the image file name
# output_image_path = '{}_out.jpg'.format(os.path.splitext(image_path)[0])
output_image_path = os.path.join(OUTPUT_DIR, '{}_out.jpg'.format(os.path.splitext(image_path)[0]))

# Read the input image
frame = cv2.imread(image_path)
H, W, _ = frame.shape

model_path = os.path.join('.', 'runs', 'detect', 'train2', 'weights', 'last.pt')

# Load the YOLOv8 model
model = YOLO(model_path)

threshold = 0.5

# Perform object detection on the image
results = model(frame)[0]

# Process the detection results
for result in results.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = result

    if score > threshold:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
        cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

# Save the output image with bounding boxes
cv2.imwrite(output_image_path, frame)

cv2.destroyAllWindows()
