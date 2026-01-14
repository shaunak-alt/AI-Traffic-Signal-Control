"""
YOLOv8 Vehicle Detection Module
================================
Modern vehicle detection using Ultralytics YOLOv8.
Replaces the legacy darkflow/YOLOv2 implementation.

Usage:
    python vehicle_detection_v8.py                    # Process test_images/
    python vehicle_detection_v8.py --video video.mp4  # Process video
    python vehicle_detection_v8.py --camera 0         # Live camera
"""

import cv2
import os
import argparse
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("Installing ultralytics...")
    os.system("pip install ultralytics")
    from ultralytics import YOLO


# Vehicle classes in COCO dataset
VEHICLE_CLASSES = {
    2: 'car',
    3: 'motorcycle',  # bike
    5: 'bus',
    7: 'truck',
}

# Custom class mapping (if you train custom model)
CUSTOM_CLASSES = {
    0: 'car',
    1: 'bike',
    2: 'bus',
    3: 'truck',
    4: 'rickshaw',
}


class VehicleDetectorV8:
    """YOLOv8-based vehicle detector."""
    
    def __init__(self, model_path="yolov8n.pt", confidence=0.3, custom_model=False):
        """
        Initialize detector.
        
        Args:
            model_path: Path to YOLO model (default: yolov8n.pt - auto-downloads)
            confidence: Minimum confidence threshold
            custom_model: If True, use custom class mapping
        """
        print(f"Loading YOLOv8 model: {model_path}")
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.custom_model = custom_model
        self.classes = CUSTOM_CLASSES if custom_model else VEHICLE_CLASSES
        
    def detect(self, image):
        """
        Detect vehicles in an image.
        
        Args:
            image: numpy array (BGR) or path to image
            
        Returns:
            List of detections: [{'label': str, 'confidence': float, 'bbox': (x1,y1,x2,y2)}]
        """
        results = self.model(image, conf=self.confidence, verbose=False)
        
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                
                # Check if it's a vehicle class
                if cls_id in self.classes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    
                    detections.append({
                        'label': self.classes[cls_id],
                        'confidence': conf,
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'topleft': {'x': int(x1), 'y': int(y1)},
                        'bottomright': {'x': int(x2), 'y': int(y2)},
                    })
        
        return detections
    
    def count_vehicles(self, image):
        """
        Count vehicles by type.
        
        Returns:
            dict: {'car': N, 'bus': N, 'truck': N, 'bike': N, 'rickshaw': N}
        """
        detections = self.detect(image)
        
        counts = {'car': 0, 'bike': 0, 'bus': 0, 'truck': 0, 'rickshaw': 0, 'motorcycle': 0}
        for d in detections:
            label = d['label']
            if label in counts:
                counts[label] += 1
        
        # Map motorcycle to bike
        counts['bike'] += counts.pop('motorcycle', 0)
        
        return counts
    
    def draw_detections(self, image, detections=None):
        """Draw bounding boxes on image."""
        if detections is None:
            detections = self.detect(image)
        
        img = image.copy()
        
        colors = {
            'car': (0, 255, 0),      # Green
            'bus': (255, 0, 0),      # Blue
            'truck': (0, 0, 255),    # Red
            'bike': (255, 255, 0),   # Cyan
            'motorcycle': (255, 255, 0),
            'rickshaw': (0, 255, 255),  # Yellow
        }
        
        for d in detections:
            x1, y1, x2, y2 = d['bbox']
            label = d['label']
            conf = d['confidence']
            color = colors.get(label, (0, 255, 0))
            
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"{label} {conf:.2f}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return img


def process_images(input_dir, output_dir, detector):
    """Process all images in a directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    for img_file in input_path.glob("*"):
        if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            print(f"Processing: {img_file.name}")
            
            img = cv2.imread(str(img_file))
            detections = detector.detect(img)
            counts = detector.count_vehicles(img)
            
            print(f"  Detected: {counts}")
            
            # Draw and save
            output_img = detector.draw_detections(img, detections)
            output_file = output_path / f"output_{img_file.name}"
            cv2.imwrite(str(output_file), output_img)
            print(f"  Saved: {output_file}")


def process_video(video_path, detector):
    """Process video file."""
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        detections = detector.detect(frame)
        counts = detector.count_vehicles(frame)
        
        # Draw detections
        frame = detector.draw_detections(frame, detections)
        
        # Show counts
        y = 30
        for label, count in counts.items():
            if count > 0:
                cv2.putText(frame, f"{label}: {count}", (10, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y += 25
        
        cv2.imshow("YOLOv8 Vehicle Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


def live_camera(camera_id, detector):
    """Live camera detection."""
    cap = cv2.VideoCapture(camera_id)
    
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        counts = detector.count_vehicles(frame)
        frame = detector.draw_detections(frame)
        
        # Show counts
        y = 30
        total = sum(counts.values())
        cv2.putText(frame, f"Total Vehicles: {total}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        y += 30
        
        for label, count in counts.items():
            if count > 0:
                cv2.putText(frame, f"{label}: {count}", (10, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y += 25
        
        cv2.imshow("YOLOv8 Vehicle Detection - Press Q to quit", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Vehicle Detection")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model path")
    parser.add_argument("--confidence", type=float, default=0.3, help="Confidence threshold")
    parser.add_argument("--video", help="Video file to process")
    parser.add_argument("--camera", type=int, help="Camera ID for live detection")
    parser.add_argument("--input", default="test_images", help="Input image directory")
    parser.add_argument("--output", default="output_images", help="Output image directory")
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = VehicleDetectorV8(
        model_path=args.model,
        confidence=args.confidence
    )
    
    if args.camera is not None:
        live_camera(args.camera, detector)
    elif args.video:
        process_video(args.video, detector)
    else:
        process_images(args.input, args.output, detector)
        print("\nDone! Check output_images/ for results.")
