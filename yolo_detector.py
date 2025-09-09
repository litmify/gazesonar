import cv2
import numpy as np
from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_name='yolo11m.pt', confidence_threshold=0.3):
        self.model = YOLO(model_name)
        self.confidence_threshold = confidence_threshold
        print(f"Loaded YOLO model: {model_name}")
        
    def detect_objects(self, frame):
        if frame is None:
            return []
        
        # Run YOLO
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        
        detections = []
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.model.names[class_id]
                    
                    detections.append({
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'confidence': confidence,
                        'class_id': class_id,
                        'class_name': class_name,
                        'center': (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    })
        
        return detections
    
    def get_object_at_gaze(self, detections, gaze_position):
        if not detections or gaze_position is None:
            return None
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            if x1 <= gaze_position[0] <= x2 and y1 <= gaze_position[1] <= y2:
                return detection
        
        return None
    
    def find_closest_object(self, gaze_position, detections, max_distance=100):
        if not detections or gaze_position is None:
            return None, None
        
        closest_object = None
        min_distance = float('inf')
        
        for detection in detections:
            center = detection['center']
            distance = np.sqrt((center[0] - gaze_position[0])**2 + 
                             (center[1] - gaze_position[1])**2)
            
            if distance < min_distance and distance <= max_distance:
                min_distance = distance
                closest_object = detection
        
        return closest_object, min_distance if closest_object else (None, None)
    
    def draw_detections(self, frame, detections, gaze_position=None, gazed_object=None):
        if frame is None:
            return frame
        
        annotated_frame = frame.copy()
        
        # Draw all detection boxes
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            # Green if gazed at, blue otherwise
            color = (0, 255, 0) if detection == gazed_object else (255, 0, 0)
            thickness = 3 if detection == gazed_object else 2
            
            # Draw box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
            label_y = y1 - 10 if y1 - 10 > 10 else y1 + label_size[1] + 10
            
            cv2.rectangle(annotated_frame, (x1, label_y - label_size[1] - 10),
                        (x1 + label_size[0], label_y), color, -1)
            cv2.putText(annotated_frame, label, (x1, label_y - 5),
                       cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw gaze circle
        if gaze_position:
            cv2.circle(annotated_frame, gaze_position, 25, (0, 255, 0), 3)
            cv2.circle(annotated_frame, gaze_position, 3, (0, 255, 0), -1)
        
        return annotated_frame