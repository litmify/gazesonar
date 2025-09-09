import cv2
import numpy as np
from ultralytics import YOLO
import logging

class YOLODetector:
    def __init__(self, model_name='yolov8n.pt', confidence_threshold=0.5, debug=False):
        logging.basicConfig(level=logging.DEBUG if debug else logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.logger.debug(f"Initializing YOLODetector with model: {model_name}, confidence: {confidence_threshold}")
        self.model = YOLO(model_name)
        self.confidence_threshold = confidence_threshold
        self.logger.info(f"Loaded YOLO model: {model_name}")
        self.logger.debug(f"Model classes: {list(self.model.names.values())[:10]}...")  # Show first 10 classes
        
    def detect_objects(self, frame):
        if frame is None:
            self.logger.debug("Frame is None, returning empty detections")
            return []
        
        self.logger.debug(f"Running YOLO on frame: shape={frame.shape}")
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        
        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                self.logger.debug(f"Found {len(boxes)} boxes in result")
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.model.names[class_id]
                    
                    detection = {
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'confidence': float(confidence),
                        'class_id': class_id,
                        'class_name': class_name,
                        'center': (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    }
                    detections.append(detection)
                    self.logger.debug(f"Detection: {class_name} at {detection['bbox']} with conf={confidence:.3f}")
            else:
                self.logger.debug("No boxes found in result")
        
        self.logger.debug(f"Total detections: {len(detections)}")
        return detections
    
    def get_object_at_gaze(self, frame, gaze_position):
        if frame is None or gaze_position is None:
            self.logger.debug(f"get_object_at_gaze: frame={frame is not None}, gaze={gaze_position}")
            return None
        
        self.logger.debug(f"Checking for object at gaze position: {gaze_position}")
        detections = self.detect_objects(frame)
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            if x1 <= gaze_position[0] <= x2 and y1 <= gaze_position[1] <= y2:
                self.logger.debug(f"Gaze hit object: {detection['class_name']} at {detection['bbox']}")
                return detection
        
        self.logger.debug("No object found at gaze position")
        return None
    
    def find_closest_object(self, gaze_position, detections, max_distance=100):
        if not detections or gaze_position is None:
            self.logger.debug(f"find_closest_object: no detections or gaze is None")
            return None, None
        
        self.logger.debug(f"Finding closest object to gaze {gaze_position} within {max_distance}px")
        closest_object = None
        min_distance = float('inf')
        
        for detection in detections:
            center = detection['center']
            distance = np.sqrt((center[0] - gaze_position[0])**2 + 
                             (center[1] - gaze_position[1])**2)
            self.logger.debug(f"  {detection['class_name']} at {center}: distance={distance:.1f}px")
            
            if distance < min_distance and distance <= max_distance:
                min_distance = distance
                closest_object = detection
        
        if closest_object:
            self.logger.debug(f"Closest object: {closest_object['class_name']} at {min_distance:.1f}px")
        else:
            self.logger.debug("No object within max distance")
        
        return closest_object, min_distance if closest_object else (None, None)
    
    def draw_detections(self, frame, detections, gaze_position=None, gazed_object=None):
        if frame is None:
            self.logger.debug("draw_detections: frame is None")
            return frame
        
        self.logger.debug(f"Drawing {len(detections)} detections, gaze={gaze_position is not None}, gazed_object={gazed_object is not None}")
        
        annotated_frame = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            color = (0, 255, 0) if detection == gazed_object else (255, 0, 0)
            thickness = 3 if detection == gazed_object else 2
            
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
            
            label = f"{class_name}: {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            label_y = y1 - 10 if y1 - 10 > 10 else y1 + label_size[1] + 10
            
            cv2.rectangle(annotated_frame, (x1, label_y - label_size[1] - 10),
                        (x1 + label_size[0], label_y), color, -1)
            cv2.putText(annotated_frame, label, (x1, label_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        if gaze_position:
            cv2.circle(annotated_frame, gaze_position, 25, (0, 255, 0), 3)
            cv2.circle(annotated_frame, gaze_position, 3, (0, 255, 0), -1)
        
        return annotated_frame


if __name__ == "__main__":
    import sys
    debug = '--debug' in sys.argv
    detector = YOLODetector(debug=debug)
    
    cap = cv2.VideoCapture(0)
    
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        detections = detector.detect_objects(frame)
        
        annotated_frame = detector.draw_detections(frame, detections)
        
        cv2.imshow("YOLO Detection Test", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()