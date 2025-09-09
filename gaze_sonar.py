import cv2
import numpy as np
import time
import logging
from tobii_capture import TobiiFrameCapture
from yolo_detector import YOLODetector

class GazeSonar:
    def __init__(self, tobii_address=None, yolo_model='yolov8n.pt', confidence_threshold=0.5, debug=False):
        logging.basicConfig(level=logging.DEBUG if debug else logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.debug = debug
        
        self.logger.debug(f"Initializing GazeSonar: tobii={tobii_address}, model={yolo_model}, conf={confidence_threshold}")
        self.tobii_capture = TobiiFrameCapture(tobii_address, debug=debug)
        self.yolo_detector = YOLODetector(yolo_model, confidence_threshold, debug=debug)
        
        self.gaze_history = []
        self.history_size = 5
        
        self.stats = {
            'frames_processed': 0,
            'objects_detected': 0,
            'gaze_hits': 0
        }
        
    def start(self):
        self.logger.debug("Starting GazeSonar system...")
        if not self.tobii_capture.connect():
            self.logger.error("Failed to connect to Tobii Glasses")
            return False
        
        self.logger.info("GazeSonar system started successfully")
        self.logger.debug(f"Gaze history size: {self.history_size}")
        return True
    
    def process_frame(self):
        self.logger.debug("Processing frame...")
        frame = self.tobii_capture.get_raw_frame()
        if frame is None:
            self.logger.debug("No frame captured")
            return None, None, None, None
        
        gaze_data = self.tobii_capture.get_gaze_position()
        gaze_position = None
        
        if gaze_data:
            height, width = frame.shape[:2]
            gaze_x = int(gaze_data['normalized'][0] * width)
            gaze_y = int(gaze_data['normalized'][1] * height)
            gaze_position = (gaze_x, gaze_y)
            self.logger.debug(f"Gaze position: {gaze_position} (frame size: {width}x{height})")
            
            self.gaze_history.append(gaze_position)
            if len(self.gaze_history) > self.history_size:
                self.gaze_history.pop(0)
            self.logger.debug(f"Gaze history length: {len(self.gaze_history)}")
        else:
            self.logger.debug("No gaze data available")
        
        detections = self.yolo_detector.detect_objects(frame)
        self.logger.debug(f"Detected {len(detections)} objects")
        
        gazed_object = None
        if gaze_position:
            gazed_object = self.yolo_detector.get_object_at_gaze(frame, gaze_position)
            
            if not gazed_object and detections:
                self.logger.debug("No direct gaze hit, finding closest object...")
                closest, distance = self.yolo_detector.find_closest_object(
                    gaze_position, detections, max_distance=150
                )
                if closest:
                    self.logger.debug(f"Found closest object: {closest['class_name']} at {distance:.1f}px")
                    gazed_object = closest
        
        self.stats['frames_processed'] += 1
        self.stats['objects_detected'] = len(detections)
        if gazed_object:
            self.stats['gaze_hits'] += 1
            self.logger.debug(f"Gaze hit #{self.stats['gaze_hits']}: {gazed_object['class_name']}")
        
        return frame, gaze_position, detections, gazed_object
    
    def get_smoothed_gaze(self):
        if not self.gaze_history:
            self.logger.debug("No gaze history for smoothing")
            return None
        
        avg_x = sum(pos[0] for pos in self.gaze_history) / len(self.gaze_history)
        avg_y = sum(pos[1] for pos in self.gaze_history) / len(self.gaze_history)
        smoothed = (int(avg_x), int(avg_y))
        self.logger.debug(f"Smoothed gaze: {smoothed} from {len(self.gaze_history)} points")
        return smoothed
    
    def visualize(self, frame, gaze_position, detections, gazed_object):
        if frame is None:
            return None
        
        annotated_frame = self.yolo_detector.draw_detections(
            frame, detections, gaze_position, gazed_object
        )
        
        info_y = 30
        cv2.putText(annotated_frame, f"FPS: {self.stats.get('fps', 0):.1f}", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        info_y += 25
        cv2.putText(annotated_frame, f"Objects: {len(detections)}", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if gazed_object:
            info_y += 25
            cv2.putText(annotated_frame, f"Looking at: {gazed_object['class_name']}", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            info_y += 25
            cv2.putText(annotated_frame, f"Confidence: {gazed_object['confidence']:.2f}", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        smoothed_gaze = self.get_smoothed_gaze()
        if smoothed_gaze and len(self.gaze_history) > 1:
            for i in range(1, len(self.gaze_history)):
                cv2.line(annotated_frame, self.gaze_history[i-1], self.gaze_history[i],
                        (0, 255, 255), 2)
        
        return annotated_frame
    
    def run(self, display=True, save_video=False, output_path='output.mp4'):
        if not self.start():
            return
        
        video_writer = None
        if save_video:
            self.logger.debug(f"Setting up video writer for {output_path}")
            frame = self.tobii_capture.get_raw_frame()
            if frame is not None:
                height, width = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))
                self.logger.info(f"Recording video to {output_path} ({width}x{height})")
        
        self.logger.info("Starting main loop. Press 'q' to quit")
        
        prev_time = time.time()
        
        try:
            while True:
                current_time = time.time()
                fps = 1 / (current_time - prev_time) if prev_time else 0
                prev_time = current_time
                self.stats['fps'] = fps
                
                frame, gaze_position, detections, gazed_object = self.process_frame()
                
                if frame is not None:
                    self.logger.debug(f"Frame {self.stats['frames_processed']}: FPS={fps:.1f}")
                    visualized_frame = self.visualize(frame, gaze_position, detections, gazed_object)
                    
                    if display and visualized_frame is not None:
                        cv2.imshow("GazeSonar - Tobii + YOLO", visualized_frame)
                    
                    if save_video and video_writer and visualized_frame is not None:
                        video_writer.write(visualized_frame)
                        self.logger.debug("Frame written to video")
                    
                    if gazed_object:
                        self.logger.info(f"Looking at: {gazed_object['class_name']} "
                                       f"(confidence: {gazed_object['confidence']:.2f})")
                else:
                    self.logger.debug("Skipping frame - no data")
                
                if display and cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
        except Exception as e:
            self.logger.error(f"Error in main loop: {e}")
            self.logger.debug(f"Exception type: {type(e).__name__}", exc_info=True)
        finally:
            if video_writer:
                video_writer.release()
            cv2.destroyAllWindows()
            self.tobii_capture.disconnect()
            
            self.logger.info(f"Session stats:")
            self.logger.info(f"  Frames processed: {self.stats['frames_processed']}")
            self.logger.info(f"  Gaze hits: {self.stats['gaze_hits']}")
            if self.stats['frames_processed'] > 0:
                hit_rate = (self.stats['gaze_hits'] / self.stats['frames_processed']) * 100
                self.logger.info(f"  Hit rate: {hit_rate:.1f}%")
    
    def stop(self):
        self.tobii_capture.disconnect()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='GazeSonar - Tobii Glasses + YOLO Object Detection')
    parser.add_argument('--tobii-address', type=str, default=None,
                       help='IP address of Tobii Glasses (auto-discover if not provided)')
    parser.add_argument('--yolo-model', type=str, default='yolov8n.pt',
                       help='YOLO model to use (default: yolov8n.pt)')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Confidence threshold for object detection (default: 0.5)')
    parser.add_argument('--no-display', action='store_true',
                       help='Run without display window')
    parser.add_argument('--save-video', action='store_true',
                       help='Save output video')
    parser.add_argument('--output', type=str, default='gaze_sonar_output.mp4',
                       help='Output video path (default: gaze_sonar_output.mp4)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    
    args = parser.parse_args()
    
    gaze_sonar = GazeSonar(
        tobii_address=args.tobii_address,
        yolo_model=args.yolo_model,
        confidence_threshold=args.confidence,
        debug=args.debug
    )
    
    gaze_sonar.run(
        display=not args.no_display,
        save_video=args.save_video,
        output_path=args.output
    )


if __name__ == "__main__":
    main()