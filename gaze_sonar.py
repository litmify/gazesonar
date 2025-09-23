import cv2
import numpy as np
import time
import logging
import pyttsx3
import threading
from tobii_capture import TobiiFrameCapture
from yolo_detector import YOLODetector
import settings

class GazeSonar:
    def __init__(self, tobii_address=None, yolo_model=None, confidence_threshold=None, debug=None):
        # Use settings.py values as defaults
        tobii_address = tobii_address or settings.TOBII_ADDRESS
        yolo_model = yolo_model or settings.YOLO_MODEL
        confidence_threshold = confidence_threshold or settings.CONFIDENCE_THRESHOLD
        debug = debug if debug is not None else settings.DEBUG_MODE
        
        logging.basicConfig(level=logging.DEBUG if debug else logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.debug = debug
        
        self.logger.debug(f"Initializing GazeSonar: tobii={tobii_address}, model={yolo_model}")
        self.tobii_capture = TobiiFrameCapture(tobii_address, debug=debug)
        self.yolo_detector = YOLODetector(yolo_model, confidence_threshold)
        
        self.gaze_history = []
        self.history_size = settings.GAZE_HISTORY_SIZE

        self.stats = {
            'frames_processed': 0,
            'fps': 0
        }

        # Initialize TTS
        self.tts_enabled = settings.TTS_ENABLED
        if self.tts_enabled:
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', settings.TTS_VOICE_RATE)
            self.tts_engine.setProperty('volume', settings.TTS_VOICE_VOLUME)
            self.tts_lock = threading.Lock()
            self.is_speaking = False

        # Gaze persistence tracking
        self.current_gazed_object = None
        self.gaze_start_time = None
        self.last_announcement_time = None
        self.announced_object = None

        # Frame-based TTS tracking
        self.last_frame_announcement = 0
        
        # For YOLO processing every 5th frame
        self.yolo_frame = None
        self.yolo_detections = []
        
        # Cache screen resolution
        import tkinter as tk
        root = tk.Tk()
        self.screen_width = root.winfo_screenwidth()
        self.screen_height = root.winfo_screenheight()
        root.destroy()
        
        # Pre-calculate panel dimensions
        self.panel_width = self.screen_width // 2
        self.panel_height = self.screen_height // 2
        
        # Cache for bottom panels (update less frequently)
        self.stats_panel_cache = None
        self.log_panel_cache = None
        self.panel_update_counter = 0
        
    def start(self):
        self.logger.debug("Starting GazeSonar system...")
        if not self.tobii_capture.connect():
            self.logger.error("Failed to connect to Tobii Glasses")
            return False
        
        self.logger.info("GazeSonar system started successfully")
        self.logger.debug(f"Gaze history size: {self.history_size}")
        return True
    
    def process_frame(self):
        frame = self.tobii_capture.get_raw_frame()
        if frame is None:
            return None, None
        
        gaze_data = self.tobii_capture.get_gaze_position()
        gaze_position = None
        
        if gaze_data:
            height, width = frame.shape[:2]
            gaze_x = int(gaze_data['normalized'][0] * width)
            gaze_y = int(gaze_data['normalized'][1] * height)
            gaze_position = (gaze_x, gaze_y)
            
            self.gaze_history.append(gaze_position)
            if len(self.gaze_history) > self.history_size:
                self.gaze_history.pop(0)
        
        # Process YOLO every Nth frame based on settings
        if self.stats['frames_processed'] % settings.YOLO_PROCESS_EVERY_N_FRAMES == 0:
            self.yolo_frame = frame.copy()
            self.yolo_detections = self.yolo_detector.detect_objects(self.yolo_frame)
            print(f"[YOLO Frame {self.stats['frames_processed']}] Detected {len(self.yolo_detections)} objects")

            # Frame-check TTS mode: announce detected objects
            if self.tts_enabled and settings.TTS_MODE == 'frame_check':
                self.announce_frame_objects()
        
        # Debug output every Nth frame based on settings
        if self.stats['frames_processed'] % settings.DEBUG_OUTPUT_EVERY_N_FRAMES == 0:
            print(f"[Frame {self.stats['frames_processed']}] FPS: {self.stats.get('fps', 0):.1f}, Gaze: {gaze_position}")
        
        self.stats['frames_processed'] += 1
        
        return frame, gaze_position
    
    def get_smoothed_gaze(self):
        if not self.gaze_history:
            return None
        
        avg_x = sum(pos[0] for pos in self.gaze_history) / len(self.gaze_history)
        avg_y = sum(pos[1] for pos in self.gaze_history) / len(self.gaze_history)
        return (int(avg_x), int(avg_y))
    
    def visualize_tobii(self, frame, gaze_position):
        if frame is None:
            return None
        
        annotated_frame = frame.copy()
        
        # Draw FPS
        info_y = 30
        cv2.putText(annotated_frame, f"FPS: {self.stats.get('fps', 0):.1f}", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw gaze position info
        if gaze_position:
            info_y += 25
            cv2.putText(annotated_frame, f"Gaze: ({gaze_position[0]}, {gaze_position[1]})", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw smoothed gaze trail
        smoothed_gaze = self.get_smoothed_gaze()
        if smoothed_gaze and len(self.gaze_history) > 1:
            for i in range(1, len(self.gaze_history)):
                cv2.line(annotated_frame, self.gaze_history[i-1], self.gaze_history[i],
                        settings.GAZE_TRAIL_COLOR, 2)
        
        # Draw gaze circle
        if gaze_position:
            cv2.circle(annotated_frame, gaze_position, settings.GAZE_CIRCLE_RADIUS, settings.GAZE_CIRCLE_COLOR, 3)
            cv2.circle(annotated_frame, gaze_position, 3, settings.GAZE_CIRCLE_COLOR, -1)
        
        return annotated_frame
    
    def visualize_yolo(self):
        if self.yolo_frame is None:
            # Get frame dimensions from first Tobii frame
            frame = self.tobii_capture.get_raw_frame()
            if frame is not None:
                height, width = frame.shape[:2]
            else:
                height, width = 480, 640
            # Return black frame with same dimensions
            return np.zeros((height, width, 3), dtype=np.uint8)
        
        # Get gaze position for YOLO frame
        latest_gaze = self.gaze_history[-1] if self.gaze_history else None
        
        # Check if gaze hits any object
        gazed_object = None
        if latest_gaze and self.yolo_detections:
            gazed_object = self.yolo_detector.get_object_at_gaze(self.yolo_detections, latest_gaze)

            # Track gaze persistence for TTS (only in gaze_persistence mode)
            if self.tts_enabled and settings.TTS_MODE == 'gaze_persistence':
                self.track_gaze_persistence(gazed_object)
        
        # Draw detections on YOLO frame
        annotated_frame = self.yolo_detector.draw_detections(
            self.yolo_frame, 
            self.yolo_detections, 
            latest_gaze, 
            gazed_object
        )
        
        # Add YOLO info text
        info_y = 30
        cv2.putText(annotated_frame, f"YOLO - Every {settings.YOLO_PROCESS_EVERY_N_FRAMES}th Frame", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        info_y += 25
        cv2.putText(annotated_frame, f"Objects: {len(self.yolo_detections)}", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if gazed_object:
            info_y += 25
            cv2.putText(annotated_frame, f"Looking at: {gazed_object['class_name']}", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return annotated_frame
    
    def create_dashboard(self, frame, gaze_position):
        # Create dashboard with cached screen resolution
        dashboard = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
        
        # Use pre-calculated panel dimensions
        panel_width = self.panel_width
        panel_height = self.panel_height
        
        # Top-left: Tobii gaze tracking
        tobii_frame = self.visualize_tobii(frame, gaze_position)
        if tobii_frame is not None:
            # Resize to fit panel
            tobii_resized = cv2.resize(tobii_frame, (panel_width, panel_height))
            dashboard[0:panel_height, 0:panel_width] = tobii_resized
        
        # Top-right: YOLO detection
        yolo_frame = self.visualize_yolo()
        if yolo_frame is not None:
            # Resize to fit panel
            yolo_resized = cv2.resize(yolo_frame, (panel_width, panel_height))
            dashboard[0:panel_height, panel_width:panel_width*2] = yolo_resized
        
        # Update bottom panels based on settings frequency
        self.panel_update_counter += 1
        if self.panel_update_counter % settings.PANEL_UPDATE_FREQUENCY == 0 or self.stats_panel_cache is None:
            # Bottom-left: System stats and performance metrics
            self.stats_panel_cache = self.create_stats_panel(panel_width, panel_height)
            # Bottom-right: Detection log and gaze history
            self.log_panel_cache = self.create_log_panel(panel_width, panel_height)
        
        # Use cached panels
        dashboard[panel_height:panel_height*2, 0:panel_width] = self.stats_panel_cache
        dashboard[panel_height:panel_height*2, panel_width:panel_width*2] = self.log_panel_cache
        
        # Add grid lines for visual separation
        cv2.line(dashboard, (panel_width, 0), (panel_width, self.screen_height), settings.GRID_LINE_COLOR, settings.GRID_LINE_THICKNESS)
        cv2.line(dashboard, (0, panel_height), (self.screen_width, panel_height), settings.GRID_LINE_COLOR, settings.GRID_LINE_THICKNESS)
        
        return dashboard
    
    def create_stats_panel(self, width, height):
        panel = np.zeros((height, width, 3), dtype=np.uint8)
        panel.fill(settings.STATS_PANEL_BG_COLOR)
        
        # Title
        cv2.putText(panel, "SYSTEM PERFORMANCE", (20, 40), 
                   cv2.FONT_HERSHEY_DUPLEX, settings.FONT_TITLE, settings.COLOR_PRIMARY, 2)
        
        y_offset = 80
        line_height = 35
        
        # FPS meter with bar graph
        fps = self.stats.get('fps', 0)
        cv2.putText(panel, f"FPS: {fps:.1f}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        # Draw FPS bar
        bar_width = int((fps / 60.0) * 300)  # Max 60 FPS
        cv2.rectangle(panel, (200, y_offset - 20), (200 + bar_width, y_offset), 
                     (0, 255, 0) if fps > 20 else (0, 165, 255), -1)
        
        y_offset += line_height
        
        # Frame count
        cv2.putText(panel, f"Frames Processed: {self.stats['frames_processed']}", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        
        y_offset += line_height
        
        # YOLO processing info
        yolo_updates = self.stats['frames_processed'] // 10
        cv2.putText(panel, f"YOLO Updates: {yolo_updates}", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        
        y_offset += line_height
        
        # Objects detected
        cv2.putText(panel, f"Objects Detected: {len(self.yolo_detections)}", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        
        y_offset += line_height * 2
        
        # System time
        current_time = time.strftime("%H:%M:%S")
        cv2.putText(panel, f"Time: {current_time}", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 100), 2)
        
        # Add some visual elements
        cv2.circle(panel, (width - 100, height - 100), 50, (50, 50, 50), -1)
        cv2.circle(panel, (width - 100, height - 100), 45, (0, 255, 0), 2)
        
        return panel
    
    def create_log_panel(self, width, height):
        panel = np.zeros((height, width, 3), dtype=np.uint8)
        panel.fill(settings.LOG_PANEL_BG_COLOR)
        
        # Title
        cv2.putText(panel, "DETECTION LOG & GAZE DATA", (20, 40), 
                   cv2.FONT_HERSHEY_DUPLEX, settings.FONT_TITLE, (255, 0, 255), 2)
        
        y_offset = 80
        line_height = 30
        
        # Current gaze position
        if self.gaze_history:
            latest_gaze = self.gaze_history[-1]
            cv2.putText(panel, f"Gaze Position: ({latest_gaze[0]}, {latest_gaze[1]})", 
                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(panel, "Gaze Position: No Data", 
                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
        
        y_offset += line_height
        
        # Smoothed gaze
        smoothed = self.get_smoothed_gaze()
        if smoothed:
            cv2.putText(panel, f"Smoothed Gaze: ({smoothed[0]}, {smoothed[1]})", 
                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
        
        y_offset += line_height * 2
        
        # Recent detections
        cv2.putText(panel, "Recent Detections:", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        y_offset += line_height
        
        # List detected objects
        for i, detection in enumerate(self.yolo_detections[:8]):  # Show max 8
            conf = detection['confidence']
            name = detection['class_name']
            color = (150, 150, 255) if conf > 0.7 else (150, 150, 150)
            cv2.putText(panel, f"  - {name}: {conf:.2%}", 
                       (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            y_offset += 25
            if y_offset > height - 50:
                break
        
        # Add decorative graph
        graph_x = width - 400
        graph_y = height - 200
        cv2.rectangle(panel, (graph_x, graph_y), (graph_x + 350, graph_y + 150), 
                     (50, 50, 50), 2)
        
        # Draw fake graph lines
        for i in range(5):
            y = graph_y + 150 - (i * 30)
            cv2.line(panel, (graph_x, y), (graph_x + 350, y), (30, 30, 30), 1)
        
        return panel
    
    def run(self, display=True, save_video=False, output_path='output.mp4'):
        if not self.start():
            return
        
        video_writer = None
        if save_video:
            self.logger.debug(f"Setting up video writer for {output_path}")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, 20.0, (self.screen_width, self.screen_height))
            self.logger.info(f"Recording video to {output_path} ({self.screen_width}x{self.screen_height})")
        
        self.logger.info("Starting main loop. Press 'q' to quit")
        
        prev_time = time.time()
        
        try:
            while True:
                current_time = time.time()
                time_diff = current_time - prev_time
                fps = 1 / time_diff if time_diff > 0 else 0
                prev_time = current_time
                self.stats['fps'] = fps
                
                frame, gaze_position = self.process_frame()
                
                if frame is not None:
                    # Create 1920x1080 display with 2x2 grid
                    combined_frame = self.create_dashboard(frame, gaze_position)
                    
                    if display and combined_frame is not None:
                        # Show in windowed or fullscreen mode based on settings
                        cv2.namedWindow(settings.WINDOW_TITLE, cv2.WINDOW_NORMAL)
                        if settings.FULLSCREEN_MODE:
                            cv2.setWindowProperty(settings.WINDOW_TITLE, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                        cv2.imshow(settings.WINDOW_TITLE, combined_frame)
                    
                    if save_video and video_writer and combined_frame is not None:
                        video_writer.write(combined_frame)
                
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
            self.stop()
            
            self.logger.info(f"Session stats:")
            self.logger.info(f"  Frames processed: {self.stats['frames_processed']}")
    
    def track_gaze_persistence(self, gazed_object):
        """Track how long the user has been looking at an object and trigger TTS"""
        current_time = time.time()

        if gazed_object:
            # Check if we're looking at the same object
            if self.current_gazed_object and \
               self.current_gazed_object['class_name'] == gazed_object['class_name'] and \
               self.current_gazed_object['bbox'] == gazed_object['bbox']:
                # Still looking at the same object
                time_gazed = current_time - self.gaze_start_time

                # Check if we should announce
                if time_gazed >= settings.GAZE_PERSISTENCE_THRESHOLD:
                    # Check if we need to announce or repeat
                    should_announce = False

                    if self.announced_object != gazed_object['class_name']:
                        # First time announcing this object
                        should_announce = True
                        self.announced_object = gazed_object['class_name']
                        self.last_announcement_time = current_time
                    elif self.last_announcement_time and \
                         (current_time - self.last_announcement_time) >= settings.TTS_REPEAT_INTERVAL:
                        # Time to repeat the announcement
                        should_announce = True
                        self.last_announcement_time = current_time

                    if should_announce:
                        self.announce_object(gazed_object['class_name'])
            else:
                # Looking at a different object, reset tracking
                self.current_gazed_object = gazed_object
                self.gaze_start_time = current_time
                self.announced_object = None
        else:
            # Not looking at any object
            self.current_gazed_object = None
            self.gaze_start_time = None
            self.announced_object = None
            self.last_announcement_time = None

    def announce_object(self, object_name):
        """Use TTS to announce the object name (for gaze persistence mode)"""
        self.announce_text(object_name)

    def announce_frame_objects(self):
        """Announce detected objects based on frame check frequency"""
        if not self.yolo_detections:
            return

        # Check if it's time to announce
        if (self.stats['frames_processed'] - self.last_frame_announcement) < settings.TTS_ANNOUNCE_EVERY_N_FRAMES:
            return

        self.last_frame_announcement = self.stats['frames_processed']

        # Prepare announcement based on settings
        if settings.TTS_ANNOUNCE_ALL_OBJECTS:
            # Announce all detected objects
            object_names = [det['class_name'] for det in self.yolo_detections]
            unique_objects = list(set(object_names))

            if unique_objects:
                # Count occurrences
                object_counts = {}
                for name in object_names:
                    object_counts[name] = object_counts.get(name, 0) + 1

                # Build announcement
                announcements = []
                for obj in unique_objects:
                    count = object_counts[obj]
                    if count > 1:
                        announcements.append(f"{count} {obj}s")
                    else:
                        announcements.append(obj)

                message = "Detected: " + ", ".join(announcements)
                self.announce_text(message)
        else:
            # Announce only the most confident detection
            if self.yolo_detections:
                best_detection = max(self.yolo_detections, key=lambda x: x['confidence'])
                message = f"Detected {best_detection['class_name']}"
                self.announce_text(message)

    def announce_text(self, text):
        """Generic TTS announcement function"""
        if not self.tts_enabled or self.is_speaking:
            return

        def speak():
            with self.tts_lock:
                self.is_speaking = True
                try:
                    self.logger.debug(f"Announcing: {text}")
                    self.tts_engine.say(text)
                    self.tts_engine.runAndWait()
                except Exception as e:
                    self.logger.error(f"TTS error: {e}")
                finally:
                    self.is_speaking = False

        # Run TTS in a separate thread to avoid blocking
        tts_thread = threading.Thread(target=speak)
        tts_thread.daemon = True
        tts_thread.start()

    def stop(self):
        if self.tts_enabled:
            self.tts_engine.stop()
        self.tobii_capture.disconnect()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='GazeSonar - Tobii + YOLO Dual Window')
    parser.add_argument('--tobii-address', type=str, default=None,
                       help='IP address of Tobii Glasses (auto-discover if not provided)')
    parser.add_argument('--yolo-model', type=str, default='yolo11m.pt',
                       help='YOLO model to use (default: yolo11m.pt)')
    parser.add_argument('--confidence', type=float, default=0.3,
                       help='Confidence threshold for object detection (default: 0.3)')
    parser.add_argument('--no-display', action='store_true',
                       help='Run without display window')
    parser.add_argument('--save-video', action='store_true',
                       help='Save output video')
    parser.add_argument('--output', type=str, default='gaze_sonar_output.mp4',
                       help='Output video path (default: gaze_sonar_output.mp4)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    parser.add_argument('--no-tts', action='store_true',
                       help='Disable Text-to-Speech announcements')
    parser.add_argument('--tts-mode', type=str, choices=['gaze_persistence', 'frame_check'],
                       default=None, help='TTS announcement mode')
    parser.add_argument('--tts-frequency', type=int, default=None,
                       help='Announce objects every N frames (for frame_check mode)')
    
    args = parser.parse_args()

    # Override TTS settings from command line
    if args.no_tts:
        settings.TTS_ENABLED = False
    if args.tts_mode:
        settings.TTS_MODE = args.tts_mode
    if args.tts_frequency:
        settings.TTS_ANNOUNCE_EVERY_N_FRAMES = args.tts_frequency

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