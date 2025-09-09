import cv2
import numpy as np
from tobiiglassesctrl import TobiiGlassesController
import time
import logging

class TobiiFrameCapture:
    def __init__(self, address=None, debug=False):
        self.address = address
        self.tobii_controller = None
        self.video_cap = None
        self.is_connected = False
        
        logging.basicConfig(level=logging.DEBUG if debug else logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.debug(f"Initializing TobiiFrameCapture with address: {address}")
        
    def connect(self):
        try:
            self.logger.info(f"Connecting to Tobii Glasses at {self.address if self.address else 'auto-discover'}...")
            self.logger.debug(f"Creating TobiiGlassesController instance...")
            self.tobii_controller = TobiiGlassesController(self.address)
            self.logger.debug(f"TobiiGlassesController created successfully")
            
            if self.address is None:
                self.address = self.tobii_controller.address
                self.logger.info(f"Auto-discovered device at: {self.address}")
            
            rtsp_url = f"rtsp://{self.address}:8554/live/scene"
            self.logger.info(f"Opening video stream at: {rtsp_url}")
            self.logger.debug(f"Attempting to open VideoCapture...")
            self.video_cap = cv2.VideoCapture(rtsp_url)
            
            if not self.video_cap.isOpened():
                self.logger.error(f"VideoCapture.isOpened() returned False for {rtsp_url}")
                raise Exception("Failed to open video stream")
            
            self.logger.debug(f"Video stream opened successfully")
            self.logger.debug(f"Starting Tobii data streaming...")
            self.tobii_controller.start_streaming()
            self.is_connected = True
            self.logger.info("Successfully connected to Tobii Glasses")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect: {e}")
            self.logger.debug(f"Exception type: {type(e).__name__}")
            self.logger.debug(f"Attempting to disconnect...")
            self.disconnect()
            return False
    
    def get_frame_with_gaze(self):
        if not self.is_connected:
            self.logger.warning("Not connected to Tobii Glasses")
            return None, None
        
        self.logger.debug("Reading frame from video capture...")
        ret, frame = self.video_cap.read()
        if not ret:
            self.logger.warning("Failed to capture frame")
            self.logger.debug(f"VideoCapture.read() returned: ret={ret}")
            return None, None
        
        self.logger.debug(f"Frame captured: {frame.shape}")
        self.logger.debug("Getting gaze data...")
        gaze_data = self.tobii_controller.get_data()['gp']
        self.logger.debug(f"Gaze data timestamp: {gaze_data['ts']}")
        
        gaze_position = None
        if gaze_data['ts'] > 0:
            height, width = frame.shape[:2]
            gaze_x = int(gaze_data['gp'][0] * width)
            gaze_y = int(gaze_data['gp'][1] * height)
            gaze_position = (gaze_x, gaze_y)
            self.logger.debug(f"Gaze position: {gaze_position} (normalized: {gaze_data['gp']})")
            
            cv2.circle(frame, gaze_position, 20, (0, 255, 0), 3)
            cv2.circle(frame, gaze_position, 2, (0, 255, 0), -1)
        else:
            self.logger.debug("No valid gaze data (timestamp <= 0)")
        
        return frame, gaze_position
    
    def get_raw_frame(self):
        if not self.is_connected:
            self.logger.warning("Not connected to Tobii Glasses")
            return None
        
        self.logger.debug("Reading raw frame...")
        ret, frame = self.video_cap.read()
        if not ret:
            self.logger.warning("Failed to capture frame")
            self.logger.debug(f"VideoCapture.read() returned: ret={ret}")
            return None
        
        self.logger.debug(f"Raw frame captured: shape={frame.shape}, dtype={frame.dtype}")
        return frame
    
    def get_gaze_position(self):
        if not self.is_connected:
            self.logger.debug("get_gaze_position called but not connected")
            return None
        
        self.logger.debug("Fetching gaze data...")
        gaze_data = self.tobii_controller.get_data()['gp']
        self.logger.debug(f"Raw gaze data: {gaze_data}")
        
        if gaze_data['ts'] > 0:
            result = {
                'normalized': (gaze_data['gp'][0], gaze_data['gp'][1]),
                'timestamp': gaze_data['ts']
            }
            self.logger.debug(f"Returning gaze position: {result}")
            return result
        self.logger.debug("No valid gaze data (timestamp <= 0)")
        return None
    
    def disconnect(self):
        try:
            self.logger.debug("Starting disconnection process...")
            if self.video_cap:
                self.logger.debug("Releasing video capture...")
                self.video_cap.release()
            if self.tobii_controller and self.is_connected:
                self.logger.debug("Stopping Tobii streaming...")
                self.tobii_controller.stop_streaming()
                self.logger.debug("Closing Tobii controller...")
                self.tobii_controller.close()
            self.is_connected = False
            self.logger.info("Disconnected from Tobii Glasses")
        except Exception as e:
            self.logger.error(f"Error during disconnection: {e}")
            self.logger.debug(f"Disconnection error type: {type(e).__name__}")
    
    def __del__(self):
        self.disconnect()


if __name__ == "__main__":
    import sys
    debug = '--debug' in sys.argv
    capture = TobiiFrameCapture(debug=debug)
    
    if capture.connect():
        print("Press 'q' to quit")
        
        while True:
            frame, gaze_pos = capture.get_frame_with_gaze()
            
            if frame is not None:
                if gaze_pos:
                    cv2.putText(frame, f"Gaze: {gaze_pos}", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow("Tobii Glasses View", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()
        capture.disconnect()
    else:
        print("Failed to connect to Tobii Glasses")