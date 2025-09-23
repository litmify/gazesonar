import cv2
import numpy as np
import socket
import json
import threading
import time
import logging


class TobiiFrameCapture:
    def __init__(self, address=None, debug=False):
        self.address = address
        self.video_cap = None
        self.is_connected = False
        self.data_socket = None
        self.video_socket = None
        self.running = False
        self.latest_gaze_data = None
        self.gaze_lock = threading.Lock()

        # Constants for Tobii communication
        self.PORT = 49152
        self.KA_DATA_MSG = '{"type": "live.data.unicast", "key": "some_GUID", "op": "start"}'
        self.KA_VIDEO_MSG = '{"type": "live.video.unicast", "key": "some_other_GUID", "op": "start"}'
        self.timeout = 1.0

        logging.basicConfig(level=logging.DEBUG if debug else logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.debug(f"Initializing TobiiFrameCapture with address: {address}")

    def _discover_glasses(self):
        """Discover Tobii glasses on the network using multicast"""
        try:
            MULTICAST_ADDR = 'ff02::1'  # ipv6: all nodes on the local network segment
            s6 = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)
            s6.settimeout(5.0)  # 5 second timeout for discovery
            s6.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s6.bind(('::', 13007))

            s6.sendto(b'{"type":"discover"}', (MULTICAST_ADDR, 13006))

            data, address = s6.recvfrom(1024)
            discovered_address = address[0].split('%')[0]  # Remove scope id if present

            # Try to extract IPv4 address from response if available
            try:
                response = json.loads(data)
                if 'ipv4' in response:
                    discovered_address = response['ipv4']
            except:
                pass

            s6.close()
            return discovered_address
        except Exception as e:
            self.logger.error(f"Failed to discover glasses: {e}")
            return None

    def _create_socket(self, peer):
        """Create UDP socket for communication"""
        iptype = socket.AF_INET
        if ':' in peer[0]:
            iptype = socket.AF_INET6
        return socket.socket(iptype, socket.SOCK_DGRAM)

    def _send_keepalive(self, sock, msg, peer):
        """Send keepalive messages to maintain connection"""
        while self.running:
            try:
                sock.sendto(msg.encode('utf-8'), peer)
                time.sleep(self.timeout)
            except Exception as e:
                self.logger.error(f"Keepalive error: {e}")
                break

    def _receive_data(self):
        """Receive gaze data in background thread"""
        while self.running:
            try:
                data, _ = self.data_socket.recvfrom(1024)
                json_data = json.loads(data)

                # Extract gaze data
                if 'gp' in json_data:
                    with self.gaze_lock:
                        self.latest_gaze_data = json_data['gp']
            except socket.timeout:
                continue
            except Exception as e:
                self.logger.error(f"Data receive error: {e}")
                if not self.running:
                    break

    def connect(self):
        try:
            # Auto-discover if no address provided
            if self.address is None:
                self.logger.info("Auto-discovering Tobii Glasses on network...")
                self.address = self._discover_glasses()
                if self.address is None:
                    self.logger.error("No Tobii Glasses found on network")
                    return False
                self.logger.info(f"Discovered device at: {self.address}")

            self.logger.info(f"Connecting to Tobii Glasses at {self.address}...")

            # Setup peer address
            peer = (self.address, self.PORT)

            # Create data socket for gaze data
            self.data_socket = self._create_socket(peer)
            self.data_socket.settimeout(0.1)  # Non-blocking receive

            # Create video socket (although we'll use RTSP for actual video)
            self.video_socket = self._create_socket(peer)

            # Start keepalive threads
            self.running = True

            # Start data keepalive
            data_thread = threading.Thread(target=self._send_keepalive,
                                         args=(self.data_socket, self.KA_DATA_MSG, peer))
            data_thread.daemon = True
            data_thread.start()

            # Start video keepalive
            video_thread = threading.Thread(target=self._send_keepalive,
                                          args=(self.video_socket, self.KA_VIDEO_MSG, peer))
            video_thread.daemon = True
            video_thread.start()

            # Start data receive thread
            receive_thread = threading.Thread(target=self._receive_data)
            receive_thread.daemon = True
            receive_thread.start()

            # Open RTSP video stream
            rtsp_url = f"rtsp://{self.address}:8554/live/scene"
            self.logger.info(f"Opening video stream at: {rtsp_url}")
            self.video_cap = cv2.VideoCapture(rtsp_url)

            # Reduce video buffering for lower latency
            self.video_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.video_cap.set(cv2.CAP_PROP_FPS, 30)

            if not self.video_cap.isOpened():
                self.logger.error(f"Failed to open video stream")
                self.disconnect()
                return False

            self.is_connected = True
            self.logger.info("Successfully connected to Tobii Glasses")
            return True

        except Exception as e:
            self.logger.error(f"Failed to connect: {e}")
            self.disconnect()
            return False

    def get_frame_with_gaze(self):
        if not self.is_connected:
            self.logger.warning("Not connected to Tobii Glasses")
            return None, None

        ret, frame = self.video_cap.read()
        if not ret:
            self.logger.warning("Failed to capture frame")
            return None, None

        gaze_position = None
        with self.gaze_lock:
            if self.latest_gaze_data and 'ts' in self.latest_gaze_data and self.latest_gaze_data['ts'] > 0:
                height, width = frame.shape[:2]
                gaze_x = int(self.latest_gaze_data['gp'][0] * width)
                gaze_y = int(self.latest_gaze_data['gp'][1] * height)
                gaze_position = (gaze_x, gaze_y)

                cv2.circle(frame, gaze_position, 20, (0, 255, 0), 3)
                cv2.circle(frame, gaze_position, 2, (0, 255, 0), -1)

        return frame, gaze_position

    def get_raw_frame(self):
        if not self.is_connected:
            return None

        # Flush buffer to get latest frame
        self.video_cap.grab()
        ret, frame = self.video_cap.retrieve()
        if not ret:
            return None

        return frame

    def get_gaze_position(self):
        if not self.is_connected:
            return None

        with self.gaze_lock:
            if self.latest_gaze_data and 'ts' in self.latest_gaze_data and self.latest_gaze_data['ts'] > 0:
                return {
                    'normalized': (self.latest_gaze_data['gp'][0], self.latest_gaze_data['gp'][1]),
                    'timestamp': self.latest_gaze_data['ts']
                }
        return None

    def disconnect(self):
        try:
            self.logger.debug("Starting disconnection process...")
            self.running = False

            if self.video_cap:
                self.logger.debug("Releasing video capture...")
                self.video_cap.release()

            if self.data_socket:
                self.logger.debug("Closing data socket...")
                self.data_socket.close()

            if self.video_socket:
                self.logger.debug("Closing video socket...")
                self.video_socket.close()

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