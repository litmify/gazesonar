# GazeSonar Configuration Settings

# Tobii Settings
TOBII_ADDRESS = None  # None for auto-discovery, or specify IP like '192.168.1.100'
GAZE_HISTORY_SIZE = 5  # Number of gaze positions to keep for smoothing
GAZE_TRAIL_COLOR = (0, 255, 255)  # Yellow trail
GAZE_CIRCLE_COLOR = (0, 255, 0)  # Green circle
GAZE_CIRCLE_RADIUS = 25

# YOLO Settings
YOLO_MODEL = "yolo11m.pt"  # Options: yolo11n.pt (fast), yolo11m.pt (balanced), yolo11l.pt (accurate)
CONFIDENCE_THRESHOLD = 0.3  # Detection confidence threshold (0.0-1.0)
YOLO_PROCESS_EVERY_N_FRAMES = 5  # Process YOLO every Nth frame for performance
MAX_DISTANCE_TO_GAZE = 150  # Maximum pixels from gaze to consider object "looked at"

# Display Settings
FULLSCREEN_MODE = True  # Use fullscreen windowed mode
WINDOW_TITLE = "GazeSonar Dashboard"
SHOW_FPS = True
DEBUG_OUTPUT_EVERY_N_FRAMES = 10  # Console debug output frequency

# Dashboard Panel Settings
PANEL_UPDATE_FREQUENCY = 10  # Update bottom panels every N frames
STATS_PANEL_BG_COLOR = 20  # Dark gray (0-255)
LOG_PANEL_BG_COLOR = 10  # Very dark gray
GRID_LINE_COLOR = (100, 100, 100)  # Gray grid lines
GRID_LINE_THICKNESS = 2

# Video Recording Settings
DEFAULT_VIDEO_OUTPUT = "gaze_sonar_output.mp4"
VIDEO_FPS = 20.0
VIDEO_CODEC = "mp4v"

# Performance Settings
RTSP_BUFFER_SIZE = 1  # Minimize RTSP stream buffering for low latency
RTSP_FPS = 30

# UI Colors (BGR format for OpenCV)
COLOR_PRIMARY = (0, 255, 255)  # Cyan
COLOR_SUCCESS = (0, 255, 0)  # Green
COLOR_WARNING = (0, 165, 255)  # Orange
COLOR_ERROR = (0, 0, 255)  # Red
COLOR_TEXT = (255, 255, 255)  # White
COLOR_TEXT_DIM = (200, 200, 200)  # Light gray
COLOR_TEXT_DARK = (100, 100, 100)  # Dark gray

# Font Settings
FONT_TITLE = 1.0  # Font scale for titles
FONT_NORMAL = 0.8  # Font scale for normal text
FONT_SMALL = 0.6  # Font scale for small text

# Debug Settings
DEBUG_MODE = False  # Enable debug logging
CONSOLE_OUTPUT = False  # Enable console output for detections
