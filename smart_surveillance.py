"""
Low-Light Electro-Optic Object Tracking Pipeline
==================================================
Real-time surveillance system with low-light image enhancement,
YOLOv8 object detection, ByteTrack persistent tracking, and
polygon-based restricted zone intrusion detection.

Author: Hari Charan V
Tech: Python, OpenCV, YOLOv8 (Ultralytics), NumPy
"""

import cv2
import numpy as np
import time
import os
import argparse
import urllib.request
from ultralytics import YOLO


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
MODEL_PATH = "yolov8n.pt"           # YOLOv8 nano (auto-downloads on first run)
CONFIDENCE_THRESHOLD = 0.45         # Minimum detection confidence
TARGET_CLASSES = [0]                # COCO class 0 = 'person'
CLAHE_CLIP_LIMIT = 3.0             # CLAHE contrast enhancement strength
CLAHE_GRID_SIZE = (8, 8)           # CLAHE tile grid size

# Curated sensor profiles for electro-optic feeds
COLORMAPS = [
    ("FUSED INFERNO (THERMAL)", cv2.COLORMAP_INFERNO),
    ("MONOCHROME BONE (NIGHT VISION)", cv2.COLORMAP_BONE),
    ("THERMAL HOT (INFRARED)", cv2.COLORMAP_HOT),
    ("SPECTRAL JET (RAINBOW)", cv2.COLORMAP_JET),
]


def build_restricted_zone(frame_w, frame_h):
    """
    Define a polygon-based restricted zone in the center-bottom
    of the frame. Coordinates scale with frame resolution.
    """
    cx, cy = frame_w // 2, frame_h // 2
    zone = np.array([
        [cx - 150, frame_h - 20],   # bottom-left
        [cx + 150, frame_h - 20],   # bottom-right
        [cx + 100, cy + 40],        # top-right
        [cx - 100, cy + 40],        # top-left
    ], np.int32)
    return zone


def enhance_low_light(frame, colormap_idx=0, enable_edge_fusion=True):
    """
    Simulate electro-optic low-light enhancement:
    1. Convert to grayscale
    2. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    3. Apply selected colormap
    4. Fuse high-frequency edge outlines for crisp object definition
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_GRID_SIZE)
    enhanced = clahe.apply(gray)
    
    # Resolve current sensor profile
    _, colormap_code = COLORMAPS[colormap_idx]
    thermal = cv2.applyColorMap(enhanced, colormap_code)
    
    if enable_edge_fusion:
        # Detect structures using Canny Edge filter
        edges = cv2.Canny(enhanced, 40, 120)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        # Color edges in Neon Cyan (B=255, G=255, R=0) to contrast with colormap
        edges_colored = np.where(edges_colored > 0, [255, 255, 0], 0).astype(np.uint8)
        # Blend the edges overlay onto the thermal colormap
        fusion = cv2.addWeighted(thermal, 0.82, edges_colored, 0.18, 0)
        return fusion
    
    return thermal


def draw_restricted_zone(frame, zone, intrusion=False):
    """Draw the restricted zone polygon with a semi-transparent overlay."""
    overlay = frame.copy()
    color = (0, 0, 255) if intrusion else (0, 180, 0) # Red if alarm, Green if safe
    
    cv2.fillPoly(overlay, [zone], color)
    cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)
    cv2.polylines(frame, [zone], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
    
    # Boundary label
    top_y = zone[:, 1].min()
    cx = int(zone[:, 0].mean())
    cv2.putText(frame, "RESTRICTED BOUNDARY (ACTIVE)", (cx - 100, top_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def check_intrusion(feet_point, zone):
    """Check if a point (person's feet) is inside the restricted zone polygon."""
    return cv2.pointPolygonTest(zone, feet_point, False) >= 0


def draw_corner_brackets(frame, pt1, pt2, color, thickness=2, r=15):
    """Draw high-tech corner brackets (reticles) around detected targets."""
    x1, y1 = pt1
    x2, y2 = pt2
    # Faint full bounding box for tracking context
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1, lineType=cv2.LINE_AA)
    
    # Top-left corner
    cv2.line(frame, (x1, y1), (x1 + r, y1), color, thickness)
    cv2.line(frame, (x1, y1), (x1, y1 + r), color, thickness)
    # Top-right corner
    cv2.line(frame, (x2, y1), (x2 - r, y1), color, thickness)
    cv2.line(frame, (x2, y1), (x2, y1 + r), color, thickness)
    # Bottom-left corner
    cv2.line(frame, (x1, y2), (x1 + r, y2), color, thickness)
    cv2.line(frame, (x1, y2), (x1, y2 - r), color, thickness)
    # Bottom-right corner
    cv2.line(frame, (x2, y2), (x2 - r, y2), color, thickness)
    cv2.line(frame, (x2, y2), (x2, y2 - r), color, thickness)


def draw_detections(frame, boxes, track_ids, confidences, zone):
    """
    Draw target reticles, persistent tracking IDs, confidence scores,
    and feet positions. Returns True if any target is inside the zone.
    """
    intrusion = False
    
    # Premium neon colors (Cyan for secure, Red/Orange for alerts)
    color_secure = (255, 220, 0)  # Neon Cyan
    color_alert = (0, 0, 255)    # Neon Red

    for box, tid, conf in zip(boxes, track_ids, confidences):
        x1, y1, x2, y2 = box

        # Bottom-center = approximate foot position
        feet = (int((x1 + x2) / 2), int(y2))
        is_intruder = check_intrusion(feet, zone)

        if is_intruder:
            intrusion = True
            color = color_alert
            label_bg = (0, 0, 180)
        else:
            color = color_secure
            label_bg = (180, 120, 0)

        # Draw tech corner reticle
        draw_corner_brackets(frame, (x1, y1), (x2, y2), color, thickness=2, r=15)

        # Foot target indicator
        cv2.circle(frame, feet, 4, color, -1)

        # Semitransparent background banner for text overlay
        label = f"TRK:{tid} [{conf:.0%}]"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1 - th - 8), (x1 + tw + 6, y1), label_bg, -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        cv2.putText(frame, label, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    return intrusion


def draw_hud(frame, fps, sensor_name, enable_edge_fusion, intrusion_detected):
    """Draw a premium heads-up display dashboard overlay on top and bottom."""
    h, w = frame.shape[:2]
    
    # Top HUD background
    hud_h = 35
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, hud_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Text overlays
    text_color = (220, 220, 220)
    
    # System Status
    cv2.putText(frame, "SYS: ACTIVE", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
    
    # Sensor State
    sensor_mode_str = f"FEED: {sensor_name}" + (" + EDGE_FUSION" if enable_edge_fusion else "")
    cv2.putText(frame, sensor_mode_str, (110, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1, cv2.LINE_AA)
    
    # Intrusion State Alarm
    if intrusion_detected:
        # Pulsing Red alarm text
        cv2.putText(frame, "[ ALARM: INTRUSION DETECTED ]", (w - 320, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "ZONE STATUS: SECURE", (w - 290, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
        
    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 80, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # Bottom controls guide bar
    cv2.rectangle(overlay, (0, h - 25), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    shortcuts_str = "[C] Cycle Colormap  |  [F] Toggle Edge Fusion  |  [Z] Toggle Zone  |  [Q] Quit"
    cv2.putText(frame, shortcuts_str, (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1, cv2.LINE_AA)


def draw_alert(frame):
    """Draw a flashing red border outline on target alarm states."""
    h, w = frame.shape[:2]
    # Warning outline border
    cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 220), 4)


def download_sample_video(dest_path):
    url = "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/people-detection.mp4"
    print(f"[INFO] Downloading sample surveillance video (approx. 1.2 MB)...")
    print(f"       From: {url}")
    try:
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req) as response, open(dest_path, 'wb') as out_file:
            out_file.write(response.read())
        print(f"[INFO] Download complete! Saved to {dest_path}")
        return True
    except Exception as e:
        print(f"[WARN] Primary download link failed: {e}")
        fallback_url = "https://github.com/intel-iot-devkit/sample-videos/raw/master/people-detection.mp4"
        print(f"[INFO] Trying fallback URL: {fallback_url}")
        try:
            req = urllib.request.Request(
                fallback_url, 
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            )
            with urllib.request.urlopen(req) as response, open(dest_path, 'wb') as out_file:
                out_file.write(response.read())
            print(f"[INFO] Download complete! Saved to {dest_path}")
            return True
        except Exception as e2:
            print(f"[ERROR] Fallback download also failed: {e2}")
            return False


def main():
    parser = argparse.ArgumentParser(description="Low-Light surveillance tracking pipeline")
    parser.add_argument("--source", type=str, default="0", help="Webcam index (e.g. 0, 1) or path to video file")
    parser.add_argument("--demo", action="store_true", help="Force demo mode using a sample video")
    args = parser.parse_args()

    # Load YOLOv8 model
    print("[INFO] Loading YOLOv8 model...")
    model = YOLO(MODEL_PATH)

    # Determine input source
    source = args.source
    if args.demo:
        source = "demo"

    # Handle demo mode or automatic fallback
    script_dir = os.path.dirname(os.path.abspath(__file__))
    demo_video_path = os.path.join(script_dir, "people_walking.mp4")

    cap = None
    is_live_webcam = False

    if source == "demo":
        print("[INFO] Demo mode selected.")
        if not os.path.exists(demo_video_path):
            success = download_sample_video(demo_video_path)
            if not success:
                print("[ERROR] Could not obtain sample video for demo. Exiting.")
                return
        source = demo_video_path

    # Try opening the webcam or video
    # If source is digit, parse it as webcam index
    if isinstance(source, str) and source.isdigit():
        webcam_idx = int(source)
        is_live_webcam = True
        
        # We try opening without CAP_DSHOW first (which uses MSMF and doesn't hang when blocked)
        print(f"[INFO] Attempting to open webcam index {webcam_idx}...")
        cap = cv2.VideoCapture(webcam_idx)
        
        # Test if we can grab a frame. If the camera is disabled or in use, cap.read() will return False
        if cap.isOpened():
            ret, test_frame = cap.read()
            if not ret:
                print("[WARN] Webcam opened but failed to grab a frame (may be in use or blocked by privacy settings).")
                cap.release()
                cap = None
        else:
            print(f"[WARN] Webcam index {webcam_idx} could not be opened.")
            cap = None
            
        # If standard open failed, try CAP_DSHOW as a secondary attempt
        if cap is None:
            print("[INFO] Retrying camera with DirectShow backend (CAP_DSHOW)...")
            cap = cv2.VideoCapture(webcam_idx, cv2.CAP_DSHOW)
            if cap.isOpened():
                ret, test_frame = cap.read()
                if not ret:
                    print("[WARN] DirectShow camera failed to grab a frame.")
                    cap.release()
                    cap = None
            else:
                cap = None

        # If camera still fails, trigger automatic demo fallback
        if cap is None:
            print("[WARN] Live camera feed is unavailable.")
            print("[INFO] Automatically falling back to Demo Mode with a sample video...")
            if not os.path.exists(demo_video_path):
                success = download_sample_video(demo_video_path)
                if not success:
                    print("[ERROR] Could not obtain sample video for fallback. Exiting.")
                    return
            source = demo_video_path
            is_live_webcam = False
            cap = cv2.VideoCapture(source)
    else:
        # Source is a video file path
        print(f"[INFO] Opening video file: {source}...")
        cap = cv2.VideoCapture(source)

    if cap is None or not cap.isOpened():
        print("[ERROR] Failed to open input source. Exiting.")
        return

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Handle potentially zero dimensions from empty video/camera initialization issues
    if frame_w == 0 or frame_h == 0:
        frame_w, frame_h = 640, 480
        
    zone = build_restricted_zone(frame_w, frame_h)

    print(f"[INFO] Resolution: {frame_w}x{frame_h}")
    print("[INFO] Press 'q' to quit.")

    # Control states for interactive enhancements
    colormap_idx = 0
    enable_edge_fusion = True
    show_zone = True
    prev_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            # If it's a video file, rewind to loop continuously
            if not is_live_webcam:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                break

        # ── Low-light enhancement & Fusion ──
        sensor_name, _ = COLORMAPS[colormap_idx]
        thermal_frame = enhance_low_light(frame, colormap_idx, enable_edge_fusion)

        # ── YOLOv8 detection + tracking ──
        results = model.track(
            thermal_frame,
            persist=True,
            classes=TARGET_CLASSES,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False
        )

        # ── Draw restricted zone ──
        if show_zone:
            # Pre-evaluate intrusion state for correct zone line coloring
            intrusion_eval = False
            if results[0].boxes.id is not None:
                boxes_eval = results[0].boxes.xyxy.cpu().numpy().astype(int)
                for box in boxes_eval:
                    feet_eval = (int((box[0] + box[2]) / 2), int(box[3]))
                    if check_intrusion(feet_eval, zone):
                        intrusion_eval = True
                        break
            draw_restricted_zone(thermal_frame, zone, intrusion_eval)

        # ── Process detections ──
        intrusion = False
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()
            intrusion = draw_detections(thermal_frame, boxes, track_ids, confidences, zone)

        # ── Alert border ──
        if intrusion:
            draw_alert(thermal_frame)

        # ── FPS calculation ──
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-9)
        prev_time = curr_time
        
        # ── HUD Dashboard Bar ──
        draw_hud(thermal_frame, fps, sensor_name, enable_edge_fusion, intrusion)

        # ── Side-by-side display ──
        display_h = 480
        scale = display_h / frame_h
        display_w = int(frame_w * scale)

        left = cv2.resize(frame, (display_w, display_h))
        right = cv2.resize(thermal_frame, (display_w, display_h))

        # Raw feed screen overlay label
        cv2.rectangle(left, (0, 0), (110, 30), (0, 0, 0), -1)
        cv2.putText(left, "RAW FEED", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        combined = np.hstack([left, right])

        cv2.imshow("Low-Light Electro-Optic Object Tracking Pipeline", combined)

        # Wait key - handles key actions
        delay = 1 if is_live_webcam else 30
        key = cv2.waitKey(delay) & 0xFF
        
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('c') or key == ord('C'):
            colormap_idx = (colormap_idx + 1) % len(COLORMAPS)
            print(f"[INFO] Switched sensor profile to: {COLORMAPS[colormap_idx][0]}")
        elif key == ord('f') or key == ord('F'):
            enable_edge_fusion = not enable_edge_fusion
            print(f"[INFO] Edge Fusion toggled: {enable_edge_fusion}")
        elif key == ord('z') or key == ord('Z'):
            show_zone = not show_zone
            print(f"[INFO] Restricted boundary visibility toggled: {show_zone}")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Pipeline stopped.")


if __name__ == "__main__":
    main()
