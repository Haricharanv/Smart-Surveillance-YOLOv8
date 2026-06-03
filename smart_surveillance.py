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
THERMAL_COLORMAP = cv2.COLORMAP_INFERNO  # Pseudo-thermal colormap


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


def enhance_low_light(frame):
    """
    Simulate electro-optic low-light enhancement:
    1. Convert to grayscale
    2. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    3. Apply a pseudo-thermal/infrared colormap
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_GRID_SIZE)
    enhanced = clahe.apply(gray)
    thermal = cv2.applyColorMap(enhanced, THERMAL_COLORMAP)
    return thermal


def draw_restricted_zone(frame, zone):
    """Draw the restricted zone polygon with a semi-transparent overlay."""
    overlay = frame.copy()
    cv2.fillPoly(overlay, [zone], (0, 0, 120))
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
    cv2.polylines(frame, [zone], isClosed=True, color=(0, 0, 255), thickness=2)
    
    # Label
    top_y = zone[:, 1].min()
    cx = int(zone[:, 0].mean())
    cv2.putText(frame, "RESTRICTED ZONE", (cx - 80, top_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)


def check_intrusion(feet_point, zone):
    """Check if a point (person's feet) is inside the restricted zone polygon."""
    return cv2.pointPolygonTest(zone, feet_point, False) >= 0


def draw_fps(frame, fps):
    """Draw FPS counter on the frame."""
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


def draw_detections(frame, boxes, track_ids, confidences, zone):
    """
    Draw bounding boxes, track IDs, confidence scores,
    and check for intrusion alerts.
    Returns True if any intrusion is detected.
    """
    intrusion = False

    for box, tid, conf in zip(boxes, track_ids, confidences):
        x1, y1, x2, y2 = box

        # Bottom-center = approximate foot position
        feet = (int((x1 + x2) / 2), int(y2))
        is_intruder = check_intrusion(feet, zone)

        if is_intruder:
            intrusion = True
            color = (0, 0, 255)      # Red
            label_bg = (0, 0, 180)
        else:
            color = (0, 255, 0)      # Green
            label_bg = (0, 160, 0)

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Foot marker
        cv2.circle(frame, feet, 5, color, -1)

        # Label background + text
        label = f"ID:{tid} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), label_bg, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return intrusion


def draw_alert(frame):
    """Flash an intrusion alert banner at the top of the frame."""
    h, w = frame.shape[:2]
    # Red banner
    cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 200), -1)
    cv2.putText(frame, "!! INTRUSION DETECTED !!", (w // 2 - 180, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)


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

        # ── Low-light enhancement ──
        thermal_frame = enhance_low_light(frame)

        # ── YOLOv8 detection + tracking ──
        results = model.track(
            thermal_frame,
            persist=True,
            classes=TARGET_CLASSES,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False
        )

        # ── Draw restricted zone ──
        draw_restricted_zone(thermal_frame, zone)

        # ── Process detections ──
        intrusion = False
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()
            intrusion = draw_detections(thermal_frame, boxes, track_ids, confidences, zone)

        # ── Alert banner ──
        if intrusion:
            draw_alert(thermal_frame)

        # ── FPS calculation ──
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-9)
        prev_time = curr_time
        draw_fps(thermal_frame, fps)

        # ── Side-by-side display ──
        display_h = 480
        scale = display_h / frame_h
        display_w = int(frame_w * scale)

        left = cv2.resize(frame, (display_w, display_h))
        right = cv2.resize(thermal_frame, (display_w, display_h))

        # Labels
        cv2.putText(left, "RAW FEED", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(right, "ELECTRO-OPTIC AI FEED", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        combined = np.hstack([left, right])

        cv2.imshow("Low-Light Electro-Optic Object Tracking Pipeline", combined)

        # Wait key - 30ms for video to keep real-time playback speed, 1ms for camera
        delay = 1 if is_live_webcam else 30
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Pipeline stopped.")


if __name__ == "__main__":
    main()
