import cv2
import numpy as np
import os
import sys
import time
from ultralytics import YOLO

# Add parent directory to path so we can import smart_surveillance
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import smart_surveillance

def main():
    print("[INFO] Starting verification script...")
    
    # Download sample video if not exists
    script_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(script_dir, "people_walking.mp4")
    
    if not os.path.exists(video_path):
        print("[INFO] Video not found. Downloading...")
        success = smart_surveillance.download_sample_video(video_path)
        if not success:
            print("[ERROR] Failed to download sample video.")
            sys.exit(1)
            
    # Load YOLOv8 model
    print("[INFO] Loading YOLOv8 model...")
    model = YOLO("yolov8n.pt")
    
    # Open video
    print(f"[INFO] Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[ERROR] Cannot open video.")
        sys.exit(1)
        
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    zone = smart_surveillance.build_restricted_zone(frame_w, frame_h)
    
    print(f"[INFO] Resolution: {frame_w}x{frame_h}")
    print("[INFO] Processing frames to find a detection...")
    
    prev_time = time.time()
    
    for i in range(120):
        ret, frame = cap.read()
        if not ret:
            print("[WARN] End of video or error reading frame.")
            break
            
        # FPS calculation
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-9)
        prev_time = curr_time
            
        # 1. Low-light enhancement (with defaults)
        thermal_frame = smart_surveillance.enhance_low_light(frame)
        
        # 2. YOLO tracking
        results = model.track(
            thermal_frame,
            persist=True,
            classes=[0],  # Person
            conf=0.45,
            verbose=False
        )
        
        # Check if anyone is tracked
        has_detection = results[0].boxes.id is not None
        
        # 3. Draw restricted zone & evaluate intrusion
        intrusion = False
        if has_detection:
            boxes_eval = results[0].boxes.xyxy.cpu().numpy().astype(int)
            for box in boxes_eval:
                feet_eval = (int((box[0] + box[2]) / 2), int(box[3]))
                if smart_surveillance.check_intrusion(feet_eval, zone):
                    intrusion = True
                    break
        
        smart_surveillance.draw_restricted_zone(thermal_frame, zone, intrusion)
        
        # 4. Draw detections
        if has_detection:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()
            smart_surveillance.draw_detections(thermal_frame, boxes, track_ids, confidences, zone)
            
        # 5. Draw alert border
        if intrusion:
            smart_surveillance.draw_alert(thermal_frame)
            
        # 6. Draw HUD overlay
        smart_surveillance.draw_hud(thermal_frame, fps, "FUSED INFERNO", enable_edge_fusion=True, intrusion_detected=intrusion)
            
        # 7. Save frame when a person is detected
        if has_detection and i > 25:
            display_h = 480
            scale = display_h / frame_h
            display_w = int(frame_w * scale)
            
            left = cv2.resize(frame, (display_w, display_h))
            right = cv2.resize(thermal_frame, (display_w, display_h))
            
            # Simple label overlay on left feed
            cv2.rectangle(left, (0, 0), (110, 30), (0, 0, 0), -1)
            cv2.putText(left, "RAW FEED", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            
            combined = np.hstack([left, right])
            output_img_path = os.path.join(script_dir, "verification_output.jpg")
            cv2.imwrite(output_img_path, combined)
            print(f"[INFO] Detection found on frame {i}! Verification frame saved to: {output_img_path}")
            break
            
    cap.release()
    print("[INFO] Verification complete.")

if __name__ == "__main__":
    main()
