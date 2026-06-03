import cv2
import numpy as np
import os
import sys
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
    print("[INFO] Processing 15 frames...")
    
    for i in range(15):
        ret, frame = cap.read()
        if not ret:
            print("[WARN] End of video or error reading frame.")
            break
            
        # 1. Low-light enhancement
        thermal_frame = smart_surveillance.enhance_low_light(frame)
        
        # 2. YOLO tracking
        results = model.track(
            thermal_frame,
            persist=True,
            classes=[0],  # Person
            conf=0.45,
            verbose=False
        )
        
        # 3. Draw restricted zone
        smart_surveillance.draw_restricted_zone(thermal_frame, zone)
        
        # 4. Draw detections & intrusion check
        intrusion = False
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()
            intrusion = smart_surveillance.draw_detections(thermal_frame, boxes, track_ids, confidences, zone)
            
        # 5. Alert
        if intrusion:
            smart_surveillance.draw_alert(thermal_frame)
            
        # 6. Save a middle frame with detections as our verification image
        if i == 12 or (intrusion and i > 5):
            display_h = 480
            scale = display_h / frame_h
            display_w = int(frame_w * scale)
            
            left = cv2.resize(frame, (display_w, display_h))
            right = cv2.resize(thermal_frame, (display_w, display_h))
            
            cv2.putText(left, "RAW FEED", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(right, "ELECTRO-OPTIC AI FEED", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            combined = np.hstack([left, right])
            output_img_path = os.path.join(script_dir, "verification_output.jpg")
            cv2.imwrite(output_img_path, combined)
            print(f"[INFO] Verification frame saved to: {output_img_path}")
            
    cap.release()
    print("[INFO] Verification complete.")

if __name__ == "__main__":
    main()
