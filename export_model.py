import time
import os
import cv2
import numpy as np
from ultralytics import YOLO

def benchmark_pytorch(model, dummy_frame, iterations=100):
    print(f"[INFO] Benchmarking PyTorch inference ({iterations} iterations)...")
    # Warmup
    for _ in range(10):
        _ = model(dummy_frame, verbose=False)
        
    start_time = time.time()
    for _ in range(iterations):
        _ = model(dummy_frame, verbose=False)
    end_time = time.time()
    
    total_time = (end_time - start_time) * 1000  # ms
    avg_latency = total_time / iterations
    fps = 1000 / avg_latency
    return avg_latency, fps

def benchmark_onnx_opencv(onnx_path, dummy_frame, iterations=100):
    print(f"[INFO] Benchmarking OpenCV DNN ONNX inference ({iterations} iterations)...")
    # Load ONNX model via OpenCV DNN
    net = cv2.dnn.readNetFromONNX(onnx_path)
    
    # Warmup
    blob = cv2.dnn.blobFromImage(dummy_frame, 1/255.0, (640, 640), swapRB=True, crop=False)
    net.setInput(blob)
    _ = net.forward()
    
    start_time = time.time()
    for _ in range(iterations):
        net.setInput(blob)
        _ = net.forward()
    end_time = time.time()
    
    total_time = (end_time - start_time) * 1000  # ms
    avg_latency = total_time / iterations
    fps = 1000 / avg_latency
    return avg_latency, fps

def main():
    print("="*60)
    print(" YOLOv8 Model Exporter & Inference Benchmarker")
    print("="*60)
    
    model_pt_path = "yolov8n.pt"
    onnx_path = "yolov8n.onnx"
    
    # 1. Load PyTorch model
    print(f"[INFO] Loading PyTorch model from {model_pt_path}...")
    model = YOLO(model_pt_path)
    
    # 2. Export to ONNX if not already done
    if not os.path.exists(onnx_path):
        print(f"[INFO] Exporting PyTorch model to ONNX format...")
        # export format="onnx" creates yolov8n.onnx
        export_path = model.export(format="onnx", opset=12)
        print(f"[INFO] Model successfully exported to: {export_path}")
    else:
        print(f"[INFO] Found existing ONNX model at {onnx_path}. Skipping export.")
        
    # 3. Create dummy input frame (640x480 surveillance resolution)
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # 4. Run PyTorch Benchmark
    pt_latency, pt_fps = benchmark_pytorch(model, dummy_frame, iterations=50)
    
    # 5. Run ONNX Benchmark (OpenCV DNN engine)
    onnx_latency, onnx_fps = benchmark_onnx_opencv(onnx_path, dummy_frame, iterations=50)
    
    # 6. Display results
    print("\n" + "="*50)
    print(" BENCHMARK PERFORMANCE COMPARISON (CPU)")
    print("="*50)
    print(f" PyTorch Backend:  {pt_latency:.2f} ms/frame | {pt_fps:.1f} FPS")
    print(f" ONNX (OpenCV):    {onnx_latency:.2f} ms/frame | {onnx_fps:.1f} FPS")
    
    speedup = (pt_latency - onnx_latency) / pt_latency * 100
    if speedup > 0:
        print(f"\n[SUCCESS] ONNX inference is {speedup:.1f}% faster than PyTorch on CPU!")
    else:
        print(f"\n[INFO] PyTorch and ONNX show comparable execution times on this hardware.")
    print("="*50)

if __name__ == "__main__":
    main()
