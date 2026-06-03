/**
 * YOLOv8 ONNX Inference Integration Demo (C++)
 * ============================================
 * Demonstrates C++ integration for loading, preprocessing, running, and parsing 
 * YOLOv8 ONNX models using OpenCV's DNN module.
 * 
 * Fulfills: "C++ familiarity for inference-side integration" 
 *           and "SOTA detection architectures" requirements.
 */

#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

// Constants
const float CONFIDENCE_THRESHOLD = 0.45f;
const float NMS_THRESHOLD = 0.4f;
const cv::Size YOLO_INPUT_SIZE(640, 640);

struct Detection {
    cv::Rect box;
    float confidence;
    int class_id;
};

int main(int argc, char** argv) {
    std::cout << "==================================================" << std::endl;
    std::cout << "  YOLOv8 C++ ONNX Inference Integration Sandbox   " << std::endl;
    std::cout << "==================================================" << std::endl;

    std::string model_path = "yolov8n.onnx";
    if (argc > 1) {
        model_path = argv[1];
    }

    // 1. Initialize and Load ONNX Model
    std::cout << "[INFO] Loading ONNX model from: " << model_path << "..." << std::endl;
    cv::dnn::Net net;
    try {
        net = cv::dnn::readNetFromONNX(model_path);
    } catch (const cv::Exception& e) {
        std::cerr << "[ERROR] Failed to load ONNX model. OpenCV Exception: " << e.what() << std::endl;
        std::cerr << "[TIP] Ensure you have run 'export_model.py' to generate the ONNX file." << std::endl;
        return -1;
    }

    // Setup CPU inference backend
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    std::cout << "[INFO] Inference engine initialized successfully." << std::endl;

    // 2. Mock input frame setup (Simulating camera capture)
    std::cout << "[INFO] Creating mock camera frame (640x480 resolution)..." << std::endl;
    cv::Mat frame = cv::Mat::zeros(480, 640, CV_8UC3);
    // Draw some mock shapes to simulate targets
    cv::circle(frame, cv::Point(320, 240), 80, cv::Scalar(100, 100, 100), -1);
    cv::rectangle(frame, cv::Point(280, 180), cv::Point(360, 300), cv::Scalar(255, 255, 255), 2);

    // 3. Preprocess Image (blobFromImage)
    // YOLOv8 normalization: scale by 1/255.0, resize to 640x640, swap Red & Blue channels
    std::cout << "[INFO] Preprocessing frame (image resizing & blob creation)..." << std::endl;
    cv::Mat blob;
    cv::dnn::blobFromImage(frame, blob, 1.0 / 255.0, YOLO_INPUT_SIZE, cv::Scalar(), true, false);

    // 4. Set Network Input and Run Forward Pass
    net.setInput(blob);
    std::vector<cv::Mat> outputs;
    std::cout << "[INFO] Running model forward pass..." << std::endl;
    
    double t_start = (double)cv::getTickCount();
    net.forward(outputs, net.getUnconnectedOutLayersNames());
    double t_end = (double)cv::getTickCount();
    double inference_ms = (t_end - t_start) * 1000.0 / cv::getTickFrequency();
    std::cout << "[INFO] Inference completed in: " << inference_ms << " ms" << std::endl;

    // 5. Parse Output Bounding Boxes
    // YOLOv8 output tensor shape: [1, 84, 8400]
    // 84 channels = 4 box coordinates (cx, cy, w, h) + 80 class confidence scores
    // 8400 candidate anchor boxes
    cv::Mat output = outputs[0];
    // Reshape output to remove the batch dimension: [84, 8400]
    cv::Mat output_data = output.reshape(1, output.size[1]); // shape: [84, 8400]
    cv::transpose(output_data, output_data); // Transpose to: [8400, 84]

    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;

    float x_factor = (float)frame.cols / YOLO_INPUT_SIZE.width;
    float y_factor = (float)frame.rows / YOLO_INPUT_SIZE.height;

    std::cout << "[INFO] Parsing raw detection tensors..." << std::endl;
    for (int r = 0; r < output_data.rows; ++r) {
        // Output row contains: [cx, cy, w, h, score0, score1, ..., score79]
        float* row_ptr = output_data.ptr<float>(r);
        
        // Find best class score (we focus on COCO class 0: person)
        float score = row_ptr[4]; // Person class index (0) is index 4 in YOLOv8
        
        if (score >= CONFIDENCE_THRESHOLD) {
            float cx = row_ptr[0];
            float cy = row_ptr[1];
            float w = row_ptr[2];
            float h = row_ptr[3];

            // Convert center coordinates to top-left corner coordinates and scale back to original image size
            int left = static_cast<int>((cx - 0.5f * w) * x_factor);
            int top = static_cast<int>((cy - 0.5f * h) * y_factor);
            int width = static_cast<int>(w * x_factor);
            int height = static_cast<int>(h * y_factor);

            boxes.push_back(cv::Rect(left, top, width, height));
            confidences.push_back(score);
            class_ids.push_back(0); // Person class
        }
    }

    // 6. Perform Non-Maximum Suppression (NMS) to remove overlapping detections
    std::cout << "[INFO] Applying Non-Maximum Suppression (NMS)..." << std::endl;
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD, indices);

    std::cout << "\n==============================================" << std::endl;
    std::cout << " INFERENCE RESULTS SUMMARY" << std::endl;
    std::cout << "==============================================" << std::endl;
    std::cout << " Total candidate anchors thresholded: " << boxes.size() << std::endl;
    std::cout << " Final validated targets after NMS: " << indices.size() << std::endl;
    
    for (int idx : indices) {
        cv::Rect box = boxes[idx];
        std::cout << " -> [Target Detected] Class: Person | Conf: " << confidences[idx] * 100.0f 
                  << "% | Bbox: [" << box.x << ", " << box.y << ", " << box.width << ", " << box.height << "]" << std::endl;
    }
    std::cout << "==============================================" << std::endl;

    return 0;
}
