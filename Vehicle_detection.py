from ultralytics import YOLO
import cv2
import numpy as np
import os

class TrafficDetector:
    def __init__(self, general_model_path, ambulance_model_path):
        # Load both models into the class
        self.general_model = YOLO(general_model_path)
        self.ambulance_model = YOLO(ambulance_model_path)

        # 1. General Model Classes (COCO standard)
        self.general_classes = {2: 'Car', 3: 'Motorcycle', 5: 'Bus', 7: 'Truck'}
        
        # 2. Your Custom Ambulance Model Class
        # Based on your data.yaml, this is ID 0
        self.amb_class_id = 0 
        
        # This ensures the label "Ambulance" is applied to that ID 0
        self.ambulance_names = {0: 'Ambulance'}

        # 3. Specific Color Mapping
        self.class_colors = {
            'Car': (255, 0, 0),        # Blue
            'Motorcycle': (0, 255, 255), # Yellow
            'Bus': (0, 255, 0),        # Green
            'Truck': (0, 0, 255),       # Red
            'Ambulance': (255, 255, 255)  # White
        }
        self.default_color = (169, 169, 169)

        print("System Initialized: General and Ambulance models loaded.")

    def process_frame(self, frame):
        vehicle_count = 0
        ambulance_detected = False

        # --- RUN GENERAL DETECTION ---
        # Using 640 for speed on standard vehicles
        gen_results = self.general_model(frame, imgsz=640, conf=0.25, verbose=False)
        for r in gen_results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls in self.general_classes:
                    vehicle_count += 1
                    self.draw_prediction(frame, box, self.general_classes[cls])

        # --- RUN AMBULANCE DETECTION ---
        # Using 1024 for better accuracy on your custom trained class
        amb_results = self.ambulance_model(frame, imgsz=1024, conf=0.35, verbose=False)
        for r in amb_results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls == self.amb_class_id:
                    ambulance_detected = True
                    self.draw_prediction(frame, box, "Ambulance")

        return frame, vehicle_count, ambulance_detected

    def draw_prediction(self, frame, box, label_name):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        color = self.class_colors.get(label_name, self.default_color)

        # Draw Bold Bounding Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)

        # Draw Label with filled background
        text_label = f"{label_name} {conf:.2f}"
        (w, h), _ = cv2.getTextSize(text_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
        
        # Black text for Ambulance (on white), White text for others
        text_color = (0, 0, 0) if label_name == 'Ambulance' else (255, 255, 255)
        cv2.putText(frame, text_label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2, cv2.LINE_AA)

# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    # Define paths
    gen_path = r"D:\M.Tech\ISR\project\ISR_project\models\yolo26n.pt"
    amb_path = r"D:\M.Tech\ISR\project\ISR_project\models\ambulance.pt"
    
    # Initialize detector with both models
    detector = TrafficDetector(gen_path, amb_path)

    video_path = r"D:\M.Tech\ISR\project\ISR_project\videos\video4.mp4"
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Process the frame through the dual-model system
        processed_frame, count, emergency = detector.process_frame(frame)

        # Overlay basic info
        status_text = f"Vehicles: {count} | Emergency: {'YES' if emergency else 'No'}"
        cv2.putText(processed_frame, status_text, (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Display at a readable size
        display_frame = cv2.resize(processed_frame, (1280, 720))
        cv2.imshow("ISR Project - Dual Model Detection", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


