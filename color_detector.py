import cv2
import numpy as np
from collections import deque

class SmartColorDetector:
    def detect_shape(self, contour):
        contour = cv2.convexHull(contour)

        shape = "Unknown"
        perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)

        # Cek lingkaran dahulu
        ((cx, cy), radius) = cv2.minEnclosingCircle(contour)
        if radius > 5:
            circle_area = radius * radius * np.pi
            circularity = area / circle_area
            if 0.85 < circularity < 1.15:
                return "Circle"

        # Analisa poligon
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        num_vertices = len(approx)

        if num_vertices == 3:
            shape = "Triangle"
        elif num_vertices == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            if 0.90 <= aspect_ratio <= 1.10:
                shape = "Square"
            else:
                shape = "Rectangle"
                
        return shape

    def __init__(self):
        self.color_ranges = {
            'Red': {
                'lower1': np.array([0, 130, 80]), 
                'upper1': np.array([10, 255, 255]),
                
                'lower2': np.array([168, 130, 80]), 
                'upper2': np.array([180, 255, 255]),
                
                'color': (0, 0, 255), 'last_position': None, 'velocity': (0, 0),
                'trail': deque(maxlen=20)
            },
            'Yellow': {
                'lower1': np.array([20, 120, 100]), 
                'upper1': np.array([35, 255, 255]),
                
                'color': (0, 255, 255), 'last_position': None, 'velocity': (0, 0),
                'trail': deque(maxlen=20)
            },
            'Blue': { # Warna kalibrasi
                'lower1': np.array([100, 150, 50]), 'upper1': np.array([140, 255, 255]),
                'color': (255, 0, 0)
            }
        }
        
        self.pixels_per_cm = None
        self.collision_detected = False
        self.proximity_threshold = 80
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        except Exception as e:
            print(f"Warning: Could not load face cascade. Face detection disabled. Error: {e}")
            self.face_cascade = None

    def create_mask(self, hsv, color_name):
        color_info = self.color_ranges[color_name]
    
        if 'lower2' in color_info:
            mask1 = cv2.inRange(hsv, color_info['lower1'], color_info['upper1'])
            mask2 = cv2.inRange(hsv, color_info['lower2'], color_info['upper2'])
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            mask = cv2.inRange(hsv, color_info['lower1'], color_info['upper1'])
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return mask

    def detect_with_simple_markers(self, frame, mask, color_name):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color_info = self.color_ranges[color_name]
        detected_objects = []

        if len(contours) > 0:
            for c in contours:
                area = cv2.contourArea(c)
                if area < 500:
                    continue

                shape_name = self.detect_shape(c) 
                M = cv2.moments(c)
                if M["m00"] > 0:
                    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                    label = f"{color_name} {shape_name}"
                    (x, y, w, h) = cv2.boundingRect(c)
                    cv2.putText(frame, label, (x, y - 15), cv2.FONT_HERSHEY_DUPLEX, 0.7, color_info['color'], 2)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color_info['color'], 3)
                    
                    detected_objects.append({
                        'center': center, 
                        'area': area, 
                        'color_name': color_name,
                        'contour': c
                    })

        if detected_objects:
            largest_object = max(detected_objects, key=lambda obj: obj['area'])
            center_of_largest = largest_object['center']
            # Memperbarui jejak
            color_info['trail'].append(center_of_largest)
        else:
            # Memutus jejak
            color_info['trail'].append(None)
        
        self.draw_smart_trail(frame, color_info['trail'], color_info['color'])
        
        return detected_objects

    def draw_smart_trail(self, frame, trail, color):
        if len(trail) < 2: return
        for i in range(1, len(trail)):
            if trail[i - 1] is not None and trail[i] is not None:
                alpha = i / len(trail)
                thickness = max(1, int(5 * alpha))
                trail_color = tuple(int(c * alpha) for c in color)
                cv2.line(frame, trail[i - 1], trail[i], trail_color, thickness)

    def detect_object_interaction(self, frame, red_objects, yellow_objects):
        self.collision_detected = False
        
        # Cek objek
        if not red_objects or not yellow_objects:
            return
            
        for red_obj in red_objects:
            for yellow_obj in yellow_objects:
                min_dist = float('inf')
                
                contour1 = red_obj['contour']
                contour2 = yellow_obj['contour']
                
                if len(contour1) > len(contour2):
                    contour1, contour2 = contour2, contour1
                
                for point in contour1:
                    point_float = (float(point[0][0]), float(point[0][1]))
                    
                    dist = cv2.pointPolygonTest(contour2, point_float, True)
                    min_dist = min(min_dist, abs(dist))
                
                distance = min_dist
                
                if distance < self.proximity_threshold:
                    cv2.line(frame, red_obj['center'], yellow_obj['center'], (255, 255, 255), 2)
                    mid_point = (
                        (red_obj['center'][0] + yellow_obj['center'][0]) // 2,
                        (red_obj['center'][1] + yellow_obj['center'][1]) // 2
                    )
                    
                    if distance < 5:
                        self.collision_detected = True
                        for r in range(10, 40, 10):
                            cv2.circle(frame, mid_point, r, (0, 255, 255), 2)
                        cv2.putText(frame, "COLLISION!", (mid_point[0] - 60, mid_point[1] - 50),
                                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 3)
                    else:
                        cv2.circle(frame, mid_point, 15, (255, 255, 255), 2)
                        cv2.putText(frame, f"{int(distance)}px", (mid_point[0] - 20, mid_point[1] + 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def detect_faces(self, frame):
        if self.face_cascade is None: return []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(40, 40))
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 3)
            cv2.putText(frame, "Face Detected!", (x, y - 15),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 255), 2)
        return faces

    def draw_clean_hud(self, frame): 
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (260, 60), (0, 0, 0), -1)
        cv2.addWeighted(frame, 0.85, overlay, 0.15, 0, frame)
        
        hud_info = [
            f"Collision: {'DETECTED!' if self.collision_detected else 'None'}",
            "",
            "Q-Quit | R-Reset"
        ]
        
        for i, text in enumerate(hud_info):
            if text == "": continue
            color = (255, 255, 255)
            if self.collision_detected and "Collision" in text:
                color = (0, 0, 255)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            size = 0.5
            thickness = 1
            
            cv2.putText(frame, text, (20, 30 + i * 20), font, size, color, thickness)

def main():
    print("Smart Red & Yellow Detection")
    
    detector = SmartColorDetector()
    
    camera_index = 0 
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"Cannot open camera at index {camera_index}")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    window_name = "Smart Red & Yellow Detector (Fullscreen)"
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    print(f"Ready! Using camera index {camera_index}.")
    print("Controls: Q=Quit, R=Reset")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame = cv2.flip(frame, 1)
        frame = cv2.bilateralFilter(frame, 5, 50, 50)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        red_mask = detector.create_mask(hsv, 'Red')
        yellow_mask = detector.create_mask(hsv, 'Yellow')
        
        red_objects = detector.detect_with_simple_markers(frame, red_mask, 'Red')
        yellow_objects = detector.detect_with_simple_markers(frame, yellow_mask, 'Yellow')
        
        detector.detect_object_interaction(frame, red_objects, yellow_objects)
        detector.detect_faces(frame)
        detector.draw_clean_hud(frame)
        
        cv2.imshow(window_name, frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27:
            break
        elif key == ord('r'):
            for color_name in ['Red', 'Yellow']:
                detector.color_ranges[color_name]['trail'].clear()
            print("Reset complete!")
    
    cap.release()
    cv2.destroyAllWindows()
    print("Shutdown complete!")

if __name__ == "__main__":
    main()