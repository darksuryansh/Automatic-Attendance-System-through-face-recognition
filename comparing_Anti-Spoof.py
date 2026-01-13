import cv2
import pickle
import numpy as np
from deepface import DeepFace
from face_recognition import face_encodings, face_distance

# Configuration
ENCODINGS_FILE = 'data/encodings.pkl'
SPOOF_THRESHOLD = 0.7  # Minimum score to consider face real
RECOGNITION_THRESHOLD = 0.5  # Face recognition tolerance (lower is stricter)
DETECTOR_BACKEND = "opencv"  # or "retinaface", "mtcnn", etc.

def load_encodings():
    """Load facial encodings from PKL file"""
    try:
        with open(ENCODINGS_FILE, 'rb') as f:
            data = pickle.load(f)
        print(f"Loaded {len(data['encodings'])} encodings for {len(set(data['names']))} people")
        return data['encodings'], data['names']
    except Exception as e:
        print(f"Error loading encodings: {e}")
        return [], []

def extract_face_encodings(face_img):
    """Extract 128D face encodings"""
    rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    encodings = face_encodings(rgb_face)
    return encodings[0] if encodings else None

def recognize_face(face_encoding, known_encodings, known_names):
    """Compare face with known encodings"""
    distances = face_distance(known_encodings, face_encoding)
    best_match_idx = np.argmin(distances)
    min_distance = distances[best_match_idx]
    
    if min_distance <= RECOGNITION_THRESHOLD:
        return known_names[best_match_idx], 1 - min_distance
    return None, None

def main():
    # Load known faces
    known_encodings, known_names = load_encodings()
    if not known_encodings:
        print("No encodings found! Please create encodings first.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Couldn't read frame")
            break

        # Detect faces with anti-spoofing
        try:
            face_objs = DeepFace.extract_faces(
                img_path=frame,
                detector_backend=DETECTOR_BACKEND,
                anti_spoofing=True,
                enforce_detection=False
            )
        except Exception as e:
            print(f"Face detection error: {e}")
            continue

        for face_obj in face_objs:
            x, y, w, h = face_obj["facial_area"]["x"], face_obj["facial_area"]["y"], \
                         face_obj["facial_area"]["w"], face_obj["facial_area"]["h"]
            
            # Check if face is real
            is_real = face_obj["is_real"] and (face_obj["antispoof_score"] >= SPOOF_THRESHOLD)
            
            if is_real:
                # Extract face ROI and get encoding
                face_roi = frame[y:y+h, x:x+w]
                face_encoding = extract_face_encodings(face_roi)
                
                if face_encoding is not None:
                    # Recognize face
                    name, confidence = recognize_face(face_encoding, known_encodings, known_names)
                    label = f"{name} ({confidence:.2f})" if name else "Unknown"
                    color = (0, 255, 0)  # Green for recognized
                else:
                    label = "No encoding"
                    color = (0, 255, 255)  # Yellow for detection issues
            else:
                label = f"Spoof ({face_obj['antispoof_score']:.2f})"
                color = (0, 0, 255)  # Red for spoofed

            # Draw bounding box and label
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Secure Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    