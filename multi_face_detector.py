import cv2
from deepface import DeepFace
try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except Exception:
    mp = None
    HAS_MEDIAPIPE = False
from typing import List, Dict, Tuple


if HAS_MEDIAPIPE:
    class MediaPipeFaceDetector:
        """Fast face detection using MediaPipe (CPU-friendly). Returns list of bounding boxes."""
        def __init__(self, min_detection_confidence: float = 0.5):
            self.mp_face_detection = mp.solutions.face_detection
            self.detector = self.mp_face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=min_detection_confidence
            )

        def detect_faces(self, image) -> List[Dict]:
            """Detect faces in a BGR image (numpy array). Returns list of dicts with x,y,width,height,confidence."""
            if image is None:
                return []

            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.detector.process(img_rgb)
            faces = []
            if results.detections:
                h, w, _ = image.shape
                for det in results.detections:
                    bbox = det.location_data.relative_bounding_box
                    x = int(max(0, bbox.xmin * w))
                    y = int(max(0, bbox.ymin * h))
                    width = int(min(w - x, bbox.width * w))
                    height = int(min(h - y, bbox.height * h))
                    # add small padding
                    pad = 8
                    x = max(0, x - pad)
                    y = max(0, y - pad)
                    width = min(w - x, width + pad * 2)
                    height = min(h - y, height + pad * 2)
                    faces.append({'x': x, 'y': y, 'width': width, 'height': height, 'confidence': float(det.score[0]) if det.score else 0.0})
            return faces
else:
    # Fallback stub so the module can be imported when mediapipe is not available.
    class MediaPipeFaceDetector:
        def __init__(self, *args, **kwargs):
            print('Warning: mediapipe not available — MediaPipeFaceDetector will not detect faces')

        def detect_faces(self, image) -> List[Dict]:
            return []


class RetinaFaceWrapper:
    """Wrap DeepFace.extract_faces (RetinaFace) to return bounding boxes for merging."""
    @staticmethod
    def detect_faces(image) -> List[Dict]:
        # DeepFace.extract_faces with RetinaFace backend - good for frontal faces
        try:
            # Lower confidence threshold to catch more faces, including partial/angled ones
            faces = DeepFace.extract_faces(img_path=image, detector_backend='retinaface', 
                                          enforce_detection=False)
            results = []
            for f in faces:
                area = f.get('facial_area', {})
                x = int(area.get('x', 0))
                y = int(area.get('y', 0))
                w = int(area.get('w', 0))
                h = int(area.get('h', 0))
                results.append({'x': x, 'y': y, 'width': w, 'height': h, 'confidence': 0.85, 'detector': 'retinaface'})
            print(f"[DEBUG] RetinaFace detected {len(results)} faces")
            return results
        except Exception as e:
            print(f"[DEBUG] RetinaFace detection error: {e}")
            return []


class OpenCVFaceDetector:
    """Use OpenCV Haar Cascade for additional face detection (good for profile/angled faces)."""
    def __init__(self):
        # Load pre-trained Haar Cascade classifier for faces
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
        self.cascade = cv2.CascadeClassifier(cascade_path)
        print(f"[DEBUG] Loaded Haar Cascade from: {cascade_path}")
    
    @staticmethod
    def detect_faces(image) -> List[Dict]:
        """Detect faces using OpenCV Haar Cascade - better for profile/tilted faces."""
        try:
            # Convert to grayscale for Haar Cascade
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Detect faces with scale factor optimized for multiple faces
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
            cascade = cv2.CascadeClassifier(cascade_path)
            faces = cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=4, 
                                            minSize=(30, 30), maxSize=(500, 500))
            
            results = []
            for (x, y, w, h) in faces:
                results.append({'x': x, 'y': y, 'width': w, 'height': h, 'confidence': 0.7, 'detector': 'haar_cascade'})
            
            print(f"[DEBUG] Haar Cascade detected {len(results)} faces")
            return results
        except Exception as e:
            print(f"[DEBUG] Haar Cascade detection error: {e}")
            return []


def boxes_overlap(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int], threshold: float = 0.1) -> bool:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1, box2: (x, y, width, height) tuples
        threshold: IoU threshold above which boxes are considered overlapping
                  - 0.1: Only heavily overlapping boxes marked as duplicates (allows close faces)
                  - 0.5: More aggressive duplicate removal (prevents close together faces)
    
    Returns:
        True if IoU > threshold (boxes overlap significantly)
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(x1 + w1, x2 + w2)
    inter_y2 = min(y1 + h1, y2 + h2)
    if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
        return False
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - inter_area
    if union <= 0:
        return False
    overlap = inter_area / union
    return overlap > threshold


class EnsembleFaceDetector:
    """
    Combine THREE detection methods for maximum coverage:
    1. RetinaFace - Excellent for frontal faces
    2. MediaPipe - Good for profile/tilted faces
    3. Haar Cascade - Robust for varied angles and lighting
    
    This ensemble approach catches faces looking in different directions.
    """
    def __init__(self):
        self.mpd = MediaPipeFaceDetector(min_detection_confidence=0.5)  # Lower confidence for side profiles
        self.haar = OpenCVFaceDetector()
        print("[DEBUG] EnsembleFaceDetector initialized with 3 detection methods")

    def detect_faces(self, image) -> List[Dict]:
        """
        Detect faces using all three methods, merge results intelligently.
        Strategy: Use all detectors, remove only very high-overlap duplicates (IoU > 0.5)
        """
        # Get detections from all three methods
        rf_faces = RetinaFaceWrapper.detect_faces(image)
        mp_faces = self.mpd.detect_faces(image)
        haar_faces = self.haar.detect_faces(image)
        
        print(f"[DEBUG] === Face Detection Summary ===")
        print(f"[DEBUG] RetinaFace: {len(rf_faces)} faces")
        print(f"[DEBUG] MediaPipe: {len(mp_faces)} faces")
        print(f"[DEBUG] Haar Cascade: {len(haar_faces)} faces")
        
        # Combine all detections
        all_faces = rf_faces + mp_faces + haar_faces
        merged = []
        used_indices = set()
        
        # NMS-style merging: keep high-confidence detections, remove overlapping lower-confidence ones
        all_faces_sorted = sorted(enumerate(all_faces), key=lambda x: x[1]['confidence'], reverse=True)
        
        for idx, face in all_faces_sorted:
            if idx in used_indices:
                continue
            
            face_box = (face['x'], face['y'], face['width'], face['height'])
            is_duplicate = False
            
            # Check against already accepted faces
            for accepted_face in merged:
                accepted_box = (accepted_face['x'], accepted_face['y'], 
                               accepted_face['width'], accepted_face['height'])
                # Use stricter threshold (0.5) to avoid merging distinct faces
                if boxes_overlap(face_box, accepted_box, threshold=0.5):
                    is_duplicate = True
                    print(f"[DEBUG] Face from {face.get('detector', 'unknown')} (conf: {face['confidence']}) "
                          f"overlaps with existing detection - skipping")
                    break
            
            if not is_duplicate:
                merged.append(face)
                used_indices.add(idx)
                print(f"[DEBUG] ✓ Accepted face from {face.get('detector', 'unknown')} "
                      f"(confidence: {face['confidence']:.2f})")
        
        print(f"[DEBUG] Total unique faces after ensemble merge: {len(merged)}")
        print(f"[DEBUG] === End Detection ===\n")
        return merged
