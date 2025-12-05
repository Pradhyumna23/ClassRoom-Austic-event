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
        # Load multiple cascade classifiers for better coverage
        self.cascade_frontal = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
        self.cascade_alt = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
        self.cascade_default = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        print(f"[DEBUG] Loaded 3 Haar Cascade variants for multi-angle detection")
    
    def detect_faces(self, image) -> List[Dict]:
        """Detect faces using multiple Haar Cascades - aggressive for different angles and lighting."""
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Equalize histogram for better contrast (helps with poor lighting)
            gray = cv2.equalizeHist(gray)
            
            all_detected = {}  # Use dict to deduplicate by position
            
            # Try all three cascades with aggressive parameters
            cascades = [
                ('alt2', self.cascade_frontal, {'scaleFactor': 1.03, 'minNeighbors': 3}),
                ('alt', self.cascade_alt, {'scaleFactor': 1.05, 'minNeighbors': 4}),
                ('default', self.cascade_default, {'scaleFactor': 1.08, 'minNeighbors': 5})
            ]
            
            for name, cascade, params in cascades:
                faces = cascade.detectMultiScale(
                    gray,
                    scaleFactor=params['scaleFactor'],
                    minNeighbors=params['minNeighbors'],
                    minSize=(25, 25),  # Lower minimum size to catch distant faces
                    maxSize=(600, 600),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                
                print(f"[DEBUG] Haar {name} detected {len(faces)} faces")
                
                # Add detections to dict (key is rounded position to avoid duplicates)
                for (x, y, w, h) in faces:
                    # Round to nearest 10 pixels to group similar detections
                    key = (round(x/10)*10, round(y/10)*10, round(w/10)*10, round(h/10)*10)
                    if key not in all_detected or w * h > all_detected[key][2] * all_detected[key][3]:
                        all_detected[key] = (x, y, w, h)
            
            results = []
            for (x, y, w, h) in all_detected.values():
                results.append({
                    'x': x, 'y': y, 'width': w, 'height': h, 
                    'confidence': 0.75,  # Slightly lower confidence for Haar
                    'detector': 'haar_cascade'
                })
            
            print(f"[DEBUG] Haar Cascade (deduplicated) detected {len(results)} unique faces")
            return results
        except Exception as e:
            print(f"[DEBUG] Haar Cascade detection error: {e}")
            import traceback
            traceback.print_exc()
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
        Strategy: Aggressive detection - keep all unique faces, only remove near-exact duplicates
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
        
        # Sort by confidence (prefer higher confidence detections)
        all_faces_sorted = sorted(enumerate(all_faces), key=lambda x: x[1].get('confidence', 0.5), reverse=True)
        
        for idx, face in all_faces_sorted:
            if idx in used_indices:
                continue
            
            face_box = (face['x'], face['y'], face['width'], face['height'])
            is_duplicate = False
            
            # Check against already accepted faces
            for accepted_face in merged:
                accepted_box = (accepted_face['x'], accepted_face['y'], 
                               accepted_face['width'], accepted_face['height'])
                
                # Use VERY strict threshold (0.6-0.7) to only merge near-exact duplicates
                # This prevents merging faces that are close together
                if boxes_overlap(face_box, accepted_box, threshold=0.65):
                    is_duplicate = True
                    detector1 = face.get('detector', 'unknown')
                    detector2 = accepted_face.get('detector', 'unknown')
                    print(f"[DEBUG] Face from {detector1} (conf: {face.get('confidence', 0):.2f}) "
                          f"is duplicate of {detector2} - keeping {detector2}")
                    break
            
            if not is_duplicate:
                merged.append(face)
                used_indices.add(idx)
                print(f"[DEBUG] ✓ Accepted face #{len(merged)} from {face.get('detector', 'unknown')} "
                      f"at ({face['x']}, {face['y']}) size {face['width']}x{face['height']} "
                      f"(confidence: {face.get('confidence', 0):.2f})")
        
        print(f"[DEBUG] Total UNIQUE faces after ensemble: {len(merged)}")
        print(f"[DEBUG] === End Detection ===\n")
        
        # Sort by position (top-left to bottom-right) for consistency
        merged = sorted(merged, key=lambda f: (f['y'], f['x']))
        
        return merged
