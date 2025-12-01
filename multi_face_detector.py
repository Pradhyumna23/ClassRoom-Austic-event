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
        # DeepFace.extract_faces accepts path or array; use array
        try:
            faces = DeepFace.extract_faces(img_path=image, detector_backend='retinaface', enforce_detection=False)
            results = []
            for f in faces:
                area = f.get('facial_area', {})
                x = int(area.get('x', 0))
                y = int(area.get('y', 0))
                w = int(area.get('w', 0))
                h = int(area.get('h', 0))
                results.append({'x': x, 'y': y, 'width': w, 'height': h, 'confidence': 0.9})
            return results
        except Exception:
            return []


def boxes_overlap(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int], threshold: float = 0.3) -> bool:
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
    """Combine MediaPipe + RetinaFace (DeepFace) detections and merge duplicates."""
    def __init__(self):
        self.mpd = MediaPipeFaceDetector()

    def detect_faces(self, image) -> List[Dict]:
        mp_faces = self.mpd.detect_faces(image)
        rf_faces = RetinaFaceWrapper.detect_faces(image)

        # merge, prefer MediaPipe boxes but add RetinaFace boxes that are not duplicates
        merged = list(mp_faces)
        for rf in rf_faces:
            rf_box = (rf['x'], rf['y'], rf['width'], rf['height'])
            duplicate = False
            for mp in mp_faces:
                mp_box = (mp['x'], mp['y'], mp['width'], mp['height'])
                if boxes_overlap(mp_box, rf_box):
                    duplicate = True
                    break
            if not duplicate:
                merged.append(rf)
        return merged
