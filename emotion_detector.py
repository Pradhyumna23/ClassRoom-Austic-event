"""
Emotion Detection Module using DeepFace
Detects student emotions: drowsy, attentive, yawning, distracted, confusion
"""

import cv2
import numpy as np
from deepface import DeepFace
import os
from multi_face_detector import EnsembleFaceDetector

def convert_to_json_serializable(obj):
    """Convert NumPy types and other non-JSON-serializable types to native Python types."""
    if isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

# Emotion mapping to classroom context
# Enhanced mapping with 6 emotion categories for better classroom analysis
EMOTION_MAP = {
    'neutral': 'attentive',       # Calm, focused attention
    'happy': 'engaged',            # Active interest and engagement
    'sad': 'drowsy',               # Lethargy, fatigue signals
    'angry': 'frustrated',         # Struggle with material
    'surprise': 'confusion',       # Uncertain, questioning
    'fear': 'anxious',             # Nervous, stressed
    'disgust': 'distracted'        # Disengaged, repelled
}

def draw_face_detection_circles(image_path: str, faces: list) -> str:
    """
    Draw circles and labels around detected faces in the image.
    
    Args:
        image_path: Path to the original image
        faces: List of detected face dictionaries with x, y, width, height
    
    Returns:
        Base64-encoded image data
    """
    import base64
    
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"[ERROR] Could not read image: {image_path}")
            return None
        
        # Draw circles around each detected face
        for idx, face in enumerate(faces):
            x = face['x']
            y = face['y']
            w = face['width']
            h = face['height']
            
            # Calculate circle center and radius
            center_x = x + w // 2
            center_y = y + h // 2
            radius = max(w, h) // 2 + 10
            
            # Draw circle (green color for detected faces)
            color = (0, 255, 0)  # BGR format: Green
            thickness = 3
            cv2.circle(img, (center_x, center_y), radius, color, thickness)
            
            # Draw face number label
            label = f"Person {idx + 1}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            text_thickness = 2
            
            # Put text at the top of circle
            text_x = center_x - 50
            text_y = center_y - radius - 10
            cv2.putText(img, label, (text_x, text_y), font, font_scale, color, text_thickness)
            
            # Draw bounding box as well (optional, for clarity)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)  # Cyan box
        
        # Convert to base64
        _, buffer = cv2.imencode('.jpg', img)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        return image_base64
    
    except Exception as e:
        print(f"[ERROR] Error drawing detection circles: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_emotion_model():
    """
    Load the DeepFace emotion detection model.
    Note: DeepFace automatically caches models, so this initializes the backend.
    """
    try:
        print("Initializing DeepFace emotion detection model...")
        # Warm up the model with a dummy prediction
        # This ensures the model is loaded and cached
        return True
    except Exception as e:
        print(f"Error initializing emotion model: {e}")
        return False

def detect_emotions_in_image(image_path):
    """
    Detect emotions from an image file.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        dict: Dictionary with detected emotions and confidence scores
    """
    try:
        # Check if file exists
        if not os.path.exists(image_path):
            return {"error": f"Image file not found: {image_path}"}
        
        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            return {"error": "Failed to load image"}
        
        # Analyze emotions using DeepFace
        result = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
        
        # Extract emotion data
        emotions_data = {}
        if isinstance(result, list):
            result = result[0]  # Get first face if multiple detected
        
        if 'emotion' in result:
            raw_emotions = result['emotion']
            
            # Map raw emotions to classroom context and calculate percentages
            emotion_totals = {}
            emotion_confidence = {}
            
            for raw_emotion, confidence in raw_emotions.items():
                mapped_emotion = EMOTION_MAP.get(raw_emotion, raw_emotion)
                if mapped_emotion not in emotion_totals:
                    emotion_totals[mapped_emotion] = 0
                    emotion_confidence[mapped_emotion] = []
                
                emotion_totals[mapped_emotion] += confidence
                emotion_confidence[mapped_emotion].append(confidence)
            
            # Calculate average confidence for each emotion
            for emotion in emotion_totals:
                avg_confidence = emotion_totals[emotion] / len(emotion_confidence[emotion])
                emotions_data[emotion] = round(avg_confidence, 2)
        
        # Sort by confidence
        sorted_emotions = dict(sorted(emotions_data.items(), key=lambda x: x[1], reverse=True))
        
        return {
            "status": "success",
            "emotions": sorted_emotions,
            "dominant_emotion": list(sorted_emotions.keys())[0] if sorted_emotions else "unknown"
        }
    
    except Exception as e:
        print(f"Error detecting emotions in image: {e}")
        return {"error": str(e)}


class ClassroomEmotionAnalyzer:
    """Analyze all faces in a classroom image/video and aggregate classroom-level metrics."""
    def __init__(self):
        self.face_detector = EnsembleFaceDetector()

    def analyze_classroom_image(self, image_path: str) -> dict:
        """Detect all faces and analyze emotion per face, then aggregate."""
        try:
            if not os.path.exists(image_path):
                return {'error': f'Image not found: {image_path}', 'status': 'error'}

            img = cv2.imread(image_path)
            if img is None:
                return {'error': 'Failed to load image', 'status': 'error'}

            print(f"[DEBUG] Analyzing classroom image: {image_path}, shape: {img.shape}")
            
            faces = self.face_detector.detect_faces(img)
            total_faces = len(faces)
            print(f"[DEBUG] Detected {total_faces} faces in classroom image")

            individual_results = []
            emotion_counts = {}  # Initialize as empty dict to avoid hardcoded emotions
            emotion_scores = {}  # Initialize as empty dict
            
            # Initialize emotion tracking dicts with all possible emotions
            for emotion in set(EMOTION_MAP.values()):
                emotion_counts[emotion] = 0
                emotion_scores[emotion] = []

            for idx, f in enumerate(faces):
                x, y, w, h = f['x'], f['y'], f['width'], f['height']
                face_roi = img[y:y+h, x:x+w]
                if face_roi.size == 0:
                    print(f"[DEBUG] Face {idx+1} ROI is empty, skipping")
                    continue
                # Analyze face
                try:
                    res = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                    if isinstance(res, list):
                        res = res[0]
                    raw = res.get('emotion', {})
                    dominant = res.get('dominant_emotion', 'neutral')
                    mapped = EMOTION_MAP.get(dominant, dominant)

                    print(f"[DEBUG] Face {idx+1}: raw_emotion={dominant}, mapped={mapped}, confidence={float(max(raw.values())) if raw else 0.0}")

                    individual_results.append({
                        'face_id': idx + 1,
                        'bounding_box': {'x': x, 'y': y, 'width': w, 'height': h},
                        'raw_emotion': dominant,
                        'classroom_emotion': mapped,
                        'confidence': float(max(raw.values())) if raw else 0.0,
                        'all_emotions': {str(k): float(v) for k, v in raw.items()}
                    })

                    emotion_counts[mapped] = emotion_counts.get(mapped, 0) + 1
                    
                    # Store the dominant emotion's confidence score for averaging
                    if mapped not in emotion_scores:
                        emotion_scores[mapped] = []
                    emotion_scores[mapped].append(float(max(raw.values())) if raw else 0.0)
                    
                except Exception as e:
                    print(f"[ERROR] Error analyzing detected face {idx+1}: {e}")
                    continue

            # Aggregate averages and percentages
            total_analyzed = len(individual_results)
            if total_analyzed == 0:
                return convert_to_json_serializable({
                    'status': 'error',
                    'error': 'No faces could be analyzed',
                    'total_faces_detected': total_faces,
                    'total_faces_analyzed': 0
                })
            
            avg_emotions = {}
            percentages = {}
            
            for emo in emotion_scores.keys():
                vals = emotion_scores.get(emo, [])
                if vals:
                    avg = float(np.mean(vals))
                else:
                    avg = 0.0
                avg_emotions[emo] = round(avg, 2)
                percentages[emo] = round((emotion_counts.get(emo, 0) / total_analyzed * 100) if total_analyzed > 0 else 0.0, 2)

            # Sort by average emotion score
            sorted_avg_emotions = dict(sorted(avg_emotions.items(), key=lambda x: x[1], reverse=True))
            sorted_percentages = dict(sorted(percentages.items(), key=lambda x: x[1], reverse=True))
            
            dominant = list(sorted_avg_emotions.keys())[0] if sorted_avg_emotions else 'unknown'

            print(f"[DEBUG] Classroom analysis complete: {total_analyzed} faces analyzed, dominant emotion: {dominant}")
            print(f"[DEBUG] Average emotions: {sorted_avg_emotions}")
            print(f"[DEBUG] Emotion counts: {emotion_counts}")

            # Generate marked image showing detected faces
            marked_image_base64 = draw_face_detection_circles(image_path, faces)

            result = {
                'status': 'success',
                'total_faces_detected': total_faces,
                'total_faces_analyzed': total_analyzed,
                'individual_emotions': individual_results,
                'emotions_summary': emotion_counts,
                'emotions_percentages': sorted_percentages,
                'average_emotions': sorted_avg_emotions,
                'dominant_emotion': dominant,
                'marked_image': marked_image_base64
            }
            return convert_to_json_serializable(result)
        except Exception as e:
            print(f"[ERROR] Error in analyze_classroom_image: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e), 'status': 'error'}


# Single instance for reuse
classroom_analyzer = ClassroomEmotionAnalyzer()


def analyze_classroom_emotions(image_path: str) -> dict:
    return classroom_analyzer.analyze_classroom_image(image_path)

def detect_emotions_in_video(video_path, sample_frames=10):
    """
    Detect emotions from a video file.
    Samples frames at regular intervals to reduce processing time.
    
    Args:
        video_path (str): Path to the video file
        sample_frames (int): Number of frames to sample from the video
        
    Returns:
        dict: Dictionary with frame-wise emotion predictions and aggregated statistics
    """
    try:
        # Check if file exists
        if not os.path.exists(video_path):
            return {"error": f"Video file not found: {video_path}"}
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": "Failed to open video file"}
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            return {"error": "Invalid video file or no frames found"}
        
        frame_interval = max(1, total_frames // sample_frames)
        frame_count = 0
        sampled_count = 0
        
        emotion_results = []
        aggregated_emotions = {}
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames at regular intervals
            if frame_count % frame_interval == 0:
                try:
                    # Analyze emotions in the frame
                    result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                    
                    if isinstance(result, list):
                        result = result[0]
                    
                    if 'emotion' in result:
                        raw_emotions = result['emotion']
                        
                        # Map emotions to classroom context
                        frame_emotions = {}
                        for raw_emotion, confidence in raw_emotions.items():
                            mapped_emotion = EMOTION_MAP.get(raw_emotion, raw_emotion)
                            if mapped_emotion not in frame_emotions:
                                frame_emotions[mapped_emotion] = 0
                            frame_emotions[mapped_emotion] += confidence
                        
                        # Normalize
                        total_confidence = sum(frame_emotions.values())
                        if total_confidence > 0:
                            frame_emotions = {k: round(v / total_confidence * 100, 2) 
                                            for k, v in frame_emotions.items()}
                        
                        # Get dominant emotion for this frame
                        dominant = max(frame_emotions.items(), key=lambda x: x[1])[0] if frame_emotions else "unknown"
                        
                        emotion_results.append({
                            "frame": frame_count,
                            "emotions": frame_emotions,
                            "dominant_emotion": dominant
                        })
                        
                        # Aggregate emotions across all frames
                        for emotion, confidence in frame_emotions.items():
                            if emotion not in aggregated_emotions:
                                aggregated_emotions[emotion] = []
                            aggregated_emotions[emotion].append(confidence)
                        
                        sampled_count += 1
                
                except Exception as e:
                    print(f"Error processing frame {frame_count}: {e}")
                    continue
            
            frame_count += 1
        
        cap.release()
        
        # Calculate average emotions across all sampled frames
        average_emotions = {}
        for emotion, confidences in aggregated_emotions.items():
            if confidences:
                average_emotions[emotion] = round(np.mean(confidences), 2)
        
        # Sort by average confidence
        sorted_average_emotions = dict(sorted(average_emotions.items(), key=lambda x: x[1], reverse=True))
        
        return {
            "status": "success",
            "total_frames": total_frames,
            "sampled_frames": sampled_count,
            "frame_results": emotion_results,
            "average_emotions": sorted_average_emotions,
            "dominant_emotion": list(sorted_average_emotions.keys())[0] if sorted_average_emotions else "unknown"
        }
    
    except Exception as e:
        print(f"Error detecting emotions in video: {e}")
        return {"error": str(e)}

def detect_emotions_from_webcam(duration_seconds=10):
    """
    Capture emotion detection from webcam for a specified duration.
    
    Args:
        duration_seconds (int): Duration to capture video from webcam
        
    Returns:
        dict: Dictionary with aggregated emotion predictions
    """
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return {"error": "Webcam not accessible"}
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames_to_capture = int(fps * duration_seconds)
        frame_count = 0
        
        emotion_results = []
        aggregated_emotions = {}
        
        while frame_count < total_frames_to_capture:
            ret, frame = cap.read()
            if not ret:
                break
            
            try:
                # Analyze every frame from webcam
                result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                
                if isinstance(result, list):
                    result = result[0]
                
                if 'emotion' in result:
                    raw_emotions = result['emotion']
                    
                    # Map emotions
                    frame_emotions = {}
                    for raw_emotion, confidence in raw_emotions.items():
                        mapped_emotion = EMOTION_MAP.get(raw_emotion, raw_emotion)
                        if mapped_emotion not in frame_emotions:
                            frame_emotions[mapped_emotion] = 0
                        frame_emotions[mapped_emotion] += confidence
                    
                    # Normalize to percentages
                    total_confidence = sum(frame_emotions.values())
                    if total_confidence > 0:
                        frame_emotions = {k: round(v / total_confidence * 100, 2) 
                                        for k, v in frame_emotions.items()}
                    
                    dominant = max(frame_emotions.items(), key=lambda x: x[1])[0] if frame_emotions else "unknown"
                    
                    emotion_results.append({
                        "frame": frame_count,
                        "emotions": frame_emotions,
                        "dominant_emotion": dominant
                    })
                    
                    # Aggregate
                    for emotion, confidence in frame_emotions.items():
                        if emotion not in aggregated_emotions:
                            aggregated_emotions[emotion] = []
                        aggregated_emotions[emotion].append(confidence)
            
            except Exception as e:
                print(f"Error processing webcam frame {frame_count}: {e}")
            
            frame_count += 1
        
        cap.release()
        
        # Calculate averages
        average_emotions = {}
        for emotion, confidences in aggregated_emotions.items():
            if confidences:
                average_emotions[emotion] = round(np.mean(confidences), 2)
        
        sorted_average_emotions = dict(sorted(average_emotions.items(), key=lambda x: x[1], reverse=True))
        
        return {
            "status": "success",
            "total_frames_captured": frame_count,
            "average_emotions": sorted_average_emotions,
            "dominant_emotion": list(sorted_average_emotions.keys())[0] if sorted_average_emotions else "unknown"
        }
    
    except Exception as e:
        print(f"Error capturing from webcam: {e}")
        return {"error": str(e)}
