from flask import Flask, request, jsonify, render_template, redirect, url_for
import numpy as np
import librosa
import os

# Set protobuf environment variable before importing tensorflow
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import tensorflow as tf
try:
    import tensorflow_hub as hub
except Exception as e:
    print(f"Warning: tensorflow_hub import failed: {e}")
    hub = None
from sklearn.preprocessing import LabelEncoder
from pymongo import MongoClient
from urllib.parse import quote_plus
from datetime import datetime
import json
from emotion_detector import (
    load_emotion_model, 
    detect_emotions_in_image, 
    detect_emotions_in_video,
    detect_emotions_from_webcam
    , analyze_classroom_emotions
)

app = Flask(__name__)

# Helper function to convert MongoDB ObjectIds to strings for JSON serialization
from bson import ObjectId

def convert_to_serializable(obj):
    """Recursively convert ObjectId to string for JSON serialization."""
    if isinstance(obj, ObjectId):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj

# Load models and label encoder
try:
    print("Loading YAMNet model...")
    # Try multiple paths for YAMNet model
    yamnet_paths = [
        "C:/Users/ASUS/Desktop/yamnet-tensorflow2-yamnet-v1",
        "D:/www/yamnet-tensorflow2-yamnet-v1",
        "./yamnet-tensorflow2-yamnet-v1",
    ]
    
    yamnet_model = None
    for path in yamnet_paths:
        try:
            if os.path.exists(path):
                yamnet_model = tf.saved_model.load(path)
                print(f"✓ YAMNet model loaded from: {path}")
                break
        except Exception as e:
            continue
    
    # If not found locally, use TensorFlow Hub (automatic download on first use)
    if yamnet_model is None:
        print("YAMNet model not found locally. Will download from TensorFlow Hub on first use.")
        yamnet_model = None  # Will be lazy-loaded from hub when needed
        
except Exception as e:
    print(f"Error loading YAMNet model: {e}")
    yamnet_model = None

try:
    print("Loading trained model...")
    trained_model = tf.keras.models.load_model('D:/www/trained_model.keras')
    print("Trained model loaded successfully!")
except Exception as e:
    print(f"Error loading trained model: {e}")
    trained_model = None

try:
    print("Loading label encoder...")
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load('D:/www/label_encoder.npy', allow_pickle=True)
    print("Label encoder loaded successfully!")
except Exception as e:
    print(f"Error loading label encoder: {e}")
    label_encoder = None

# MongoDB Atlas connection string
MONGO_URI = "mongodb+srv://cote8806:1234567890@cluster0.vfzv2.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Initialize MongoDB client with error handling
try:
    print("Connecting to MongoDB...")
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    # Test the connection
    client.admin.command('ping')
    print("MongoDB connection successful!")
    
    # Access the database and collections
    db = client['class']
    users_collection = db['users']
    class_collection = db['class_details']  # New collection for storing class details
    # Additional collections for history
    audio_collection = db['audio_analysis']
    emotion_collection = db['emotion_analysis']
except Exception as e:
    print(f"MongoDB connection failed: {e}")
    print("App will run without database functionality")
    client = None
    db = None
    users_collection = None
    class_collection = None
    audio_collection = None
    emotion_collection = None


def extract_features(audio_data, sr):
    global yamnet_model
    
    # Lazy-load YAMNet from TensorFlow Hub if not already loaded
    if yamnet_model is None:
        try:
            print("Loading YAMNet from TensorFlow Hub...")
            # Load YAMNet model from TensorFlow Hub
            yamnet_model = tf.saved_model.load('https://tfhub.dev/google/yamnet/1')
            print("✓ YAMNet model loaded from TensorFlow Hub")
        except Exception as e:
            print(f"Error loading YAMNet from hub: {e}")
            return {"error": "YAMNet model not available"}
    
    # Ensure audio is float32 and in [-1, 1]
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)
    if np.max(np.abs(audio_data)) > 1.0:
        audio_data = audio_data / np.max(np.abs(audio_data))
    
    # Ensure sample rate is 16kHz for YAMNet
    if sr != 16000:
        audio_data = librosa.resample(y=audio_data, orig_sr=sr, target_sr=16000)
    
    # Split audio into 10-second chunks
    chunk_size = 16000 * 10  # 10 seconds per chunk
    chunks = [audio_data[i:i + chunk_size] for i in range(0, len(audio_data), chunk_size)]
    embeddings = []

    for chunk in chunks:
        # Pad if necessary
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
        
        # Convert to tensorflow tensor
        chunk = tf.convert_to_tensor(chunk, dtype=tf.float32)
        chunk = tf.reshape(chunk, [-1])  # Flatten to 1D array
        
        # Get embeddings from YAMNet
        try:
            scores, embeddings_chunk, _ = yamnet_model(chunk)
            embeddings.append(tf.reduce_mean(embeddings_chunk, axis=0))
        except Exception as e:
            print(f"Error processing audio chunk: {e}")
            continue

    if not embeddings:
        return {"error": "Could not process audio"}
    
    # Average embeddings across chunks
    avg_embedding = tf.reduce_mean(tf.stack(embeddings), axis=0)
    return avg_embedding.numpy()

def process_audio(file_path):
    try:
        # Check if models are loaded
        if yamnet_model is None:
            return {"error": "YAMNet model not loaded. Please check model path."}
        if trained_model is None:
            return {"error": "Trained model not loaded. Please check model path."}
        if label_encoder is None:
            return {"error": "Label encoder not loaded. Please check label encoder path."}
        
        # Load audio file
        audio, sr = librosa.load(file_path, sr=None)
        
        # Extract features using YAMNet
        features = extract_features(audio, sr)
        
        # Reshape for model input
        features = features.reshape(1, -1)
        
        # Get model predictions
        predictions = trained_model.predict(features)
        
        # Convert predictions to class probabilities
        results = {}
        for i, prob in enumerate(predictions[0]):
            class_name = label_encoder.classes_[i]
            results[class_name] = float(prob * 100)
        
        return results
        
    except Exception as e:
        print(f"Error processing audio: {e}")
        return {"error": str(e)}

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if users_collection is None:
            # Skip authentication if database is not available
            print("Database not available, skipping authentication")
            return redirect(url_for('index'))
            
        user = users_collection.find_one({'username': username})
        if user and user['password'] == password:
            return redirect(url_for('index'))  # Redirect to index after login
        return jsonify({'message': 'Invalid credentials'}), 401
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')  # Consider hashing the password
        
        if users_collection is None:
            print("Database not available, cannot create user")
            return jsonify({'message': 'Database not available'}), 500
            
        if username and email and password:
            users_collection.insert_one({
                'username': username,
                'email': email,
                'password': password  # Consider hashing the password
            })
            return redirect(url_for('login'))  # Redirect to login after sign-up
        return jsonify({'message': 'Missing fields'}), 400
    return render_template('signup.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'audio' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Save uploaded file temporarily
    temp_path = 'temp_audio.wav'
    file.save(temp_path)
    
    try:
        # Process the audio file
        results = process_audio(temp_path)
        # Store audio analysis result in DB (if available)
        try:
            if audio_collection is not None and isinstance(results, dict) and not results.get('error'):
                record = {
                    'type': 'audio',
                    'timestamp': datetime.utcnow().isoformat(),
                    'filename': file.filename,
                    'result': results
                }
                audio_collection.insert_one(record)
        except Exception as e:
            print(f"Failed to store audio record: {e}")

        # Clean up
        os.remove(temp_path)

        return jsonify(results)
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({'error': str(e)}), 500

@app.route('/submit', methods=['POST'])
def submit_data():
    if request.method == 'POST':
        if class_collection is None:
            print("Database not available, cannot store data")
            return jsonify({'message': 'Database not available'}), 500
            
        class_details = {
            "class": request.form.get('class'),
            "section": request.form.get('section'),
            "date": request.form.get('date'),
            "time": request.form.get('time'),
            "pie_chart": request.form.get('pie_chart')
        }
        class_collection.insert_one(class_details)
        return jsonify({'message': 'Data stored successfully'})


@app.route('/submit_all', methods=['POST'])
def submit_all():
    """Store combined classroom, audio and emotion pie charts and labels in DB."""
    if class_collection is None:
        print("Database not available, cannot store combined data")
        return jsonify({'message': 'Database not available'}), 500

    # Accept JSON or form data
    if request.is_json:
        data = request.get_json()
    else:
        # form fields may contain JSON strings
        data = request.form.to_dict()

    # Basic server-side validation for required metadata
    required_fields = ['class', 'section', 'date', 'time']
    missing = [f for f in required_fields if not data.get(f) or str(data.get(f)).strip() == '']
    if missing:
        return jsonify({'message': 'Missing required fields', 'missing': missing}), 400

    try:
        record = {
            'type': 'combined',
            'timestamp': datetime.utcnow().isoformat(),
            'class': data.get('class'),
            'section': data.get('section'),
            'date': data.get('date'),
            'time': data.get('time'),
            'classroom_pie': None,
            'audio_pie': None,
            'emotion_pie': None,
            'labels': None
        }

        # Parse pie fields if present
        try:
            if data.get('classroom_pie'):
                record['classroom_pie'] = json.loads(data.get('classroom_pie')) if isinstance(data.get('classroom_pie'), str) else data.get('classroom_pie')
        except Exception:
            record['classroom_pie'] = data.get('classroom_pie')

        try:
            if data.get('audio_pie'):
                record['audio_pie'] = json.loads(data.get('audio_pie')) if isinstance(data.get('audio_pie'), str) else data.get('audio_pie')
        except Exception:
            record['audio_pie'] = data.get('audio_pie')

        try:
            if data.get('emotion_pie'):
                record['emotion_pie'] = json.loads(data.get('emotion_pie')) if isinstance(data.get('emotion_pie'), str) else data.get('emotion_pie')
        except Exception:
            record['emotion_pie'] = data.get('emotion_pie')

        try:
            if data.get('labels'):
                record['labels'] = json.loads(data.get('labels')) if isinstance(data.get('labels'), str) else data.get('labels')
        except Exception:
            record['labels'] = data.get('labels')

        class_collection.insert_one(record)
        return jsonify({'message': 'Combined analysis stored successfully'})
    except Exception as e:
        print(f"Failed to store combined record: {e}")
        return jsonify({'message': 'Failed to store data', 'error': str(e)}), 500

@app.route('/history/data')
def history_data():
    if class_collection is None:
        return jsonify({
            'data': [],
            'total_pages': 0,
            'current_page': 1,
            'error': 'Database not available'
        })
    
    page = request.args.get('page', 1, type=int)
    per_page = 5
    skip = (page - 1) * per_page
    
    # Get total count for pagination
    total_count = class_collection.count_documents({})
    
    # Fetch paginated data from the class_details collection
    data = list(class_collection.find({}, {"_id": 0}).skip(skip).limit(per_page))
    
    return jsonify({
        'data': data,
        'total_pages': (total_count + per_page - 1) // per_page,
        'current_page': page
    })


@app.route('/history/all_data')
def history_all_data():
    """Return combined history from classroom, audio, and emotion collections."""
    # Check DB availability
    if class_collection is None and audio_collection is None and emotion_collection is None:
        return jsonify({'data': [], 'total_pages': 0, 'current_page': 1, 'error': 'Database not available'})

    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    type_filter = request.args.get('type', '', type=str)  # optional filter: audio, emotion, classroom

    combined = []

    # Fetch classroom and combined records
    try:
        if class_collection is not None:
            for doc in class_collection.find({}):
                # Use the type from the document if present, otherwise default to 'classroom'
                doc_type = doc.get('type', 'classroom')
                item = {
                    '_id': doc.get('_id'),
                    'type': doc_type,
                    'timestamp': None,
                    'data': doc
                }
                # try to derive timestamp from date/time fields or use server timestamp
                dt = None
                try:
                    if doc.get('timestamp'):
                        dt = doc.get('timestamp')
                    elif 'date' in doc and 'time' in doc and doc.get('date') and doc.get('time'):
                        # assume format 'YYYY-MM-DD' and 'HH:MM'
                        dt = f"{doc.get('date')}T{doc.get('time')}"
                except Exception:
                    dt = None
                item['timestamp'] = dt
                combined.append(item)
    except Exception as e:
        print(f"Failed to fetch classroom history: {e}")

    # Fetch audio records
    try:
        if audio_collection is not None:
            for doc in audio_collection.find({}):
                item = {
                    '_id': doc.get('_id'),
                    'type': 'audio',
                    'timestamp': doc.get('timestamp'),
                    'data': doc
                }
                combined.append(item)
    except Exception as e:
        print(f"Failed to fetch audio history: {e}")

    # Fetch emotion records
    try:
        if emotion_collection is not None:
            for doc in emotion_collection.find({}):
                item = {
                    '_id': doc.get('_id'),
                    'type': 'emotion',
                    'timestamp': doc.get('timestamp'),
                    'data': doc
                }
                combined.append(item)
    except Exception as e:
        print(f"Failed to fetch emotion history: {e}")

    # Optionally filter by type
    if type_filter:
        combined = [c for c in combined if c.get('type') == type_filter]

    # Sort by timestamp (desc). Items without timestamp go to the end.
    def sort_key(x):
        t = x.get('timestamp')
        return t if t is not None else ''

    combined.sort(key=lambda x: sort_key(x) or '', reverse=True)

    total_count = len(combined)
    total_pages = (total_count + per_page - 1) // per_page if per_page > 0 else 1

    # Paginate
    start = (page - 1) * per_page
    end = start + per_page
    page_items = combined[start:end]

    return jsonify({
        'data': convert_to_serializable(page_items),
        'total_pages': total_pages,
        'current_page': page,
        'total_count': total_count
    })

@app.route('/history/delete/<record_id>', methods=['DELETE'])
def delete_history_record(record_id):
    """Delete a history record by ID."""
    try:
        from bson import ObjectId
        
        # Try to delete from class_collection first
        if class_collection is not None:
            try:
                result = class_collection.delete_one({'_id': ObjectId(record_id)})
                if result.deleted_count > 0:
                    return jsonify({'success': True, 'message': 'Record deleted successfully'})
            except Exception as e:
                print(f"Error deleting from class_collection: {e}")
        
        # Try audio_collection
        if audio_collection is not None:
            try:
                result = audio_collection.delete_one({'_id': ObjectId(record_id)})
                if result.deleted_count > 0:
                    return jsonify({'success': True, 'message': 'Record deleted successfully'})
            except Exception as e:
                print(f"Error deleting from audio_collection: {e}")
        
        # Try emotion_collection
        if emotion_collection is not None:
            try:
                result = emotion_collection.delete_one({'_id': ObjectId(record_id)})
                if result.deleted_count > 0:
                    return jsonify({'success': True, 'message': 'Record deleted successfully'})
            except Exception as e:
                print(f"Error deleting from emotion_collection: {e}")
        
        return jsonify({'success': False, 'error': 'Record not found'}), 404
    except Exception as e:
        print(f"Delete error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/history')
def view_history():
    return render_template('history.html')

@app.route('/index')
def index():
    return render_template('index.html')

# ============== EMOTION DETECTION ENDPOINTS ==============

@app.route('/emotion-detect', methods=['POST'])
def emotion_detect_image():
    """
    Detect emotions from an uploaded image.
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image file uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Save uploaded file temporarily
    temp_path = 'temp_image.jpg'
    file.save(temp_path)
    
    try:
        # Detect emotions
        result = detect_emotions_in_image(temp_path)
        # Store emotion image analysis in DB
        try:
            if emotion_collection is not None and isinstance(result, dict) and not result.get('error'):
                record = {
                    'type': 'emotion_image',
                    'timestamp': datetime.utcnow().isoformat(),
                    'filename': file.filename,
                    'result': result
                }
                emotion_collection.insert_one(record)
        except Exception as e:
            print(f"Failed to store emotion image record: {e}")

        # Clean up
        os.remove(temp_path)

        return jsonify(result)
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({'error': str(e)}), 500

@app.route('/emotion-detect-video', methods=['POST'])
def emotion_detect_video():
    """
    Detect emotions from an uploaded video file.
    Samples frames to reduce processing time.
    """
    if 'video' not in request.files:
        return jsonify({'error': 'No video file uploaded'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Save uploaded file temporarily
    temp_path = 'temp_video.mp4'
    file.save(temp_path)
    
    try:
        # Get number of frames to sample (default: 10)
        sample_frames = request.form.get('sample_frames', 10, type=int)
        
        # Detect emotions
        result = detect_emotions_in_video(temp_path, sample_frames=sample_frames)
        # Store emotion video analysis in DB
        try:
            if emotion_collection is not None and isinstance(result, dict) and not result.get('error'):
                record = {
                    'type': 'emotion_video',
                    'timestamp': datetime.utcnow().isoformat(),
                    'filename': file.filename,
                    'sample_frames': sample_frames,
                    'result': result
                }
                emotion_collection.insert_one(record)
        except Exception as e:
            print(f"Failed to store emotion video record: {e}")

        # Clean up
        os.remove(temp_path)

        return jsonify(result)
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({'error': str(e)}), 500

@app.route('/emotion-detect-webcam', methods=['POST'])
def emotion_detect_webcam():
    """
    Detect emotions from webcam capture.
    Captures video for specified duration (default: 10 seconds).
    """
    try:
        # Get duration from request (default: 10 seconds)
        duration = request.json.get('duration', 10) if request.json else 10
        
        # Validate duration
        if duration <= 0 or duration > 60:
            return jsonify({'error': 'Duration must be between 1 and 60 seconds'}), 400
        
        # Detect emotions
        result = detect_emotions_from_webcam(duration_seconds=duration)

        # Store webcam emotion analysis in DB
        try:
            if emotion_collection is not None and isinstance(result, dict) and not result.get('error'):
                record = {
                    'type': 'emotion_webcam',
                    'timestamp': datetime.utcnow().isoformat(),
                    'duration_seconds': duration,
                    'result': result
                }
                emotion_collection.insert_one(record)
        except Exception as e:
            print(f"Failed to store webcam emotion record: {e}")

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500



@app.route('/classroom-analyze', methods=['POST'])
def classroom_analyze_image():
    """Analyze a classroom image with multi-face detection and return aggregated results."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image file uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    temp_path = 'temp_class_image.jpg'
    try:
        file.save(temp_path)
        print(f"[DEBUG] Classroom image saved to {temp_path}")

        result = analyze_classroom_emotions(temp_path)
        print(f"[DEBUG] Classroom analysis result: {result}")

        # Optionally store summary in DB
        if class_collection is not None and isinstance(result, dict) and result.get('status') == 'success':
            summary = {
                'type': 'classroom',
                'timestamp': datetime.utcnow().isoformat(),
                'total_faces_detected': result.get('total_faces_detected'),
                'total_faces_analyzed': result.get('total_faces_analyzed'),
                'emotions_summary': result.get('emotions_summary'),
                'dominant_emotion': result.get('dominant_emotion')
            }
            try:
                class_collection.insert_one(summary)
            except Exception as e:
                print(f"Failed to store class summary: {e}")

        if os.path.exists(temp_path):
            os.remove(temp_path)

        return jsonify(result)
    except Exception as e:
        print(f"[ERROR] Classroom analysis failed: {e}")
        import traceback
        traceback.print_exc()
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({'error': str(e), 'status': 'error'}), 500


if __name__ == '__main__':
    app.run(debug=False, use_reloader=False)
