# 🎓 Classroom Audio-Visual Emotion Detection System

## Project Overview

**Classroom Austic Event** is an AI-powered system that analyzes student classroom behavior through two complementary modules:

1. **Audio Event Classification** - Detects classroom sounds and events (teacher speech, student discussion, silence, etc.)
2. **Emotion Detection** - Analyzes student facial expressions to detect emotional states

This system helps teachers understand classroom dynamics, student engagement levels, and identify students who may need attention or support.

---

## ✨ Key Features

### 🔊 Audio Analysis
- Detects classroom events using YAMNet model
- Classifies sounds into categories:
  - Teacher Speech
  - Student Discussion
  - Silence/Background noise
  - Other classroom activities
- Processes audio files and real-time streams
- Stores analysis results in MongoDB

### 😊 Emotion Detection
- **6 Emotion Categories**:
  - 😊 **Attentive** - Student is focused and calm
  - 👀 **Engaged** - Student is actively interested
  - 🤔 **Confused** - Student is uncertain or questioning
  - 😕 **Distracted** - Student is disengaged
  - 🥱 **Drowsy** - Student is tired/fatigued
  - 😤 **Frustrated** - Student is struggling
  - 😰 **Anxious** - Student is nervous/stressed
  - 😑 **Yawning** - Student is extremely tired

- **Multiple Input Methods**:
  - 📷 Image upload (single frame analysis)
  - 🎬 Video upload (multi-frame analysis)
  - 📹 Webcam recording (live capture)

- **Visual Dashboard**:
  - Emoji indicators for each emotion
  - Color-coded results (Green=Good, Red=Problem, Purple=Alert)
  - Interactive pie charts
  - Percentage breakdown

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    WEB INTERFACE (Flask)                     │
│              (HTML, CSS, JavaScript, Chart.js)               │
└─────────────────────────────────────────────────────────────┘
              ↓                                    ↓
    ┌──────────────────┐            ┌──────────────────────┐
    │  Audio Analysis  │            │ Emotion Detection    │
    │  Module          │            │ Module               │
    └──────────────────┘            └──────────────────────┘
              ↓                                    ↓
    ┌──────────────────┐            ┌──────────────────────┐
    │  YAMNet Model    │            │  DeepFace Model      │
    │  TensorFlow      │            │  (RetinaFace +       │
    │  (16kHz audio)   │            │   VGGFace2)          │
    └──────────────────┘            └──────────────────────┘
              ↓                                    ↓
    ┌──────────────────┐            ┌──────────────────────┐
    │  Custom Trained  │            │  Emotion Mapper      │
    │  Classification  │            │  (7 emotions → 6+)   │
    │  Model           │            │                      │
    └──────────────────┘            └──────────────────────┘
              ↓                                    ↓
              └──────────┬──────────────────────┬───────────┘
                         ↓
          ┌──────────────────────────────────┐
          │   MongoDB Atlas (Cloud DB)       │
          │   • User credentials             │
          │   • Analysis results             │
          │   • Class details                │
          └──────────────────────────────────┘
```

---

## 🤖 Models Used

### 1. YAMNet Audio Classification Model
- **Source**: TensorFlow Hub
- **Purpose**: Pre-trained model for audio event classification
- **Input**: 16 kHz mono audio
- **Output**: 521 sound event classes with confidence scores
- **Framework**: TensorFlow 2.12.0

**Audio Processing Pipeline**:
```
Audio File → Resample to 16kHz → Extract Features → 
YAMNet Model → Classification → Custom Training Model → 
Classroom Event Labels
```

### 2. Custom Trained Classification Model
- **Source**: `trained_model.keras` (Keras/TensorFlow)
- **Purpose**: Maps YAMNet embeddings to classroom-specific event categories
- **Features**: Uses YAMNet's internal representations
- **Architecture**: Dense neural network layers
- **Performance**: Optimized for classroom environment sounds

### 3. DeepFace Emotion Detection
- **Source**: DeepFace 0.0.79 library
- **Face Detection**: RetinaFace (fast, accurate)
- **Emotion Recognition**: VGGFace2 backend
- **Input**: Images or video frames
- **Output**: 7 raw emotions (happy, sad, angry, surprise, fear, neutral, disgust)

**Emotion Processing Pipeline**:
```
Image/Video → Face Detection (RetinaFace) → 
Face Alignment → Emotion Extraction (VGGFace2) →
Emotion Mapping → Classroom Context (6+ emotions)
```

### 4. Emotion Mapping System
- **7 DeepFace Emotions** → **6+ Classroom Emotions**
- **Mapping Logic**:
  ```
  happy       → Engaged (active interest)
  neutral     → Attentive (calm focus)
  sad         → Drowsy (fatigue signals)
  angry       → Frustrated (struggling)
  surprise    → Confusion (uncertain)
  fear        → Anxious (stressed)
  disgust     → Distracted (disengaged)
  ```

---

## 📋 Tech Stack

### Backend
- **Framework**: Flask 2.0.1
- **Python**: 3.10
- **Audio Processing**: librosa 0.9.2
- **ML Frameworks**:
  - TensorFlow 2.12.0
  - TensorFlow Hub 0.12.0
  - Keras 2.12.0
  - DeepFace 0.0.79
  - scikit-learn (preprocessing)

### Frontend
- **Languages**: HTML5, CSS3, JavaScript (Vanilla)
- **Visualization**: Chart.js (pie charts)
- **Media Capture**: MediaRecorder API, getUserMedia API

### Database
- **MongoDB Atlas** (Cloud)
- **Collections**:
  - `users` - User credentials
  - `class_details` - Analysis results and metadata

### Infrastructure
- **Web Server**: Flask (development)
- **Dependencies Management**: pip, virtualenv

---

## 🚀 Installation & Setup

### Prerequisites
- Windows/Linux/macOS
- Python 3.10
- pip (Python package manager)
- Virtual environment (recommended)

### Step 1: Clone the Repository
```bash
git clone https://github.com/Pradhyumna23/ClassRoom-Austic-event.git
cd ClassRoom-Austic-event
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv myenv
myenv\Scripts\activate

# Linux/macOS
python3 -m venv myenv
source myenv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download Required Models

#### A. YAMNet Model (Audio Classification)
```bash
# The model is automatically downloaded from TensorFlow Hub
# Or manually download and place in:
# C:/Users/ASUS/Desktop/yamnet-tensorflow2-yamnet-v1
```

**Note**: Update the path in `app.py` line 20 if you're using a different location:
```python
yamnet_model = tf.saved_model.load("YOUR_MODEL_PATH")
```

#### B. Trained Model (Already Included)
- `trained_model.keras` - Pre-trained classifier
- `label_encoder.npy` - Label mappings

### Step 5: Configure MongoDB (Optional)
Update the MongoDB connection string in `app.py` (line 47):
```python
MONGO_URI = "your_mongodb_connection_string"
```

If you don't have MongoDB:
- Create free account at [MongoDB Atlas](https://www.mongodb.com/cloud/atlas)
- Create a cluster
- Get connection string
- Or run app without database (reduced functionality)

### Step 6: Run the Application
```bash
python app.py
```

**Expected Output**:
```
Loading YAMNet model...
YAMNet model loaded successfully!
Loading trained model...
Trained model loaded successfully!
Loading label encoder...
Label encoder loaded successfully!
Connecting to MongoDB...
MongoDB connection successful!
 * Running on http://127.0.0.1:5000
```

### Step 7: Access the Web Interface
Open your browser and go to:
```
http://localhost:5000
```

---

## 📊 How It Works

### Audio Analysis Workflow
1. **Upload Audio File** (WAV, MP3, etc.)
2. **Preprocessing**:
   - Resample to 16 kHz (YAMNet requirement)
   - Normalize audio levels
   - Split into 10-second chunks
3. **Feature Extraction**:
   - YAMNet processes audio chunks
   - Extracts acoustic embeddings
4. **Classification**:
   - Custom model predicts classroom event
   - Generates confidence scores
5. **Storage**:
   - Results saved to MongoDB
   - Visualization in dashboard

### Emotion Detection Workflow
1. **Input Source**:
   - 📷 Upload image, 🎬 Upload video, or 📹 Record from webcam
2. **Face Detection**:
   - RetinaFace detects all faces in frame
   - Extracts face regions
3. **Emotion Analysis**:
   - VGGFace2 extracts emotion features
   - Generates 7 emotion probabilities
4. **Emotion Mapping**:
   - Maps raw emotions to classroom context
   - Calculates averages (for videos)
   - Determines dominant emotion
5. **Visualization**:
   - Shows emoji indicators
   - Color-coded display
   - Pie chart with percentages
   - Detailed breakdown

### Webcam Recording Features
- **Start/Stop buttons** for user control
- **Live preview** during recording
- **Video preview** after recording stops
- **Automatic analysis** when recording stops
- **Real-time emotion display** with results

---

## 🎯 Usage Examples

### Example 1: Audio Event Detection
```bash
1. Open http://localhost:5000
2. Go to "Audio Analysis" section
3. Click "Upload Audio"
4. Select an MP3 or WAV file
5. Click "Submit Audio Analysis"
6. View pie chart showing detected events:
   - Teacher Speech: 45%
   - Student Discussion: 30%
   - Background Noise: 15%
   - Silence: 10%
```

### Example 2: Emotion Detection (Image)
```bash
1. Open http://localhost:5000
2. Go to "Emotion Detection" section
3. Click "Upload Image"
4. Select a photo of a student
5. System analyzes:
   - Detects face(s)
   - Analyzes emotions
   - Maps to classroom emotions
6. View results:
   - 😊 ATTENTIVE (75.50%)
   - 👀 Engaged (15.25%)
   - 🤔 Confused (9.25%)
```

### Example 3: Emotion Detection (Video)
```bash
1. Click "Upload Video"
2. Select a video file
3. System samples 10 frames from video
4. Analyzes each frame
5. Averages emotions across frames
6. Displays multi-frame analysis with timeline
```

### Example 4: Webcam Recording
```bash
1. Click "🔴 Start Recording" button
2. Live webcam feed appears
3. Speak or show expressions
4. Click "⏹️ Stop Recording" button
5. Recording stops, video preview appears
6. Automatic emotion analysis begins
7. Results display with 6 emotions
```

---

## 📂 Project Structure

```
ClassRoom-Austic-event/
├── app.py                          # Main Flask application
├── emotion_detector.py             # Emotion detection module
├── requirements.txt                # Python dependencies
├── trained_model.keras             # Pre-trained audio classifier
├── label_encoder.npy               # Audio label mappings
├── style.css                       # Dashboard styling
├── templates/
│   ├── index.html                  # Main dashboard
│   ├── login.html                  # Login page
│   ├── signup.html                 # Signup page
│   └── history.html                # Analysis history
├── data/
│   └── audio_clip/                 # Sample audio files
│       ├── STUDENT/                # Student sounds
│       ├── TEACHER/                # Teacher sounds
│       ├── DISTURBANCE/            # Background noise
│       └── BACKGROUND NOISE/       # Environmental sounds
├── README.md                       # This file
└── Documentation/
    ├── EMOTION_ENHANCEMENT_PLAN.md
    ├── IMPLEMENTATION_STATUS.md
    ├── VISUAL_IMPLEMENTATION_GUIDE.md
    └── ... (other guides)
```

---

## 🔧 Configuration

### Audio Processing Settings (app.py)
```python
# Sample rate for audio processing
SAMPLE_RATE = 16000  # YAMNet requirement

# Audio chunk size
CHUNK_DURATION = 10  # seconds

# Number of video frames to analyze
SAMPLE_FRAMES = 10
```

### Emotion Detection Settings (emotion_detector.py)
```python
# Confidence threshold for emotion detection
MIN_CONFIDENCE = 0.0

# Video sampling
VIDEO_SAMPLE_FRAMES = 10

# Webcam recording duration (can be user-controlled)
WEBCAM_DURATION = 10  # seconds
```

### MongoDB Settings (app.py)
```python
# Database name
DATABASE_NAME = "class"

# Collections
USERS_COLLECTION = "users"
CLASS_COLLECTION = "class_details"
```

---

## 🎓 API Endpoints

### Audio Analysis
- **POST** `/upload` - Upload audio file
- **POST** `/submit` - Analyze uploaded audio
- **GET** `/history` - View analysis history

### Emotion Detection
- **POST** `/emotion-detect` - Analyze image emotions
- **POST** `/emotion-detect-video` - Analyze video emotions
- **POST** `/emotion-detect-webcam` - Analyze webcam recording

### User Management
- **POST** `/signup` - Create new user account
- **POST** `/login` - User login
- **GET** `/logout` - User logout

### Dashboard
- **GET** `/` - Main dashboard
- **GET** `/history` - View past analyses

---

## 📊 Expected Outputs

### Audio Analysis Output
```json
{
  "status": "success",
  "detected_events": {
    "Teacher Speech": 45.2,
    "Student Discussion": 30.1,
    "Background Noise": 15.3,
    "Silence": 9.4
  },
  "duration": 120.5,
  "confidence": 0.89
}
```

### Emotion Detection Output
```json
{
  "status": "success",
  "dominant_emotion": "attentive",
  "average_emotions": {
    "attentive": 75.50,
    "engaged": 15.25,
    "confused": 5.00,
    "distracted": 3.00,
    "drowsy": 1.00,
    "frustrated": 0.25
  },
  "emotion_breakdown": {
    "😊 Attentive": 75.50,
    "👀 Engaged": 15.25,
    "🤔 Confused": 5.00,
    "😕 Distracted": 3.00,
    "🥱 Drowsy": 1.00,
    "😤 Frustrated": 0.25
  }
}
```

---

## ⚙️ System Requirements

### Minimum Requirements
- **CPU**: Intel i5 or equivalent (for real-time processing)
- **RAM**: 8 GB (16 GB recommended for multiple concurrent analyses)
- **Disk Space**: 2 GB (for models and temporary files)
- **Network**: Internet connection (for MongoDB Atlas)

### Recommended Requirements
- **CPU**: Intel i7 or better
- **RAM**: 16 GB
- **GPU**: NVIDIA GPU with CUDA support (for faster processing)
- **Disk**: SSD with 5+ GB free space

---

## 🐛 Troubleshooting

### Issue: YAMNet model not found
**Solution**:
```bash
# Download model from TensorFlow Hub or update path in app.py
# Or reinstall tensorflow-hub
pip install --upgrade tensorflow-hub
```

### Issue: DeepFace model download fails
**Solution**:
```bash
# Clear cache and reinstall
pip uninstall deepface -y
pip install deepface==0.0.79
```

### Issue: MongoDB connection fails
**Solution**:
1. Check MongoDB Atlas connection string
2. Verify IP whitelist in MongoDB settings
3. Or run without database (limited functionality)

### Issue: Audio file not processed
**Solution**:
1. Ensure audio is 16 kHz mono (resample if needed)
2. Check file format is supported (WAV, MP3, OGG)
3. Verify file is not corrupted

### Issue: Webcam not working
**Solution**:
1. Check browser permissions for camera access
2. Verify camera is not in use by another app
3. Try in different browser
4. Restart browser and try again

---

## 📝 Logging & Debugging

### Enable Debug Mode
```python
# In app.py, set debug=True
if __name__ == '__main__':
    app.run(debug=True)
```

### Check Application Logs
Logs are printed to console and include:
- Model loading status
- Database connections
- Processing errors
- Analysis results

### Enable Verbose Output
```bash
# Run with verbose logging
FLASK_ENV=development FLASK_DEBUG=1 python app.py
```

---

## 📜 License

This project is licensed under the MIT License - see LICENSE file for details.

---

## 👥 Contributors

**Pradhyumna23** - Project Lead & Developer

## 📧 Contact

For questions, issues, or suggestions:
- GitHub: [Pradhyumna23/ClassRoom-Austic-event](https://github.com/Pradhyumna23/ClassRoom-Austic-event)
- Email: Contact through GitHub profile

---

## 🎉 Features & Improvements

### Current Features
- ✅ Audio event classification (YAMNet + Custom model)
- ✅ Emotion detection with 6+ categories
- ✅ Webcam recording with live preview
- ✅ Image and video emotion analysis
- ✅ Color-coded visual display
- ✅ Emoji indicators
- ✅ MongoDB data storage
- ✅ Responsive web dashboard

### Future Enhancements
- 🔄 Real-time streaming analysis
- 🔄 Advanced analytics dashboard
- 🔄 Class performance metrics
- 🔄 Student engagement tracking
- 🔄 Teacher behavior analysis
- 🔄 Mobile application
- 🔄 Multi-camera support
- 🔄 Export reports functionality

---

## 📚 Documentation

Comprehensive documentation available in the project:
- `EMOTION_ENHANCEMENT_PLAN.md` - Emotion detection details
- `IMPLEMENTATION_STATUS.md` - Implementation guide
- `VISUAL_IMPLEMENTATION_GUIDE.md` - Visual diagrams
- `QUICK_START.md` - Quick start guide
- And 7+ more detailed guides

---

## ✅ Verification Checklist

Before deploying to production:
- [ ] All models downloaded and paths configured
- [ ] MongoDB connection tested
- [ ] All dependencies installed correctly
- [ ] Application runs without errors
- [ ] Audio analysis working
- [ ] Emotion detection working
- [ ] Webcam recording tested
- [ ] Dashboard displays correctly

---

## 🎯 Quick Start (TL;DR)

```bash
# 1. Clone & setup
git clone <repo>
cd ClassRoom-Austic-event
python -m venv myenv
myenv\Scripts\activate  # Windows or source myenv/bin/activate on Linux

# 2. Install & run
pip install -r requirements.txt
python app.py

# 3. Open browser
http://localhost:5000

# 4. Start analyzing!
# Upload audio/images/videos or use webcam
```

---

**Last Updated**: November 11, 2025  
**Version**: 2.0 (with Enhanced Emotion Detection)  
**Status**: Production Ready ✅

🎓 Happy Analyzing! 🎓
