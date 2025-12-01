# 🚀 Render Deployment Guide - Classroom AI

## Overview
This guide explains how to deploy the Classroom AI application on Render.com with automatic model downloading.

---

## 📋 Pre-Deployment Checklist

✅ **Local Setup Complete:**
- [ ] `app.py` works locally
- [ ] `trained_model.keras` exists
- [ ] `label_encoder.npy` exists
- [ ] `requirements.txt` is up-to-date
- [ ] `.gitignore` configured correctly

✅ **Git Setup:**
- [ ] GitHub repository created
- [ ] All source files pushed (except myenv/, data/)
- [ ] Model files included (`trained_model.keras`, `label_encoder.npy`)

✅ **MongoDB:**
- [ ] MongoDB Atlas account created
- [ ] Cluster deployed
- [ ] Connection string ready (format: `mongodb+srv://user:pass@cluster.mongodb.net/database`)

---

## 🔧 Step 1: Prepare for GitHub Push

### Remove Virtual Environment from Git History

```bash
cd D:\www

# Remove myenv from git tracking (but keep locally)
git rm -r --cached myenv/ --force

# Commit the change
git commit -m "Remove virtual environment from git tracking"

# Push to GitHub
git push -u origin main
```

### Verify Files to Push

```bash
git status
```

Should show:
- ✅ `app.py`, `emotion_detector.py`, `multi_face_detector.py`
- ✅ `requirements.txt`
- ✅ `trained_model.keras`
- ✅ `label_encoder.npy`
- ✅ `templates/` folder
- ✅ `style.css`
- ❌ NOT `myenv/`
- ❌ NOT `data/audio_clip/`

---

## 🌐 Step 2: Set Up on Render.com

### 2.1 Create Render Account
1. Go to [render.com](https://render.com)
2. Sign up with GitHub account
3. Authorize Render to access your repositories

### 2.2 Create New Web Service
1. Dashboard → **New +** → **Web Service**
2. Select repository: `ClassRoom-Austic-event`
3. Configuration:

| Setting | Value |
|---------|-------|
| **Name** | classroom-ai |
| **Environment** | Python 3 |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `gunicorn app:app` |
| **Plan** | Standard (or Free for testing) |

### 2.3 Environment Variables
In Render dashboard, set these **Environment Variables**:

```
MONGO_URI=mongodb+srv://USERNAME:PASSWORD@cluster.mongodb.net/classroom_db
FLASK_ENV=production
DEBUG=False
```

Replace:
- `USERNAME` - Your MongoDB username
- `PASSWORD` - Your MongoDB password
- `cluster` - Your MongoDB Atlas cluster name

---

## ⚙️ Step 3: Update app.py for Production

Make these changes to `app.py`:

```python
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get MongoDB URI from environment
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')

# MongoDB Connection
client = MongoClient(MONGO_URI)
db = client['classroom_db']

# Flask Production Config
app = Flask(__name__)
app.config['DEBUG'] = os.getenv('DEBUG', 'False') == 'True'
```

---

## 🔍 Step 4: Important Notes on Model Loading

### Models That Auto-Download:
- ✅ **YAMNet** - Downloads from TensorFlow Hub (~50 MB)
- ✅ **DeepFace** - Downloads weights automatically (~50-100 MB)
- ✅ **MediaPipe** - Downloads face models automatically (~10-20 MB)

**First request may take 1-2 minutes** as models download. This is normal!

### Local Models (Included in Repo):
- ✅ `trained_model.keras` - Your custom audio classifier
- ✅ `label_encoder.npy` - Audio event label mapping

---

## 🚀 Step 5: Deploy

1. **Push code to GitHub:**
   ```bash
   git push origin main
   ```

2. **Render automatically detects the push** and starts deployment

3. **Monitor deployment in Render dashboard:**
   - Logs show progress
   - Wait for "Your service is live" message

4. **Visit your app:**
   ```
   https://classroom-ai.onrender.com
   ```

---

## 🧪 Testing the Deployment

### Test Endpoints:
```bash
# Test login page
curl https://classroom-ai.onrender.com/

# Test emotion detection
curl -X POST https://classroom-ai.onrender.com/analyze-emotion \
  -F "image=@test_image.jpg"

# Test audio analysis
curl -X POST https://classroom-ai.onrender.com/analyze-audio \
  -F "audio=@test_audio.wav"
```

---

## ⚠️ Common Issues & Solutions

### Issue 1: "Model Download Timeout"
**Problem:** Models take too long to download on first request
**Solution:** 
- Render Free tier has 30-second timeout
- Upgrade to Standard plan for longer builds
- Or use model caching strategy

### Issue 2: "MongoDB Connection Failed"
**Problem:** App can't connect to MongoDB Atlas
**Solution:**
1. Verify MONGO_URI in Render Environment Variables
2. Check IP whitelist in MongoDB Atlas (allow 0.0.0.0/0)
3. Test connection: `mongo "mongodb+srv://user:pass@cluster.mongodb.net"`

### Issue 3: "Out of Memory"
**Problem:** App runs out of memory during model loading
**Solution:**
- Upgrade to Standard plan (more RAM)
- Or switch to smaller models
- Use lazy loading (load models on demand)

### Issue 4: "Large File Upload Timeout"
**Problem:** File uploads timeout during analysis
**Solution:**
- Increase Flask timeout: `app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024`
- Upgrade plan for more resources

---

## 📦 Render Build Time Estimates

| Stage | Time | Notes |
|-------|------|-------|
| Git clone | 10s | |
| Install pip packages | 2-3 min | TensorFlow is large |
| First model download | 1-2 min | YAMNet, DeepFace auto-download |
| Total | ~4-5 min | Only on first deploy |

---

## 🔄 Updating the Application

To update your app:

1. Make changes locally
2. Test with `flask run`
3. Commit: `git add . && git commit -m "Update features"`
4. Push: `git push origin main`
5. Render automatically redeploys within 1-2 minutes

---

## 💾 Database Setup

### Create MongoDB Atlas Cluster:

1. Go to [mongodb.com/cloud](https://mongodb.com/cloud)
2. Create account (free tier available)
3. Create cluster (choose AWS, region closest to you)
4. Create database user with strong password
5. Add IP whitelist (0.0.0.0/0 for Render)
6. Get connection string: `mongodb+srv://user:pass@cluster.mongodb.net/`

### Collections Needed:
```javascript
// Will be auto-created by app.py:
- classroom_db.users
- classroom_db.class_details
- classroom_db.emotion_analysis
- classroom_db.audio_analysis
```

---

## 🎯 Performance Optimization

### For Faster Cold Starts:
1. **Pre-download models** during build (add to Procfile)
2. **Use model caching** with Redis
3. **Lazy load** models on first request

### Current Setup:
- ✅ Models auto-download on first request
- ✅ Subsequent requests use cached models
- ✅ Reasonable for medium traffic

---

## 📞 Support Links

- **Render Docs:** https://render.com/docs
- **Flask Deployment:** https://flask.palletsprojects.com/deployment/
- **MongoDB Atlas:** https://www.mongodb.com/docs/atlas/
- **TensorFlow Hub:** https://tfhub.dev

---

## ✅ Final Checklist Before Deploy

- [ ] `.gitignore` excludes `myenv/`, `data/`
- [ ] `trained_model.keras` is committed
- [ ] `label_encoder.npy` is committed
- [ ] `requirements.txt` has all dependencies
- [ ] `app.py` reads MONGO_URI from environment
- [ ] MongoDB Atlas cluster is ready
- [ ] GitHub repo is public (or Render has access)
- [ ] All code is pushed to main branch

---

## 🎉 You're Ready!

Your app will:
1. ✅ Deploy automatically on GitHub push
2. ✅ Download ML models on first request
3. ✅ Connect to MongoDB Atlas
4. ✅ Run emotion detection in production
5. ✅ Analyze audio in real-time

**Deployment time:** ~5 minutes
**Uptime:** 24/7 (on Standard plan)
**Cost:** Free tier or $7/month Standard

---
