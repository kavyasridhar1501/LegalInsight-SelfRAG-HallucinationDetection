# LegalInsight Deployment Guide

Complete guide to deploying LegalInsight with GitHub Pages frontend and cloud backend.

## Table of Contents
- [Architecture Overview](#architecture-overview)
- [GitHub Pages Deployment (Frontend)](#github-pages-deployment)
- [Backend Deployment Options](#backend-deployment-options)
- [Local Development Setup](#local-development-setup)
- [Configuration](#configuration)

---

## Architecture Overview

LegalInsight uses a **decoupled architecture**:

```
┌─────────────────────────┐
│   GitHub Pages          │
│   (Static Frontend)     │ ──────┐
│   - HTML/CSS/JS         │       │
│   - User Interface      │       │
└─────────────────────────┘       │ HTTPS/CORS
                                  │
                                  ▼
                        ┌─────────────────────────┐
                        │   Backend API           │
                        │   (Flask Server)        │
                        │   - Self-RAG Model      │
                        │   - EigenScore          │
                        │   - Retrieval System    │
                        └─────────────────────────┘
```

**Frontend**: Deployed on GitHub Pages (free, static hosting)
**Backend**: Deployed on cloud service (Railway, Render, Heroku, or VPS)

---

## GitHub Pages Deployment

### Step 1: Prepare Frontend Files

Your frontend files are already in the `frontend/` directory:
- `index.html` - Main application
- `styles.css` - Styling
- `app.js` - JavaScript logic

### Step 2: Deploy to GitHub Pages

#### Option A: Using Main Branch

1. **Create a `docs` folder in your repository root:**
   ```bash
   mkdir -p docs
   cp frontend/* docs/
   ```

2. **Commit and push:**
   ```bash
   git add docs/
   git commit -m "Add frontend for GitHub Pages"
   git push origin main
   ```

3. **Enable GitHub Pages:**
   - Go to your repository on GitHub
   - Click **Settings** → **Pages**
   - Under **Source**, select:
     - Branch: `main`
     - Folder: `/docs`
   - Click **Save**

4. **Access your site:**
   - GitHub will build your site
   - Access at: `https://kavyasridhar1501.github.io/LegalInsight-SelfRAG-HallucinationDetection/`

#### Option B: Using gh-pages Branch

1. **Create gh-pages branch:**
   ```bash
   git checkout -b gh-pages

   # Copy frontend files to root
   cp frontend/* .

   # Commit
   git add index.html styles.css app.js
   git commit -m "Deploy to GitHub Pages"
   git push origin gh-pages

   # Switch back to main
   git checkout main
   ```

2. **Enable GitHub Pages:**
   - Repository Settings → Pages
   - Source: `gh-pages` branch, `/` (root)

### Step 3: Configure API Endpoint

Once your backend is deployed (see next section), update the frontend:

1. **Edit `app.js` or configure via UI:**
   - The API URL input field in the UI allows runtime configuration
   - Default is `http://localhost:5000`
   - Update to your backend URL: `https://your-backend.railway.app`

2. **Or hardcode in `app.js`:**
   ```javascript
   let API_BASE_URL = 'https://your-backend-url.com';
   ```

---

## Backend Deployment Options

The backend requires:
- Python 3.8+
- 8GB+ RAM (for model inference)
- Storage for the ~4GB model file

### Option 1: Railway (Recommended)

**Pros:** Easy deployment, free tier available, auto-deploys from GitHub
**Cons:** Free tier has limitations

1. **Create `railway.toml`:**
   ```toml
   [build]
   builder = "NIXPACKS"

   [deploy]
   startCommand = "python backend/api.py"
   healthcheckPath = "/health"
   healthcheckTimeout = 300
   restartPolicyType = "ON_FAILURE"
   ```

2. **Create `Procfile`:**
   ```
   web: python backend/api.py
   ```

3. **Deploy:**
   - Sign up at [railway.app](https://railway.app)
   - Click "New Project" → "Deploy from GitHub"
   - Select your repository
   - Add environment variables:
     ```
     SELFRAG_MODEL_PATH=/app/data/models/selfrag-7b-q4_k_m.gguf
     PORT=5000
     ```
   - Railway will auto-deploy on push

4. **Upload Model:**
   - Use Railway CLI or volume mount
   - Or download in startup script

### Option 2: Render

1. **Create `render.yaml`:**
   ```yaml
   services:
     - type: web
       name: legalinsight-api
       env: python
       buildCommand: "pip install -r requirements.txt"
       startCommand: "python backend/api.py"
       envVars:
         - key: SELFRAG_MODEL_PATH
           value: /opt/render/project/data/models/selfrag-7b-q4_k_m.gguf
   ```

2. **Deploy:**
   - Sign up at [render.com](https://render.com)
   - New → Web Service
   - Connect GitHub repository
   - Render auto-deploys from `render.yaml`

### Option 3: Google Cloud Run / AWS Lambda

For serverless deployment (advanced):
- Package model in container
- Use Cloud Storage for model
- Configure cold start timeouts (model loading takes time)

### Option 4: VPS (DigitalOcean, Linode, AWS EC2)

For full control:

1. **Provision VPS:**
   - Ubuntu 22.04
   - 8GB+ RAM
   - 20GB+ storage

2. **Setup:**
   ```bash
   # Clone repository
   git clone https://github.com/kavyasridhar1501/LegalInsight-SelfRAG-HallucinationDetection.git
   cd LegalInsight-SelfRAG-HallucinationDetection

   # Install dependencies
   sudo apt update
   sudo apt install python3-pip python3-venv
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

   # Download model
   python scripts/download_model.py

   # Run backend
   export SELFRAG_MODEL_PATH=/path/to/model
   python backend/api.py
   ```

3. **Production server (Gunicorn):**
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 backend.api:app
   ```

4. **Reverse proxy (Nginx):**
   ```nginx
   server {
       listen 80;
       server_name your-domain.com;

       location / {
           proxy_pass http://localhost:5000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```

5. **HTTPS with Let's Encrypt:**
   ```bash
   sudo apt install certbot python3-certbot-nginx
   sudo certbot --nginx -d your-domain.com
   ```

---

## Local Development Setup

### Prerequisites
- Python 3.8+
- 16GB RAM (recommended)
- 10GB free disk space

### Step 1: Clone Repository
```bash
git clone https://github.com/kavyasridhar1501/LegalInsight-SelfRAG-HallucinationDetection.git
cd LegalInsight-SelfRAG-HallucinationDetection
```

### Step 2: Install Dependencies
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### Step 3: Download Model
```bash
python scripts/download_model.py
```

Or manually download from:
- https://huggingface.co/selfrag/selfrag_llama2_7b

Place in: `data/models/selfrag-7b-q4_k_m.gguf`

### Step 4: Setup Dataset (Optional)
```bash
python scripts/setup_dataset.py
```

### Step 5: Run Backend
```bash
# Set model path
export SELFRAG_MODEL_PATH=data/models/selfrag-7b-q4_k_m.gguf

# Run server
python backend/api.py
```

Backend will be available at: `http://localhost:5000`

### Step 6: Run Frontend
```bash
# Option 1: Simple HTTP server
cd frontend
python -m http.server 8000

# Option 2: Open directly in browser
open index.html  # Mac
start index.html # Windows
xdg-open index.html # Linux
```

Frontend will be available at: `http://localhost:8000`

---

## Configuration

### Environment Variables

Create `.env` file:
```bash
# Model Configuration
SELFRAG_MODEL_PATH=data/models/selfrag-7b-q4_k_m.gguf
MODEL_N_GPU_LAYERS=0  # Set to -1 for GPU acceleration

# API Configuration
PORT=5000
FLASK_ENV=production
CORS_ORIGINS=https://kavyasridhar1501.github.io

# Dataset Configuration
LEGALBENCH_CORPUS_PATH=data/legalbench/corpus
LEGALBENCH_QUERIES_PATH=data/legalbench/queries/legalbench_queries_mini.json
```

### Frontend Configuration

Update `frontend/app.js`:
```javascript
// Production backend URL
let API_BASE_URL = 'https://your-backend-url.com';

// Or allow user configuration via UI (current setup)
```

### CORS Configuration

In `backend/api.py`, update CORS origins:
```python
CORS(app, origins=[
    "https://kavyasridhar1501.github.io",
    "http://localhost:8000",
    "http://127.0.0.1:8000"
])
```

---

## Performance Optimization

### Model Optimization
- Use Q4_K_M quantization (4GB model) for balance of speed/quality
- Use Q2_K quantization (2.8GB model) for faster inference
- Enable GPU acceleration if available: `MODEL_N_GPU_LAYERS=-1`

### Backend Optimization
- Use Gunicorn with multiple workers
- Add caching for repeated queries
- Implement request queuing for high load

### Frontend Optimization
- Minify JavaScript and CSS
- Add loading states and progress indicators
- Implement client-side caching

---

## Monitoring & Analytics

### Backend Health Check
```bash
curl http://your-backend-url/health
```

### Analytics Endpoint
```bash
curl http://your-backend-url/analytics
```

### Logging
Add logging in `backend/api.py`:
```python
import logging
logging.basicConfig(level=logging.INFO)
```

---

## Troubleshooting

### Frontend can't connect to backend
- Check CORS configuration
- Verify backend URL in frontend
- Check backend is running and accessible

### Model loading fails
- Verify model file exists
- Check file permissions
- Ensure enough RAM (16GB recommended)

### Slow inference
- Reduce max_new_tokens
- Use smaller quantization (Q2_K)
- Enable GPU acceleration
- Reduce num_generations for EigenScore

---

## Security Considerations

1. **API Rate Limiting**: Add rate limiting to prevent abuse
2. **Input Validation**: Sanitize contract text input
3. **HTTPS**: Always use HTTPS in production
4. **Authentication**: Add API keys for production use
5. **Data Privacy**: Don't log sensitive contract data

---

## Cost Estimation

### GitHub Pages
- **FREE** for public repositories
- Unlimited bandwidth for reasonable use

### Backend (Railway Free Tier)
- **$0-5/month** for hobby projects
- 500 hours/month free
- Upgrade for production: ~$20-50/month

### Alternative: VPS
- **DigitalOcean**: $12/month (2GB RAM) - too small
- **DigitalOcean**: $24/month (4GB RAM) - minimum
- **Linode**: $36/month (8GB RAM) - recommended

---

## Next Steps

1. ✅ Deploy frontend to GitHub Pages
2. ✅ Deploy backend to cloud service
3. ✅ Configure API URL in frontend
4. ✅ Test end-to-end functionality
5. 📊 Monitor usage and performance
6. 🔧 Optimize based on metrics

---

## Support & Resources

- **Documentation**: See README.md
- **Issues**: GitHub Issues
- **Model**: https://huggingface.co/selfrag/selfrag_llama2_7b
- **Dataset**: https://huggingface.co/datasets/nguha/legalbench

---

**Congratulations!** Your LegalInsight application is now deployed and ready to analyze legal contracts with Self-RAG + EigenScore! 🎉
