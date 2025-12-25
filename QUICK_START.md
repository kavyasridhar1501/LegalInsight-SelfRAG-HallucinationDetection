# 🚀 LegalInsight - Quick Start Guide

Get your legal contract analysis system up and running in minutes!

## 📋 Prerequisites

- Python 3.8+
- 16GB RAM
- 10GB free disk space
- Git

## ⚡ 5-Minute Setup

### Step 1: Clone and Install (2 minutes)

```bash
# Clone the repository
git clone https://github.com/kavyasridhar1501/LegalInsight-SelfRAG-HallucinationDetection.git
cd LegalInsight-SelfRAG-HallucinationDetection

# Create virtual environment
python -m venv venv

# Activate (choose your OS)
source venv/bin/activate          # Mac/Linux
# OR
venv\Scripts\activate             # Windows

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Download Model (3 minutes)

```bash
python scripts/download_model.py
```

Select option 1 for the recommended Q4_K_M model (~4GB download).

### Step 3: Run the Application

#### Start Backend (Terminal 1)

```bash
export SELFRAG_MODEL_PATH=data/models/selfrag-7b-q4_k_m.gguf  # Mac/Linux
# OR
set SELFRAG_MODEL_PATH=data/models/selfrag-7b-q4_k_m.gguf     # Windows

python backend/api.py
```

Wait for "Server starting on http://localhost:5000"

#### Start Frontend (Terminal 2)

```bash
cd frontend
python -m http.server 8000
```

### Step 4: Use the Application

1. Open your browser to `http://localhost:8000`
2. Click "Load Example" to populate with a sample contract
3. Click "Analyze Contract"
4. See your results with time savings!

## 🌐 Deploy to GitHub Pages

### Frontend Deployment (FREE!)

```bash
# The docs/ folder is already set up
git add docs/
git commit -m "Deploy frontend"
git push origin main

# Then on GitHub:
# Settings → Pages → Source: main branch, /docs folder
```

Your frontend will be live at:
`https://YOUR_USERNAME.github.io/LegalInsight-SelfRAG-HallucinationDetection/`

### Backend Deployment

**Option 1: Railway (Recommended)**

1. Sign up at [railway.app](https://railway.app)
2. "New Project" → "Deploy from GitHub"
3. Select your repository
4. Set environment variable:
   - `SELFRAG_MODEL_PATH=/app/data/models/selfrag-7b-q4_k_m.gguf`
5. Upload your model file or use volume mount
6. Get your backend URL (e.g., `https://your-app.railway.app`)

**Option 2: Render**

1. Sign up at [render.com](https://render.com)
2. "New" → "Web Service"
3. Connect GitHub repository
4. Render will use `render.yaml` automatically
5. Get your backend URL

**Option 3: VPS (DigitalOcean, Linode, etc.)**

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed VPS setup.

### Connect Frontend to Backend

1. Open your GitHub Pages site
2. In the "Configuration" section, update API URL to your backend:
   - Example: `https://your-app.railway.app`
3. Click "Test Connection"
4. Once connected, you're ready to analyze contracts!

## 📊 What You Get

### Time Savings Metrics
- **Manual Analysis Time**: Estimated based on contract length
- **AI Analysis Time**: Actual processing time
- **Time Saved**: How much time you saved
- **Efficiency Improvement**: Percentage faster
- **Speedup Factor**: How many times faster

### EigenScore Hallucination Detection
- **Low Risk** (< -2.0): Highly reliable ✅
- **Medium Risk** (-2.0 to 0): Verify key points ⚠️
- **High Risk** (> 0): Double-check carefully ❌

### Contract Analysis Features
- Automatic summarization
- Key term extraction
- Answer specific questions
- Identify obligations and risks
- Semantic consistency verification

## 🎯 Example Use Cases

1. **Quick Contract Summary**
   - Paste contract → Click "Quick Summary"
   - Get overview in seconds

2. **Specific Questions**
   - Paste contract
   - Ask: "What are the payment terms?"
   - Get precise answer with context

3. **Risk Assessment**
   - Analyze contract
   - Check EigenScore for reliability
   - Review alternative responses

## 📈 Performance Tips

### Speed Optimization
- Use Q2_K model for faster inference (smaller, less accurate)
- Enable GPU: Set `MODEL_N_GPU_LAYERS=-1` in backend/api.py
- Reduce `num_generations` from 5 to 3 for faster EigenScore

### Quality Optimization
- Use Q4_K_M model (default, balanced)
- Keep `num_generations=5` for better hallucination detection
- Use `top_k=3` for retrieval (more context)

## 🔧 Troubleshooting

### Frontend can't connect to backend
- Verify backend is running (`http://localhost:5000/health`)
- Check CORS settings in `backend/api.py`
- Ensure API URL is correct in frontend

### Model loading fails
- Check file exists: `data/models/selfrag-7b-q4_k_m.gguf`
- Verify 16GB+ RAM available
- Check file permissions

### Slow inference
- First request is always slower (model loading)
- Subsequent requests are faster
- Consider GPU acceleration
- Use smaller model variant

## 📚 Next Steps

1. ✅ **Deploy Frontend**: Push to GitHub Pages
2. ✅ **Deploy Backend**: Choose Railway, Render, or VPS
3. ✅ **Connect Both**: Update API URL in frontend
4. 📊 **Analyze Contracts**: Start saving time!
5. 📈 **Monitor Analytics**: Track your time savings

## 🆘 Need Help?

- **Documentation**: See [README.md](README.md)
- **Deployment**: See [DEPLOYMENT.md](DEPLOYMENT.md)
- **Issues**: Open a GitHub issue
- **Questions**: Check existing issues first

## 🎉 You're All Set!

Your LegalInsight system is ready to analyze legal contracts with:
- ✅ Self-RAG + EigenScore accuracy
- ✅ Time tracking metrics
- ✅ GitHub Pages deployment
- ✅ Professional web interface

**Start saving time on contract analysis today!** ⚖️

---

**Pro Tip**: Bookmark your deployed GitHub Pages URL and backend URL for quick access. The frontend saves your API URL in localStorage, so you only need to configure it once!
