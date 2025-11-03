# 🚀 Quick Start - Deploy in 5 Minutes!

## Step 1: Create GitHub Repository (2 minutes)

1. Go to https://github.com/new
2. Fill in:
   - **Repository name:** `f1-win-predictor`
   - **Description:** `F1 Race Win Predictor with 95.3% ML accuracy`
   - **Visibility:** ✅ **Public**
3. Click "Create repository"

## Step 2: Push Your Code (1 minute)

Open Terminal and run these commands:

```bash
cd /Users/anithalakshmipathy/Documents/f1-project/f1-win-predictor

# Add your GitHub repo (REPLACE YOUR_USERNAME!)
git remote add origin https://github.com/YOUR_USERNAME/f1-win-predictor.git

# Push code
git push -u origin main
```

**Replace `YOUR_USERNAME`** with your actual GitHub username!

**Example:**
```bash
git remote add origin https://github.com/harshithk/f1-win-predictor.git
```

## Step 3: Deploy to Streamlit (2 minutes)

1. Go to https://streamlit.io/cloud
2. Click "Sign in with GitHub"
3. Click "New app"
4. Select:
   - **Repository:** `YOUR_USERNAME/f1-win-predictor`
   - **Branch:** `main`
   - **Main file:** `f1_predictor_complete.py`
5. Click "Deploy!"

## 🎉 Done!

Your app will be live at:
```
https://YOUR-USERNAME-f1-win-predictor.streamlit.app
```

---

## ⚠️ Troubleshooting

### "Need authentication for git push?"
1. Go to: https://github.com/settings/tokens
2. Click "Generate new token (classic)"
3. Check ✅ `repo` scope
4. Copy the token
5. Use it as password when pushing

### "Can't find repository on Streamlit?"
- Make sure repository is **PUBLIC** on GitHub
- Refresh the page
- Sign out and sign in again

---

## 📱 Share Your App!

Once live, share your URL:
- LinkedIn
- Twitter/X  
- Portfolio
- Resume

---

**That's it! Your F1 Predictor is now live!** 🏎️✨

