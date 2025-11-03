#!/bin/bash

# F1 Predictor - GitHub Deployment Script
# Run this to push your code to GitHub

echo "🏎️  F1 RACE WIN PREDICTOR - DEPLOYMENT"
echo "=" 
echo ""

# Step 1: Check if git remote exists
if git remote | grep -q origin; then
    echo "✅ Git remote already configured"
else
    echo "⚠️  Please add your GitHub repository URL:"
    echo ""
    echo "Run this command (replace YOUR_USERNAME):"
    echo "git remote add origin https://github.com/YOUR_USERNAME/f1-win-predictor.git"
    echo ""
    exit 1
fi

# Step 2: Check current branch
BRANCH=$(git branch --show-current)
echo "📍 Current branch: $BRANCH"

# Step 3: Push to GitHub
echo ""
echo "🚀 Pushing to GitHub..."
echo ""

git push -u origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "=" 
    echo "✅ SUCCESS! Code pushed to GitHub!"
    echo "=" 
    echo ""
    echo "🌐 NEXT STEPS:"
    echo ""
    echo "1. Go to: https://streamlit.io/cloud"
    echo "2. Click 'New app'"
    echo "3. Select your repository: f1-win-predictor"
    echo "4. Main file: f1_predictor_complete.py"
    echo "5. Click 'Deploy!'"
    echo ""
    echo "📱 Your app will be live in 2-5 minutes at:"
    echo "   https://YOUR-USERNAME-f1-win-predictor.streamlit.app"
    echo ""
    echo "=" 
else
    echo ""
    echo "❌ Push failed. Common issues:"
    echo ""
    echo "1. Need authentication? Create Personal Access Token:"
    echo "   https://github.com/settings/tokens"
    echo ""
    echo "2. No remote configured? Run:"
    echo "   git remote add origin https://github.com/YOUR_USERNAME/f1-win-predictor.git"
    echo ""
fi

