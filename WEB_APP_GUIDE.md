# 🏎️ F1 Race Predictor Web App - Complete Guide

## 🚀 Quick Start (3 Steps)

### Step 1: Install Dependencies
```bash
cd /Users/anithalakshmipathy/Documents/f1-project/f1-win-predictor
conda activate f1-win

# Install Streamlit if not already installed
pip install streamlit plotly
```

### Step 2: Prepare Data (First Time Only)
```bash
# Run the pipeline to prepare data
make data       # Collect F1 data (takes 5-10 mins)
make features   # Process and engineer features
make train      # Train the model
```

### Step 3: Launch the App
```bash
# Option A: Use the launch script
./run_app.sh

# Option B: Run directly
streamlit run app.py

# Option C: Custom port
streamlit run app.py --server.port 8080
```

**The app will open automatically in your browser at `http://localhost:8501`**

---

## 📱 App Pages Overview

### 🏠 Page 1: Dashboard
**What you'll see:**
- Total races, seasons, drivers, teams metrics
- Recent season performance table
- Feature statistics overview
- Model training status
- Overall prediction accuracy

**Use it for:**
- Quick project overview
- Checking model status
- Viewing high-level stats

---

### 📊 Page 2: Data Explorer
**What you'll see:**
- Interactive filters (season, driver, points)
- Wins by driver bar chart
- Grid position distribution histogram
- Raw data table (first 100 rows)

**Use it for:**
- Exploring historical data
- Filtering specific drivers/seasons
- Understanding data distribution
- Exporting filtered data

**Try this:**
1. Select seasons: 2022, 2023, 2024
2. Pick drivers: verstappen, hamilton, leclerc
3. See their performance comparison

---

### 🤖 Page 3: Train Model
**What you'll see:**
- Model selection dropdown (5 models)
- Feature list (13 features used)
- Training data statistics
- Train button
- Accuracy metrics after training
- Per-season accuracy chart

**Available Models:**
- 🚀 XGBoost (Default) - Fast, accurate
- ⭐ Tuned XGBoost (Best) - Grid-search optimized
- 🌲 Random Forest - Ensemble method
- 📈 Gradient Boosting - Slower but accurate
- 📊 Logistic Regression - Baseline model

**How to use:**
1. Select a model
2. Click "🚀 Train Model"
3. Wait 30-60 seconds
4. View results and accuracy

---

### 🎯 Page 4: Live Predictions
**What you'll see:**
- Race selector dropdown
- Top 3 predictions with emojis and confidence
- Full driver ranking bar chart
- Actual result comparison

**How to use:**
1. Select any race from dropdown
2. View top 3 predicted winners with confidence %
3. See full ranking chart
4. Check if prediction was correct

**Example:**
```
Selected: 2024 - Round 5: Miami Grand Prix

🥇 1st Place
🇳🇱 VERSTAPPEN
Confidence: 85%
████████████████████░░

🥈 2nd Place  
🇬🇧 NORRIS
Confidence: 68%
██████████████░░░░░░░

🥉 3rd Place
🇲🇨 LECLERC  
Confidence: 62%
████████████░░░░░░░░
```

---

### 📈 Page 5: Analytics
**What you'll see:**
- Overall accuracy metrics
- Accuracy by season line chart
- Prediction confidence distribution
- SHAP feature importance (if available)

**Use it for:**
- Evaluating model performance
- Finding seasonal trends
- Understanding feature importance
- Identifying model strengths/weaknesses

**Key Insights:**
- Which seasons have highest accuracy?
- What's the typical confidence range?
- Which features matter most?

---

### 💰 Page 6: Betting Simulator
**What you'll see:**
- Configuration inputs (bankroll, stake, confidence)
- Run simulation button
- Results metrics (bets, wins, ROI)
- Bankroll over time chart
- Detailed betting history table

**How to use:**
1. Set starting bankroll: $1000
2. Set stake per bet: $50
3. Set minimum confidence: 70%
4. Click "🚀 Run Simulation"
5. View P&L and ROI

**Example Strategy:**
```
Starting Bankroll: $1000
Stake per Bet: $50
Min Confidence: 75%

Results:
🎲 Total Bets: 25
✅ Wins: 17 (68%)
💰 Final Bankroll: $1,340
📈 ROI: +34%
```

---

## 🎨 Visual Features

### Theme
- **Colors:** F1 Red (#e10600) + Black gradient
- **Font:** Formula1-inspired headers
- **Style:** Dark mode racing theme

### Animations
- Hover effects on buttons
- Smooth transitions
- Progressive loading
- Interactive charts

### Charts (Plotly)
- **Bar charts** - Driver rankings, wins
- **Line charts** - Accuracy trends, bankroll
- **Histograms** - Confidence distribution
- **Horizontal bars** - Feature importance

---

## 🔧 Customization

### Change Theme Colors
Edit `app.py` line 29-50:
```python
.stButton>button {
    background: linear-gradient(90deg, #YOUR_COLOR1 0%, #YOUR_COLOR2 100%);
}
```

### Add New Pages
Add to sidebar (line 56):
```python
page = st.radio(
    "Navigation",
    ["🏠 Dashboard", ..., "🆕 Your New Page"]
)
```

Then add page logic:
```python
elif page == "🆕 Your New Page":
    st.title("Your New Page")
    # Your code here
```

### Modify Metrics
Change displayed metrics in any page section.

---

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'streamlit'"
**Solution:**
```bash
pip install streamlit plotly
```

### Issue: "No data found" errors
**Solution:**
```bash
# Run the data pipeline first
make data
make features
make train
```

### Issue: App won't start
**Solution:**
```bash
# Check if port is in use
lsof -i :8501

# Use different port
streamlit run app.py --server.port 8502
```

### Issue: Slow performance
**Solution:**
- Reduce data range in filters
- Use smaller year range
- Clear Streamlit cache: `Ctrl+C` then restart

### Issue: Charts not loading
**Solution:**
```bash
pip install --upgrade plotly
```

---

## 🚀 Deployment Options

### Option 1: Streamlit Cloud (FREE)
1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo
4. Click "Deploy"
5. Get public URL

### Option 2: Heroku
```bash
# Create Procfile
echo "web: streamlit run app.py --server.port $PORT" > Procfile

# Deploy
heroku create f1-predictor
git push heroku main
```

### Option 3: AWS/GCP
- Use EC2/Compute Engine
- Install dependencies
- Run with `nohup streamlit run app.py &`

---

## 💡 Tips & Tricks

### 1. Keyboard Shortcuts
- `r` - Rerun app
- `c` - Clear cache
- `/` - Search in sidebar

### 2. Performance
- Use `@st.cache_data` for slow functions
- Filter data before plotting
- Limit displayed rows

### 3. Best Practices
- Train model before first use
- Update data periodically
- Test different confidence thresholds in betting
- Compare multiple models

### 4. Advanced Features
- Export predictions to CSV from Analytics page
- Download betting history from Simulator
- Use SHAP values to understand predictions

---

## 📊 Data Flow

```
Raw Data (Ergast API)
    ↓
Preprocessing (src/preprocess.py)
    ↓
Feature Engineering (src/features.py)
    ↓
f1_training.parquet
    ↓
Model Training (src/train.py)
    ↓
preds.parquet
    ↓
Streamlit App (app.py)
    ↓
Interactive Visualizations
```

---

## 🎯 Usage Scenarios

### Scenario 1: Race Weekend Prediction
1. Go to **🎯 Live Predictions**
2. Select current season's next race
3. View top 3 predictions
4. Share with friends!

### Scenario 2: Model Comparison
1. Go to **🤖 Train Model**
2. Train XGBoost → Note accuracy
3. Train Random Forest → Compare
4. Choose best model

### Scenario 3: Historical Analysis
1. Go to **📊 Data Explorer**
2. Filter: Season 2023, Driver: Verstappen
3. Analyze wins and performance
4. Compare with 2024

### Scenario 4: Betting Strategy
1. Go to **💰 Betting Simulator**
2. Test conservative: 80% confidence
3. Test aggressive: 60% confidence
4. Compare ROI results

---

## 🔐 Security Notes

- App runs locally by default
- No authentication required for local use
- Add auth for public deployment
- Don't commit API keys

---

## 📚 Further Reading

- [Streamlit Docs](https://docs.streamlit.io)
- [Plotly Python](https://plotly.com/python/)
- [F1 Ergast API](http://ergast.com/mrd/)

---

## ✨ What's Next?

### Planned Features
- [ ] Live API integration for 2025 races
- [ ] Real-time odds fetching
- [ ] User accounts & saved strategies
- [ ] Mobile app version
- [ ] Email alerts for race predictions
- [ ] Multi-language support

### You Can Add
- Historical driver photos
- Team logos
- Circuit maps
- Weather data integration
- Qualifying results

---

## 🆘 Support

**Issues?** Check:
1. This guide first
2. `README.md` for setup
3. GitHub issues (if public)

**Questions?** Reach out to Harshith K.

---

**Enjoy predicting F1 races! 🏁🏆**

