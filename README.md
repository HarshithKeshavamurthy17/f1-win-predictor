# 🏎️ F1 Race Win Predictor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

A professional machine learning system that predicts Formula 1 race winners with **95.3% accuracy** using 30 years of historical data (1995-2024).

![F1 Predictor Demo](https://img.shields.io/badge/Python-3.12+-blue.svg)
![ML](https://img.shields.io/badge/ML-Random%20Forest-orange.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-95.3%25-brightgreen.svg)

---

## 🎯 Features

### 🏆 Race Predictions
- **Complete Race Analysis** - View actual vs predicted results side-by-side
- **Top 10 Predictions** - See win probabilities for all drivers
- **Historical Accuracy** - Check model performance on any race from 1995-2024

### 📊 Championship Predictions
- **Season Winners** - Predict Driver and Constructor World Champions
- **Points Breakdown** - Full championship standings with race-by-race analysis
- **Top 10 Drivers** - Complete season rankings

### 🔮 2025 Season Predictor
- **Future Races** - Predict winners for upcoming 2025 races
- **Real 2025 Grid** - Uses actual driver lineups and team assignments
- **Top 10 Finishers** - Full race predictions with confidence scores

---

## 📈 Model Performance

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | 95.3% (530/556 races) |
| **Cross-Validation AUC** | 1.000 (Perfect) |
| **Training Data** | 11,288 race results |
| **Seasons Covered** | 1995-2024 (30 years) |
| **Features Used** | 25 advanced features |

---

## 🧠 How It Works

The model uses **25 research-based features** that capture real F1 racing dynamics:

### Key Predictive Factors:
1. **Grid Position** (46%) - Starting position is crucial
2. **Recent Form** (12%) - Hot streaks matter in F1
3. **Team Performance** (8%) - Dominant teams keep winning
4. **Driver Experience** (6%) - Veterans manage races better
5. **Championship Standing** (5%) - Leaders have momentum
6. **Reliability** (4%) - Finishing races consistently
7. **Qualifying Trends** (3%) - Performance trajectory
8. **Teammate Performance** (3%) - Relative speed indicator

### Machine Learning:
- **Algorithm:** Random Forest (800 trees, depth 25)
- **Validation:** 5-fold GroupKFold cross-validation by season
- **Training:** 30 years of F1 history (1995-2024)
- **Accuracy:** 95.3% winner prediction rate

---

## 🚀 Live Demo

**Try it now:** [F1 Predictor Live App](https://f1-win-predictor-app.streamlit.app)

> 💡 **Monitoring:** This app is monitored 24/7 with [Uptime Robot](https://uptimerobot.com/) to ensure 99.9%+ uptime and keep the app always active. See [UPTIME_ROBOT_SETUP.md](UPTIME_ROBOT_SETUP.md) for setup instructions.

---

## 💻 Local Installation

### Prerequisites
- Python 3.12+
- pip or conda

### Quick Start

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/f1-win-predictor.git
cd f1-win-predictor

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run f1_predictor_complete.py
```

The app will open in your browser at `http://localhost:8501`

---

## 📊 Project Structure

```
f1-win-predictor/
├── f1_predictor_complete.py    # Main Streamlit app
├── src/
│   ├── features_improved.py    # Feature engineering (25 features)
│   ├── train_improved.py       # Model training
│   ├── preprocess.py           # Data preprocessing
│   ├── collect.py              # Data collection
│   └── utils.py                # Helper functions
├── data/
│   ├── raw/                    # Raw F1 data
│   ├── curated/                # Processed data
│   │   ├── f1_training.parquet # Feature-rich training data
│   │   └── models/             # Trained models & predictions
│   └── notebooks/              # Jupyter analysis notebooks
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 🎓 Technical Details

### Features Engineered:
- **Experience**: Career starts, team tenure, team changes
- **Form**: Last 5 races average, recent podiums, reliability score
- **Championship**: Driver/constructor points, gap to leader, title fight status
- **Performance**: Grid trends, teammate comparison, points vs teammate
- **Context**: Race number in season, grid delta from pole
- **Team**: Recent wins, constructor championship standing

### Model Architecture:
- **Type:** Random Forest Classifier
- **Trees:** 800 (for stability)
- **Depth:** 25 (captures complex patterns)
- **Class Balancing:** Weighted (handles imbalanced data)
- **Features:** 25 (research-based F1 factors)
- **Validation:** Season-based cross-validation (5-fold)

---

## 📝 Data Sources

- **Historical Data:** F1 race results 1995-2024
- **Features:** 25 engineered features based on F1 racing research
- **Size:** 11,288 individual race results across 556 races
- **Coverage:** 30 complete F1 seasons

---

## 🛠️ Technologies Used

- **Python 3.12** - Programming language
- **Streamlit** - Web application framework
- **Scikit-learn** - Machine learning models
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Plotly** - Interactive visualizations
- **PyArrow** - Efficient data storage

---

## 📊 Use Cases

### For F1 Fans:
- Predict race winners before qualifying
- Analyze historical "what if" scenarios
- Understand what factors lead to wins
- Compare different eras of F1

### For Data Scientists:
- Learn feature engineering for sports data
- Study time-series prediction in racing
- Explore imbalanced classification techniques
- Understand domain-specific ML applications

### For Betting Analysis:
- Evaluate race outcome probabilities
- Assess prediction confidence levels
- Historical betting performance simulation
- Risk assessment for race predictions

---

## 🎯 Future Enhancements

- [ ] Real-time 2025 season predictions
- [ ] Pit stop strategy predictions
- [ ] Weather impact modeling
- [ ] Tire strategy optimization
- [ ] Safety car probability
- [ ] Overtaking likelihood analysis

---

## 👨‍💻 Author

**Harshith K.**
- Data Science Portfolio Project
- Built with ❤️ for Formula 1

---

## 📄 License

This project is open source and available for educational purposes.

---

## 🙏 Acknowledgments

- **Ergast API** - Historical F1 data source
- **F1 Community** - Domain knowledge and insights
- **Streamlit** - Excellent web app framework

---

## ⭐ Support

If you find this project useful, please give it a star on GitHub!

---

**Built with passion for Formula 1 and Machine Learning** 🏎️💨
