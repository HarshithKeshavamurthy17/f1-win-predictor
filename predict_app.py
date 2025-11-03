#!/usr/bin/env python3
"""
F1 Race Winner Prediction App
Select any race → Get top 10 predictions
Predict future 2025 season races
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# Page config
st.set_page_config(
    page_title="F1 Race Win Predictor 🏎️",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced F1 theme
st.markdown("""
<style>
    .main {background: linear-gradient(135deg, #0a0a0a 0%, #1a0505 100%);}
    h1 {color: #e10600 !important; font-size: 3.5rem !important; font-weight: 900 !important; text-align: center; text-shadow: 0 0 30px rgba(225, 6, 0, 0.8);}
    h2 {color: #ff3333 !important; font-weight: 700 !important;}
    .stMetric {background: linear-gradient(135deg, #2d0a0a 0%, #1a1a1a 100%); padding: 20px; border-radius: 15px; border: 2px solid #e10600; box-shadow: 0 4px 15px rgba(225, 6, 0, 0.3);}
    .st Button>button {background: linear-gradient(90deg, #e10600 0%, #ff3333 100%); color: white; font-weight: bold; border: none; border-radius: 25px; padding: 15px 40px; font-size: 1.1rem; box-shadow: 0 4px 15px rgba(225, 6, 0, 0.4);}
    .stButton>button:hover {transform: scale(1.05); box-shadow: 0 6px 25px rgba(225, 6, 0, 0.6);}
    .prediction-box {background: linear-gradient(135deg, #1a1a1a 0%, #2d0a0a 100%); border: 3px solid #e10600; border-radius: 20px; padding: 25px; margin: 15px 0; box-shadow: 0 6px 20px rgba(225, 6, 0, 0.4);}
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_predictions():
    try:
        return pd.read_parquet("data/curated/models/preds.parquet")
    except:
        return None

@st.cache_data
def load_training_data():
    try:
        return pd.read_parquet("data/curated/f1_training.parquet")
    except:
        return None

# Header
st.title("🏎️ F1 RACE WIN PREDICTOR")
st.markdown("### Professional ML System | 1995-2024 Data | Top 10 Predictions")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/3/33/F1.svg", width=180)
    st.markdown("## 🏁 Navigation")
    
    mode = st.radio(
        "Select Mode:",
        ["📊 Historical Predictions", "🔮 Future Race Predictor (2025)"],
        help="Choose to analyze past races or predict future 2025 races"
    )
    
    st.markdown("---")
    st.markdown("### 📈 Model Stats")
    preds = load_predictions()
    if preds is not None:
        total_races = preds.groupby(['season', 'round']).ngroups
        correct = ((preds['predicted_winner'] == 1) & (preds['y_win'] == 1)).sum()
        accuracy = correct / total_races if total_races > 0 else 0
        
        st.metric("Races Analyzed", f"{total_races:,}")
        st.metric("Accuracy", f"{accuracy:.1%}")
        st.metric("Data Range", f"{preds['season'].min()}-{preds['season'].max()}")
    
    st.markdown("---")
    st.caption("Built with ❤️ | ML-Powered | Streamlit")

# ==================== MODE 1: HISTORICAL PREDICTIONS ====================
if mode == "📊 Historical Predictions":
    st.markdown("## 📊 Historical Race Predictions")
    st.info("💡 Select any race from 1995-2024 to see the model's Top 10 predictions with win probabilities")
    
    preds = load_predictions()
    
    if preds is None:
        st.error("⚠️ No predictions found! Run: `python3 -m src.train_noxgb --in data/curated/f1_training.parquet --out data/curated/models --model gbm`")
        st.stop()
    
    # Race selector
    st.markdown("### 🏁 Select a Race")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        seasons = sorted(preds['season'].unique(), reverse=True)
        selected_season = st.selectbox("Season", seasons, help="Choose the F1 season")
    
    with col2:
        season_data = preds[preds['season'] == selected_season]
        rounds = sorted(season_data['round'].unique())
        selected_round = st.selectbox("Round", rounds, help="Choose the race round")
    
    with col3:
        race_data = preds[(preds['season'] == selected_season) & (preds['round'] == selected_round)]
        race_name = race_data['race_name'].iloc[0] if len(race_data) > 0 else "Unknown"
        st.metric("Race Name", race_name)
    
    st.markdown("---")
    
    # Get race predictions sorted by probability
    race_predictions = race_data.sort_values('p_win', ascending=False).head(10).reset_index(drop=True)
    
    # Show actual winner
    actual_winner = race_data[race_data['y_win'] == 1]
    if len(actual_winner) > 0:
        actual_winner_name = actual_winner['driver_id'].iloc[0]
        st.markdown(f"### 🏆 Actual Winner: **{actual_winner_name.upper()}**")
    
    # Top 3 Podium Predictions
    st.markdown("### 🥇 Predicted Podium (Top 3)")
    
    medals = ["🥇", "🥈", "🥉"]
    colors = ["#FFD700", "#C0C0C0", "#CD7F32"]
    
    cols = st.columns(3)
    
    for idx, col in enumerate(cols):
        if idx < len(race_predictions):
            driver = race_predictions.iloc[idx]
            with col:
                confidence = driver['p_win'] * 100
                
                st.markdown(f"""
                <div class="prediction-box" style="border-color: {colors[idx]};">
                    <h1 style="text-align: center; font-size: 3rem;">{medals[idx]}</h1>
                    <h2 style="text-align: center; color: {colors[idx]};">{driver['driver_id'].upper()}</h2>
                    <h1 style="text-align: center; color: {colors[idx]}; font-size: 2.5rem;">{confidence:.1f}%</h1>
                    <p style="text-align: center; color: #aaa;">Win Probability</p>
                    <p style="text-align: center; color: #888;">Grid: P{int(driver['grid'])}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show if correct
                if driver['y_win'] == 1:
                    st.success("✅ CORRECT!")
    
    st.markdown("---")
    
    # Top 10 Full List
    st.markdown("### 📊 Top 10 Predictions (Full Field)")
    
    # Create bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=race_predictions['p_win'] * 100,
            y=race_predictions['driver_id'].str.upper(),
            orientation='h',
            marker=dict(
                color=race_predictions['p_win'] * 100,
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title="Win %")
            ),
            text=race_predictions['p_win'].apply(lambda x: f"{x*100:.1f}%"),
            textposition='auto',
            hovertemplate='<b>%{y}</b><br>Win Probability: %{x:.1f}%<br>Grid: P' + race_predictions['grid'].astype(str) + '<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title=f"{selected_season} - Round {selected_round}: {race_name}",
        xaxis_title="Win Probability (%)",
        yaxis_title="Driver",
        height=500,
        template='plotly_dark',
        yaxis={'categoryorder': 'total ascending'}
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed table
    st.markdown("### 📋 Detailed Predictions Table")
    
    display_df = race_predictions[['driver_id', 'grid', 'p_win', 'y_win']].copy()
    display_df.columns = ['Driver', 'Grid Position', 'Win Probability', 'Actually Won']
    display_df['Win Probability'] = display_df['Win Probability'].apply(lambda x: f"{x*100:.1f}%")
    display_df['Actually Won'] = display_df['Actually Won'].apply(lambda x: "✅ YES" if x == 1 else "No")
    display_df['Rank'] = range(1, len(display_df) + 1)
    display_df = display_df[['Rank', 'Driver', 'Grid Position', 'Win Probability', 'Actually Won']]
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Prediction analysis
    st.markdown("---")
    st.markdown("### 🔍 Prediction Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        predicted_winner = race_predictions.iloc[0]
        st.metric("Predicted Winner", predicted_winner['driver_id'].upper())
    
    with col2:
        st.metric("Confidence", f"{predicted_winner['p_win']*100:.1f}%")
    
    with col3:
        was_correct = predicted_winner['y_win'] == 1
        st.metric("Prediction", "✅ Correct" if was_correct else "❌ Incorrect")
    
    with col4:
        top3_correct = (race_predictions.head(3)['y_win'] == 1).any()
        st.metric("In Top 3?", "✅ Yes" if top3_correct else "❌ No")

# ==================== MODE 2: FUTURE RACE PREDICTOR ====================
else:
    st.markdown("## 🔮 Future Race Predictor (2025 Season)")
    st.info("💡 Enter driver and team information for a future 2025 race to get predictions")
    
    st.markdown("### 🎯 Input Race Details")
    
    st.warning("⚠️ **Coming Soon!** Feature under development. For now, use historical predictions to understand model performance.")
    
    # Future feature placeholder
    st.markdown("""
    ### 📝 How Future Prediction Will Work:
    
    1. **Enter Race Details**: Season 2025, Round number, Circuit name
    2. **Add Drivers**: Current F1 grid with their teams
    3. **Input Grid Positions**: Qualifying results (or estimated)
    4. **Add Context**: Driver points, team form, recent performance
    5. **Get Predictions**: Model predicts Top 10 with probabilities
    
    ### 🚀 Required Inputs:
    - Driver lineup (current 20 drivers)
    - Constructor/Team for each driver
    - Grid positions (1-20)
    - Current championship points
    - Team championship points
    - Recent form (last 5 races average)
    
    ### 🎯 Output:
    - Top 10 predicted finishers
    - Win probability for each driver
    - Confidence scores
    - Visual rankings
    """)
    
    # Simple form for demo
    with st.expander("🧪 Test Future Prediction (Demo)"):
        st.markdown("Enter details for a hypothetical 2025 race:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            future_season = st.number_input("Season", min_value=2025, max_value=2030, value=2025)
            future_round = st.number_input("Round", min_value=1, max_value=24, value=1)
        
        with col2:
            future_race_name = st.text_input("Race Name", "Bahrain Grand Prix")
            future_circuit = st.text_input("Circuit", "Bahrain International Circuit")
        
        st.markdown("#### Add Drivers (Top 5 for demo)")
        
        demo_drivers = []
        for i in range(5):
            col1, col2, col3 = st.columns(3)
            with col1:
                driver = st.text_input(f"Driver {i+1}", value=["verstappen", "hamilton", "leclerc", "norris", "sainz"][i] if i < 5 else "", key=f"driver_{i}")
            with col2:
                team = st.selectbox(f"Team {i+1}", ["Red Bull", "Mercedes", "Ferrari", "McLaren", "Aston Martin"], key=f"team_{i}")
            with col3:
                grid = st.number_input(f"Grid {i+1}", 1, 20, i+1, key=f"grid_{i}")
            
            if driver:
                demo_drivers.append({'driver': driver, 'team': team, 'grid': grid})
        
        if st.button("🔮 Predict Future Race", use_container_width=True):
            st.success("🎉 Prediction generated!")
            
            # Show demo prediction
            st.markdown("### 🏆 Predicted Top 5:")
            
            for idx, driver_data in enumerate(demo_drivers):
                # Simple heuristic: pole position has highest chance
                confidence = max(10, 70 - (driver_data['grid'] - 1) * 8)
                
                st.markdown(f"""
                <div class="prediction-box">
                    <h3>#{idx+1} - {driver_data['driver'].upper()} ({driver_data['team']})</h3>
                    <p style="font-size: 1.5rem; color: #e10600;"><b>{confidence}%</b> win probability</p>
                    <p>Starting Grid: P{driver_data['grid']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.info("💡 This is a demo. Full future prediction requires training on current 2025 data.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; padding: 30px;'>
    <h3 style='color: #e10600;'>🏎️ F1 Race Win Predictor - Production System</h3>
    <p><b>30 Years of Data</b> | 1995-2024 | 11,288+ Race Results</p>
    <p><b>Machine Learning Model</b> | Gradient Boosting | 13 Advanced Features</p>
    <p><b>Features:</b> Grid Position, Championship Points, Team Form, Driver Experience, Recent Performance</p>
    <p style='margin-top: 20px;'><b>Built by Harshith K.</b> | Data Science Portfolio Project</p>
</div>
""", unsafe_allow_html=True)

