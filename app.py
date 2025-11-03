import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.train import build_model, train_and_predict, FEATURES
from src.evaluate import top1_accuracy, season_summary
from src.bet_backtest import main as bet_backtest

# Page config
st.set_page_config(
    page_title="F1 Race Win Predictor",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for F1 theme
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #1a1a1a 0%, #2d0a0a 100%);
    }
    .stApp {
        background-color: #0e1117;
    }
    h1, h2, h3 {
        color: #e10600 !important;
        font-family: 'Formula1', 'Arial', sans-serif;
    }
    .stMetric {
        background: linear-gradient(135deg, #2d0a0a 0%, #1a1a1a 100%);
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #e10600;
    }
    .stButton>button {
        background: linear-gradient(90deg, #e10600 0%, #ff3333 100%);
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 25px;
        padding: 10px 30px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 20px #e10600;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/3/33/F1.svg", width=200)
    st.title("🏎️ F1 Predictor")
    st.markdown("---")
    
    page = st.radio(
        "Navigation",
        ["🏠 Dashboard", "📊 Data Explorer", "🤖 Train Model", "🎯 Live Predictions", "📈 Analytics", "💰 Betting Simulator"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.caption("Built with ❤️ by Harshith K.")

# Helper functions
@st.cache_data
def load_data():
    """Load training data"""
    try:
        df = pd.read_parquet("data/curated/f1_training.parquet")
        return df
    except FileNotFoundError:
        return None

@st.cache_data
def load_predictions():
    """Load model predictions"""
    try:
        df = pd.read_parquet("data/curated/models/preds.parquet")
        return df
    except FileNotFoundError:
        return None

def get_driver_emoji(driver_id):
    """Get emoji for driver (fallback)"""
    emojis = {
        'verstappen': '🇳🇱',
        'hamilton': '🇬🇧',
        'leclerc': '🇲🇨',
        'norris': '🇬🇧',
        'sainz': '🇪🇸',
        'russell': '🇬🇧',
        'perez': '🇲🇽',
        'alonso': '🇪🇸',
    }
    return emojis.get(driver_id, '🏎️')

# ==================== PAGE 1: DASHBOARD ====================
if page == "🏠 Dashboard":
    st.title("🏎️ F1 Race Win Predictor Dashboard")
    st.markdown("### Welcome to the Ultimate F1 Prediction System!")
    
    # Load data
    df = load_data()
    preds = load_predictions()
    
    if df is None:
        st.error("⚠️ No training data found! Please run the pipeline first.")
        st.code("make data && make features", language="bash")
        st.stop()
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Total Races", f"{len(df)}")
    with col2:
        st.metric("🏁 Seasons", f"{df['season'].nunique()}")
    with col3:
        st.metric("🏎️ Drivers", f"{df['driver_id'].nunique()}")
    with col4:
        st.metric("🏆 Teams", f"{df['constructor_cat'].nunique()}")
    
    st.markdown("---")
    
    # Quick stats
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Recent Season Performance")
        recent_seasons = df.groupby('season')['y_win'].agg(['sum', 'count']).tail(5)
        recent_seasons.columns = ['Wins', 'Total Races']
        st.dataframe(recent_seasons, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Feature Statistics")
        feature_stats = pd.DataFrame({
            'Feature': ['Driver Points', 'Team Points', 'Grid Position', 'Form (5 races)'],
            'Min': [
                df['driver_points_at_stage_of_season'].min(),
                df['constructorId_points_at_stage_of_season'].min(),
                df['grid'].min(),
                df['form_5'].min()
            ],
            'Max': [
                df['driver_points_at_stage_of_season'].max(),
                df['constructorId_points_at_stage_of_season'].max(),
                df['grid'].max(),
                df['form_5'].max()
            ]
        })
        st.dataframe(feature_stats, use_container_width=True)
    
    # Model Status
    st.markdown("---")
    st.subheader("🤖 Model Status")
    
    if preds is not None:
        accuracy = top1_accuracy(preds)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("✅ Model Trained", "Yes", delta="Ready")
        with col2:
            st.metric("🎯 Overall Accuracy", f"{accuracy:.1%}")
        with col3:
            total_preds = (preds['predicted_winner'] == 1).sum()
            st.metric("🏁 Predictions Made", total_preds)
    else:
        st.warning("⚠️ No model predictions found. Train a model first!")
        if st.button("🚀 Go to Model Training"):
            st.rerun()

# ==================== PAGE 2: DATA EXPLORER ====================
elif page == "📊 Data Explorer":
    st.title("📊 F1 Data Explorer")
    
    df = load_data()
    if df is None:
        st.error("⚠️ No data found!")
        st.stop()
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        seasons = st.multiselect("Select Seasons", sorted(df['season'].unique()), default=sorted(df['season'].unique())[-3:])
    with col2:
        drivers = st.multiselect("Select Drivers", sorted(df['driver_id'].unique())[:10])
    with col3:
        min_points = st.slider("Min Driver Points", 0, int(df['driver_points_at_stage_of_season'].max()), 0)
    
    # Filter data
    filtered_df = df[df['season'].isin(seasons)]
    if drivers:
        filtered_df = filtered_df[filtered_df['driver_id'].isin(drivers)]
    filtered_df = filtered_df[filtered_df['driver_points_at_stage_of_season'] >= min_points]
    
    st.markdown(f"### Showing {len(filtered_df)} races")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏆 Wins by Driver")
        wins_by_driver = filtered_df[filtered_df['y_win'] == 1].groupby('driver_id').size().sort_values(ascending=False).head(10)
        fig = px.bar(wins_by_driver, x=wins_by_driver.index, y=wins_by_driver.values,
                     labels={'x': 'Driver', 'y': 'Wins'},
                     color=wins_by_driver.values,
                     color_continuous_scale='Reds')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Grid Position Distribution")
        fig = px.histogram(filtered_df, x='grid', nbins=20,
                          labels={'grid': 'Starting Grid Position'},
                          color_discrete_sequence=['#e10600'])
        st.plotly_chart(fig, use_container_width=True)
    
    # Raw data
    st.markdown("---")
    st.subheader("📋 Raw Data")
    display_cols = ['season', 'round', 'driver_id', 'grid', 'driver_points_at_stage_of_season', 
                    'constructorId_points_at_stage_of_season', 'form_5', 'y_win']
    st.dataframe(filtered_df[display_cols].head(100), use_container_width=True)

# ==================== PAGE 3: TRAIN MODEL ====================
elif page == "🤖 Train Model":
    st.title("🤖 Train F1 Prediction Model")
    
    df = load_data()
    if df is None:
        st.error("⚠️ No training data found!")
        st.stop()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚙️ Model Configuration")
        
        model_type = st.selectbox(
            "Select Model",
            ["xgb", "tuned_xgb", "rf", "gbm", "logreg"],
            format_func=lambda x: {
                "xgb": "🚀 XGBoost (Default)",
                "tuned_xgb": "⭐ Tuned XGBoost (Best)",
                "rf": "🌲 Random Forest",
                "gbm": "📈 Gradient Boosting",
                "logreg": "📊 Logistic Regression"
            }[x]
        )
        
        st.info(f"**Features used:** {len(FEATURES)}")
        with st.expander("View Features"):
            for i, feat in enumerate(FEATURES, 1):
                st.write(f"{i}. `{feat}`")
    
    with col2:
        st.subheader("📊 Training Data Stats")
        st.metric("Total Samples", len(df))
        st.metric("Winners", (df['y_win'] == 1).sum())
        st.metric("Win Rate", f"{(df['y_win'] == 1).mean():.1%}")
        st.metric("Seasons", df['season'].nunique())
    
    st.markdown("---")
    
    if st.button("🚀 Train Model", use_container_width=True):
        with st.spinner("Training model... This may take a minute ⏱️"):
            try:
                # Train
                preds = train_and_predict(df, model_type)
                
                # Save
                Path("data/curated/models").mkdir(parents=True, exist_ok=True)
                preds.to_parquet("data/curated/models/preds.parquet", index=False)
                
                # Evaluate
                accuracy = top1_accuracy(preds)
                season_acc = season_summary(preds)
                
                st.success("✅ Model trained successfully!")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("🎯 Overall Accuracy", f"{accuracy:.1%}")
                with col2:
                    st.metric("📊 Predictions Made", (preds['predicted_winner'] == 1).sum())
                
                st.subheader("📈 Per-Season Accuracy")
                st.dataframe(season_acc, use_container_width=True)
                
                # Chart
                fig = px.bar(season_acc, x='season', y='top1_accuracy',
                            labels={'season': 'Season', 'top1_accuracy': 'Accuracy'},
                            color='top1_accuracy',
                            color_continuous_scale='RdYlGn',
                            range_color=[0, 1])
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Error training model: {str(e)}")

# ==================== PAGE 4: LIVE PREDICTIONS ====================
elif page == "🎯 Live Predictions":
    st.title("🎯 Live Race Predictions")
    
    preds = load_predictions()
    if preds is None:
        st.error("⚠️ No predictions found! Train a model first.")
        st.stop()
    
    # Select race
    races = preds.groupby(['season', 'round', 'race_name']).size().reset_index()[['season', 'round', 'race_name']]
    races['display'] = races['season'].astype(str) + " - Round " + races['round'].astype(str) + ": " + races['race_name']
    
    selected_race = st.selectbox("Select Race", races['display'].tolist())
    
    if selected_race:
        # Parse selection
        season = int(selected_race.split(" - ")[0])
        round_num = int(selected_race.split("Round ")[1].split(":")[0])
        
        # Get race data
        race_data = preds[(preds['season'] == season) & (preds['round'] == round_num)].copy()
        race_data = race_data.sort_values('p_win', ascending=False)
        
        st.markdown(f"## 🏁 {selected_race}")
        st.markdown("---")
        
        # Top 3 predictions
        col1, col2, col3 = st.columns(3)
        
        top3 = race_data.head(3)
        
        with col1:
            driver1 = top3.iloc[0]
            st.markdown(f"### 🥇 1st Place")
            st.markdown(f"### {get_driver_emoji(driver1['driver_id'])} {driver1['driver_id'].upper()}")
            st.metric("Confidence", f"{driver1['p_win']*100:.1f}%")
            st.progress(driver1['p_win'])
        
        with col2:
            driver2 = top3.iloc[1]
            st.markdown(f"### 🥈 2nd Place")
            st.markdown(f"### {get_driver_emoji(driver2['driver_id'])} {driver2['driver_id'].upper()}")
            st.metric("Confidence", f"{driver2['p_win']*100:.1f}%")
            st.progress(driver2['p_win'])
        
        with col3:
            driver3 = top3.iloc[2]
            st.markdown(f"### 🥉 3rd Place")
            st.markdown(f"### {get_driver_emoji(driver3['driver_id'])} {driver3['driver_id'].upper()}")
            st.metric("Confidence", f"{driver3['p_win']*100:.1f}%")
            st.progress(driver3['p_win'])
        
        st.markdown("---")
        
        # Full predictions chart
        st.subheader("📊 All Driver Predictions")
        
        fig = go.Figure(data=[
            go.Bar(
                x=race_data['p_win'] * 100,
                y=race_data['driver_id'],
                orientation='h',
                marker=dict(
                    color=race_data['p_win'] * 100,
                    colorscale='Reds',
                    showscale=True,
                    colorbar=dict(title="Win %")
                ),
                text=race_data['p_win'].apply(lambda x: f"{x*100:.1f}%"),
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            xaxis_title="Win Probability (%)",
            yaxis_title="Driver",
            height=600,
            template='plotly_dark'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Actual result
        st.markdown("---")
        st.subheader("🏆 Actual Result")
        actual_winner = race_data[race_data['y_win'] == 1]
        if len(actual_winner) > 0:
            winner = actual_winner.iloc[0]
            predicted_winner = race_data.iloc[0]
            
            if winner['driver_id'] == predicted_winner['driver_id']:
                st.success(f"✅ **CORRECT!** Model predicted: {predicted_winner['driver_id'].upper()}")
            else:
                st.error(f"❌ **INCORRECT.** Predicted: {predicted_winner['driver_id'].upper()}, Actual: {winner['driver_id'].upper()}")
        else:
            st.info("ℹ️ No winner data available for this race")

# ==================== PAGE 5: ANALYTICS ====================
elif page == "📈 Analytics":
    st.title("📈 F1 Analytics Dashboard")
    
    preds = load_predictions()
    if preds is None:
        st.error("⚠️ No predictions found!")
        st.stop()
    
    # Overall metrics
    st.subheader("📊 Overall Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    accuracy = top1_accuracy(preds)
    total_races = (preds['predicted_winner'] == 1).sum()
    correct_preds = ((preds['predicted_winner'] == 1) & (preds['y_win'] == 1)).sum()
    
    with col1:
        st.metric("🎯 Accuracy", f"{accuracy:.1%}")
    with col2:
        st.metric("✅ Correct", correct_preds)
    with col3:
        st.metric("❌ Incorrect", total_races - correct_preds)
    with col4:
        st.metric("🏁 Total Races", total_races)
    
    st.markdown("---")
    
    # Season-by-season
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Accuracy by Season")
        season_acc = season_summary(preds)
        fig = px.line(season_acc, x='season', y='top1_accuracy',
                     markers=True,
                     labels={'season': 'Season', 'top1_accuracy': 'Accuracy'},
                     color_discrete_sequence=['#e10600'])
        fig.update_traces(line=dict(width=3))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🏆 Prediction Confidence Distribution")
        fig = px.histogram(preds[preds['predicted_winner'] == 1], x='p_win',
                          nbins=20,
                          labels={'p_win': 'Win Probability'},
                          color_discrete_sequence=['#e10600'])
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance (if available)
    st.markdown("---")
    st.subheader("🔍 Feature Importance")
    
    try:
        shap_df = pd.read_csv("data/curated/models/shap_importance.csv")
        fig = px.bar(shap_df.head(10), x='mean_abs_shap', y='feature',
                    orientation='h',
                    labels={'mean_abs_shap': 'Importance (SHAP)', 'feature': 'Feature'},
                    color='mean_abs_shap',
                    color_continuous_scale='Reds')
        st.plotly_chart(fig, use_container_width=True)
    except FileNotFoundError:
        st.info("ℹ️ Run `make interpret` to generate SHAP importance")

# ==================== PAGE 6: BETTING SIMULATOR ====================
elif page == "💰 Betting Simulator":
    st.title("💰 Betting Strategy Simulator")
    
    preds = load_predictions()
    if preds is None:
        st.error("⚠️ No predictions found!")
        st.stop()
    
    st.markdown("### Configure Your Betting Strategy")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        starting_bank = st.number_input("💵 Starting Bankroll ($)", min_value=100, value=1000, step=100)
    with col2:
        stake = st.number_input("🎯 Stake per Bet ($)", min_value=1, value=50, step=10)
    with col3:
        confidence = st.slider("📊 Min Confidence (%)", min_value=50, max_value=95, value=70)
    
    st.markdown("---")
    
    if st.button("🚀 Run Simulation", use_container_width=True):
        # Filter predictions by confidence
        betting_df = preds[preds['predicted_winner'] == 1].copy()
        betting_df = betting_df[betting_df['p_win'] >= confidence/100]
        
        # Simulate betting (even odds for demo)
        betting_df['decimal_odds'] = 2.0
        betting_df['payout'] = betting_df.apply(
            lambda row: (row['decimal_odds'] - 1) * stake if row['y_win'] == 1 else -stake,
            axis=1
        )
        betting_df['cum_pnl'] = betting_df['payout'].cumsum()
        betting_df['bankroll'] = starting_bank + betting_df['cum_pnl']
        
        # Results
        total_bets = len(betting_df)
        wins = (betting_df['y_win'] == 1).sum()
        losses = total_bets - wins
        final_bankroll = starting_bank + betting_df['cum_pnl'].iloc[-1] if len(betting_df) > 0 else starting_bank
        roi = ((final_bankroll - starting_bank) / starting_bank * 100) if starting_bank > 0 else 0
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🎲 Total Bets", total_bets)
        with col2:
            st.metric("✅ Wins", wins, delta=f"{wins/total_bets*100:.1f}%" if total_bets > 0 else "0%")
        with col3:
            st.metric("💰 Final Bankroll", f"${final_bankroll:.2f}", 
                     delta=f"${final_bankroll - starting_bank:.2f}")
        with col4:
            st.metric("📈 ROI", f"{roi:.1f}%", 
                     delta="Profit" if roi > 0 else "Loss")
        
        # Chart
        st.markdown("---")
        st.subheader("📊 Bankroll Over Time")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(betting_df))),
            y=betting_df['bankroll'],
            mode='lines+markers',
            name='Bankroll',
            line=dict(color='#e10600', width=3),
            fill='tozeroy',
            fillcolor='rgba(225, 6, 0, 0.1)'
        ))
        
        fig.add_hline(y=starting_bank, line_dash="dash", 
                     line_color="white", 
                     annotation_text="Starting Bankroll")
        
        fig.update_layout(
            xaxis_title="Bet Number",
            yaxis_title="Bankroll ($)",
            template='plotly_dark',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed results
        st.markdown("---")
        st.subheader("📋 Betting History")
        display_cols = ['season', 'round', 'race_name', 'driver_id', 'p_win', 'y_win', 'payout', 'cum_pnl', 'bankroll']
        st.dataframe(betting_df[display_cols], use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888;'>
    <p>🏎️ F1 Race Win Predictor | Built with Streamlit & ❤️</p>
    <p>Data: 1995-2024 | Models: XGBoost, Random Forest, GBM, Logistic Regression</p>
</div>
""", unsafe_allow_html=True)

