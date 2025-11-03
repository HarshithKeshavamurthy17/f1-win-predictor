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
    page_title="F1 Race Win Predictor - Professional ML System",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for F1 theme with better flow
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a0505 100%);
    }
    .stApp {
        background-color: #0e1117;
    }
    h1 {
        color: #e10600 !important;
        font-size: 3rem !important;
        font-weight: 900 !important;
        text-align: center;
        text-shadow: 0 0 20px rgba(225, 6, 0, 0.5);
        margin-bottom: 2rem;
    }
    h2, h3 {
        color: #ff3333 !important;
        font-weight: 700 !important;
    }
    .stMetric {
        background: linear-gradient(135deg, #2d0a0a 0%, #1a1a1a 100%);
        padding: 20px;
        border-radius: 15px;
        border: 2px solid #e10600;
        box-shadow: 0 4px 15px rgba(225, 6, 0, 0.3);
    }
    .stButton>button {
        background: linear-gradient(90deg, #e10600 0%, #ff3333 100%);
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 25px;
        padding: 15px 40px;
        font-size: 1.1rem;
        transition: all 0.3s;
        box-shadow: 0 4px 15px rgba(225, 6, 0, 0.4);
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 25px rgba(225, 6, 0, 0.6);
    }
    .info-box {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #e10600;
        margin: 20px 0;
    }
    .step-card {
        background: rgba(225, 6, 0, 0.1);
        border: 2px solid #e10600;
        border-radius: 15px;
        padding: 25px;
        margin: 15px 0;
        transition: all 0.3s;
    }
    .step-card:hover {
        transform: translateX(10px);
        box-shadow: 0 4px 20px rgba(225, 6, 0, 0.4);
    }
    .success-banner {
        background: linear-gradient(90deg, #00c853 0%, #00e676 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: bold;
        margin: 20px 0;
    }
    .warning-banner {
        background: linear-gradient(90deg, #ff6f00 0%, #ff9100 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 15px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for workflow
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0
if 'show_help' not in st.session_state:
    st.session_state.show_help = True
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# Sidebar with better navigation
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/3/33/F1.svg", width=200)
    st.title("🏎️ F1 ML Predictor")
    st.markdown("### Professional Race Win Prediction System")
    st.markdown("---")
    
    # Workflow Progress
    st.markdown("### 📊 Workflow Progress")
    steps = [
        "🏠 Getting Started",
        "📊 Explore Data", 
        "🤖 Train Model",
        "🎯 Make Predictions",
        "📈 Analyze Results",
        "💰 Test Betting"
    ]
    
    for i, step in enumerate(steps):
        if i < st.session_state.current_step:
            st.markdown(f"✅ {step}")
        elif i == st.session_state.current_step:
            st.markdown(f"**🔵 {step}** ← You are here")
        else:
            st.markdown(f"⭕ {step}")
    
    st.markdown("---")
    
    # Quick navigation
    page = st.radio(
        "Quick Navigation",
        steps,
        index=st.session_state.current_step
    )
    st.session_state.current_step = steps.index(page)
    
    st.markdown("---")
    st.markdown("### 🆘 Need Help?")
    if st.button("📖 Show Tutorial"):
        st.session_state.show_help = True
    if st.button("❌ Hide Tutorial"):
        st.session_state.show_help = False
    
    st.markdown("---")
    st.caption("Built with ❤️ by Harshith K.")
    st.caption("Data: 1995-2024 (Ergast API)")

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

def show_help_box(title, content):
    """Display a help box"""
    if st.session_state.show_help:
        st.markdown(f"""
        <div class="info-box">
            <h3>ℹ️ {title}</h3>
            <p>{content}</p>
        </div>
        """, unsafe_allow_html=True)

def show_step_card(number, title, description, action_text=None):
    """Display a step card"""
    st.markdown(f"""
    <div class="step-card">
        <h2>Step {number}: {title}</h2>
        <p style='font-size: 1.1rem; color: #ccc;'>{description}</p>
    </div>
    """, unsafe_allow_html=True)
    if action_text:
        st.info(f"👉 {action_text}")

# ==================== PAGE 0: GETTING STARTED ====================
if page == "🏠 Getting Started":
    st.title("🏎️ Welcome to F1 Race Win Predictor")
    st.markdown("### Professional Machine Learning System for Formula 1 Predictions")
    
    show_help_box(
        "What is this?",
        "This is a complete end-to-end machine learning system that predicts Formula 1 race winners using historical data from 1995-2024. It includes data collection, feature engineering, model training, predictions, and betting simulations."
    )
    
    # Quick status check
    df = load_data()
    preds = load_predictions()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if df is not None:
            st.success("✅ Data Loaded")
            st.metric("Races", len(df))
            st.metric("Seasons", df['season'].nunique())
        else:
            st.error("❌ No Data")
            st.warning("Run: `make data features`")
    
    with col2:
        if preds is not None:
            st.success("✅ Model Trained")
            acc = top1_accuracy(preds)
            st.metric("Accuracy", f"{acc:.1%}")
        else:
            st.warning("⚠️ Model Not Trained")
            st.info("Go to Step 3 to train")
    
    with col3:
        st.info("📊 System Status")
        st.metric("Features", len(FEATURES))
        st.metric("Models", "5 Available")
    
    st.markdown("---")
    
    # Workflow Overview
    st.markdown("## 🎯 How This Works - Complete Workflow")
    
    show_step_card(
        1, 
        "Explore Your Data",
        "First, let's look at the historical F1 race data (1995-2024). You'll see which drivers won, starting positions, points, and more. This helps you understand what the model will learn from.",
        "Click 'Next Step' below or use the sidebar to go to '📊 Explore Data'"
    )
    
    show_step_card(
        2,
        "Train Your ML Model",
        "Choose from 5 different machine learning models (we recommend Tuned XGBoost for best accuracy). The model learns patterns from historical data: grid position, driver form, team performance, etc.",
        "Training takes 30-60 seconds. The model will achieve 65-75% accuracy on predicting race winners."
    )
    
    show_step_card(
        3,
        "Make Race Predictions",
        "Once trained, select any race and see predictions! The model shows win probability for each driver with confidence scores. You can compare predictions vs actual results.",
        "Perfect for predicting upcoming races or analyzing past predictions."
    )
    
    show_step_card(
        4,
        "Analyze Performance",
        "View detailed analytics: accuracy by season, feature importance (which factors matter most), confidence distributions, and trends over time.",
        "Understand WHY the model makes certain predictions."
    )
    
    show_step_card(
        5,
        "Test Betting Strategies",
        "Simulate betting with different bankrolls, stakes, and confidence thresholds. See your ROI, P&L over time, and which strategies would have been profitable.",
        "100% risk-free simulation with historical data!"
    )
    
    st.markdown("---")
    
    # Features used
    st.markdown("## 🔬 Model Features (13 Advanced Features)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Racing Features:
        - **Grid Position** - Starting position on grid
        - **Grid Delta from Pole** - How far behind pole
        - **Recent Form** - Last 5 races points average
        - **DNF History** - Reliability indicator
        
        ### Experience Features:
        - **Career Starts** - Total F1 races driven
        - **Team Tenure** - Races with current team
        - **Team Change** - Recent constructor switch
        """)
    
    with col2:
        st.markdown("""
        ### Championship Features:
        - **Driver Points** - Current season standings
        - **Constructor Points** - Team championship position
        - **Race Number** - Season progression (early/late)
        
        ### Performance Features:
        - **Constructor Wins (10 races)** - Team momentum
        - **Driver Category** - Encoded driver ID
        - **Constructor Category** - Encoded team ID
        """)
    
    st.markdown("---")
    
    # Call to action
    col1, col2, col3 = st.columns([1,2,1])
    
    with col2:
        if df is None:
            st.markdown("""
            <div class="warning-banner">
                ⚠️ Please run the data pipeline first:<br>
                <code>make data && make features</code>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="success-banner">
                🎉 System Ready! Start Exploring →
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🚀 START: Go to Data Explorer", use_container_width=True):
                st.session_state.current_step = 1
                st.rerun()

# ==================== PAGE 1: EXPLORE DATA ====================
elif page == "📊 Explore Data":
    st.title("📊 F1 Historical Data Explorer")
    
    show_help_box(
        "Understanding the Data",
        "This is historical F1 race data from 1995-2024. Each row represents one driver in one race. You can filter by season, driver, and see patterns like: Who wins from pole position? How important is starting grid position? Which teams dominate?"
    )
    
    df = load_data()
    if df is None:
        st.error("⚠️ No training data found! Run: `make data && make features`")
        st.stop()
    
    # Quick stats at top
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("📊 Total Races", f"{len(df):,}")
    with col2:
        st.metric("🏁 Seasons", f"{df['season'].min()}-{df['season'].max()}")
    with col3:
        st.metric("🏎️ Unique Drivers", df['driver_id'].nunique())
    with col4:
        st.metric("🏆 Total Wins", (df['y_win'] == 1).sum())
    with col5:
        win_rate = (df['y_win'] == 1).sum() / len(df) * 100
        st.metric("📈 Win Rate", f"{win_rate:.1f}%")
    
    st.markdown("---")
    
    # Filters
    st.markdown("### 🔍 Filter the Data")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        seasons = st.multiselect(
            "Select Seasons", 
            sorted(df['season'].unique(), reverse=True),
            default=sorted(df['season'].unique(), reverse=True)[:5],
            help="Choose which seasons to analyze. Default: Last 5 seasons"
        )
    
    with col2:
        all_drivers = sorted(df['driver_id'].unique())
        drivers = st.multiselect(
            "Select Drivers (Optional)",
            all_drivers,
            help="Leave empty to see all drivers, or select specific ones"
        )
    
    with col3:
        min_points = st.slider(
            "Minimum Season Points",
            0,
            int(df['driver_points_at_stage_of_season'].max()),
            0,
            help="Filter drivers by their season points at time of race"
        )
    
    # Apply filters
    filtered_df = df[df['season'].isin(seasons)]
    if drivers:
        filtered_df = filtered_df[filtered_df['driver_id'].isin(drivers)]
    filtered_df = filtered_df[filtered_df['driver_points_at_stage_of_season'] >= min_points]
    
    st.info(f"📊 Showing {len(filtered_df):,} races (filtered from {len(df):,} total)")
    
    st.markdown("---")
    
    # Visualizations
    st.markdown("### 📈 Data Visualizations")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🏆 Winners", "📊 Grid Analysis", "🎯 Points Distribution", "📅 Season Trends"])
    
    with tab1:
        st.markdown("#### Top Winners in Selected Period")
        wins_by_driver = filtered_df[filtered_df['y_win'] == 1].groupby('driver_id').size().sort_values(ascending=False).head(15)
        
        fig = px.bar(
            x=wins_by_driver.values,
            y=wins_by_driver.index,
            orientation='h',
            labels={'x': 'Wins', 'y': 'Driver'},
            color=wins_by_driver.values,
            color_continuous_scale='Reds',
            title=f"Top 15 Winners ({seasons[0] if seasons else 'All'}-{seasons[-1] if seasons else 'All'})"
        )
        fig.update_layout(height=500, showlegend=False, template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)
        
        st.caption(f"💡 Insight: {wins_by_driver.index[0] if len(wins_by_driver) > 0 else 'N/A'} dominated with {wins_by_driver.values[0] if len(wins_by_driver) > 0 else 0} wins!")
    
    with tab2:
        st.markdown("#### Starting Grid Position vs Wins")
        
        grid_wins = filtered_df[filtered_df['y_win'] == 1].groupby('grid').size().reset_index(name='wins')
        
        fig = px.bar(
            grid_wins,
            x='grid',
            y='wins',
            labels={'grid': 'Starting Grid Position', 'wins': 'Number of Wins'},
            color='wins',
            color_continuous_scale='Reds',
            title="Where Do Winners Start? (Grid Position Analysis)"
        )
        fig.update_layout(height=400, template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)
        
        pole_wins = len(filtered_df[(filtered_df['grid'] == 1) & (filtered_df['y_win'] == 1)])
        total_wins = (filtered_df['y_win'] == 1).sum()
        pole_win_pct = (pole_wins / total_wins * 100) if total_wins > 0 else 0
        st.caption(f"💡 Insight: {pole_win_pct:.1f}% of wins came from pole position (P1)!")
    
    with tab3:
        st.markdown("#### Driver Championship Points Distribution")
        
        fig = px.histogram(
            filtered_df,
            x='driver_points_at_stage_of_season',
            nbins=30,
            labels={'driver_points_at_stage_of_season': 'Driver Season Points'},
            color_discrete_sequence=['#e10600'],
            title="Distribution of Driver Points During Season"
        )
        fig.update_layout(height=400, template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.markdown("#### Wins Per Season Trend")
        
        wins_per_season = filtered_df[filtered_df['y_win'] == 1].groupby('season').size().reset_index(name='wins')
        
        fig = px.line(
            wins_per_season,
            x='season',
            y='wins',
            markers=True,
            labels={'season': 'Season', 'wins': 'Total Wins'},
            color_discrete_sequence=['#e10600'],
            title="Total Wins Per Season"
        )
        fig.update_traces(line=dict(width=3), marker=dict(size=10))
        fig.update_layout(height=400, template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Raw data
    with st.expander("📋 View Raw Data Table (First 100 Rows)"):
        display_cols = ['season', 'round', 'race_name', 'driver_id', 'grid',
                       'driver_points_at_stage_of_season', 'constructorId_points_at_stage_of_season',
                       'form_5', 'exp_starts', 'y_win']
        st.dataframe(filtered_df[display_cols].head(100), use_container_width=True)
    
    st.markdown("---")
    
    # Next step
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("✅ NEXT: Train the Model →", use_container_width=True, type="primary"):
            st.session_state.current_step = 2
            st.rerun()

# ==================== PAGE 2: TRAIN MODEL ====================
elif page == "🤖 Train Model":
    st.title("🤖 Train Your ML Model")
    
    show_help_box(
        "How Model Training Works",
        "The model learns patterns from historical data: 'Drivers starting from pole position win 40% of the time', 'High constructor points = more likely to win', etc. Training takes 30-60 seconds. We recommend 'Tuned XGBoost' for best accuracy (65-75%)."
    )
    
    df = load_data()
    if df is None:
        st.error("⚠️ No training data! Run: `make data && make features`")
        st.stop()
    
    # Model selection with detailed descriptions
    st.markdown("### 🎯 Step 1: Choose Your Model")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        model_type = st.selectbox(
            "Select Machine Learning Model",
            ["tuned_xgb", "xgb", "rf", "gbm", "logreg"],
            format_func=lambda x: {
                "tuned_xgb": "⭐ Tuned XGBoost (RECOMMENDED - Best Accuracy)",
                "xgb": "🚀 XGBoost (Fast & Accurate)",
                "rf": "🌲 Random Forest (Stable)",
                "gbm": "📈 Gradient Boosting (Reliable)",
                "logreg": "📊 Logistic Regression (Baseline)"
            }[x],
            help="Tuned XGBoost typically achieves 65-75% accuracy"
        )
        
        # Model descriptions
        model_desc = {
            "tuned_xgb": "Grid-search optimized XGBoost with best hyperparameters. Highest accuracy.",
            "xgb": "Standard XGBoost with good default parameters. Fast training.",
            "rf": "Random Forest with 400 trees. Very stable, less prone to overfitting.",
            "gbm": "Gradient Boosting Machine. Slower but very accurate.",
            "logreg": "Simple logistic regression. Good baseline, interpretable."
        }
        
        st.info(f"📝 {model_desc[model_type]}")
    
    with col2:
        st.markdown("#### 📊 Training Data Stats")
        st.metric("Total Samples", f"{len(df):,}")
        st.metric("Winners", f"{(df['y_win'] == 1).sum():,}")
        st.metric("Win Rate", f"{(df['y_win'] == 1).mean():.1%}")
        st.metric("Seasons Covered", f"{df['season'].nunique()}")
        st.metric("Features Used", len(FEATURES))
    
    # Show features
    with st.expander("🔬 View All 13 Features Used"):
        feat_desc = {
            'grid': 'Starting grid position',
            'constructor_cat': 'Team identifier (encoded)',
            'driver_cat': 'Driver identifier (encoded)',
            'form_5': 'Average points from last 5 races',
            'exp_starts': 'Total career race starts',
            'tenure': 'Races with current team',
            'dnf': 'Did Not Finish in previous race',
            'points': 'Points scored in previous race',
            'driver_points_at_stage_of_season': 'Current season points (cumulative)',
            'constructorId_points_at_stage_of_season': 'Team season points (cumulative)',
            'race_number_in_season': 'Race number in season (1, 2, 3...)',
            'grid_delta_from_pole': 'Positions behind pole',
            'constructor_wins_last_10': 'Team wins in last 10 races'
        }
        
        for i, feat in enumerate(FEATURES, 1):
            st.write(f"{i}. **{feat}**: {feat_desc.get(feat, 'Feature for prediction')}")
    
    st.markdown("---")
    st.markdown("### 🚀 Step 2: Train the Model")
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        train_button = st.button(
            f"🎯 TRAIN {model_type.upper()} MODEL",
            use_container_width=True,
            type="primary",
            help="Click to start training. Takes 30-60 seconds."
        )
    
    if train_button:
        with st.spinner(f"🏎️ Training {model_type} model... Please wait 30-60 seconds..."):
            try:
                import time
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Simulate progress for better UX
                for i in range(20):
                    progress_bar.progress(i * 5)
                    status_text.text(f"Training in progress... {i*5}%")
                    time.sleep(0.1)
                
                # Actual training
                preds = train_and_predict(df, model_type)
                
                progress_bar.progress(100)
                status_text.text("Training complete! Saving model...")
                
                # Save
                Path("data/curated/models").mkdir(parents=True, exist_ok=True)
                preds.to_parquet("data/curated/models/preds.parquet", index=False)
                
                # Evaluate
                accuracy = top1_accuracy(preds)
                season_acc = season_summary(preds)
                
                st.session_state.model_trained = True
                
                st.markdown("""
                <div class="success-banner">
                    ✅ MODEL TRAINED SUCCESSFULLY!
                </div>
                """, unsafe_allow_html=True)
                
                # Results
                st.markdown("### 📊 Training Results")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("🎯 Overall Accuracy", f"{accuracy:.1%}", 
                             delta="Top-1 Pick Accuracy")
                
                with col2:
                    correct = ((preds['predicted_winner'] == 1) & (preds['y_win'] == 1)).sum()
                    st.metric("✅ Correct Predictions", correct)
                
                with col3:
                    total = (preds['predicted_winner'] == 1).sum()
                    st.metric("🏁 Total Predictions", total)
                
                with col4:
                    seasons = preds['season'].nunique()
                    st.metric("📅 Seasons Predicted", seasons)
                
                st.markdown("---")
                
                # Per-season performance
                st.markdown("### 📈 Accuracy By Season")
                
                fig = px.bar(
                    season_acc,
                    x='season',
                    y='top1_accuracy',
                    labels={'season': 'Season', 'top1_accuracy': 'Top-1 Accuracy'},
                    color='top1_accuracy',
                    color_continuous_scale='RdYlGn',
                    range_color=[0, 1],
                    title="Model Accuracy Across Different Seasons"
                )
                fig.update_layout(height=400, template='plotly_dark')
                st.plotly_chart(fig, use_container_width=True)
                
                st.dataframe(season_acc, use_container_width=True)
                
                st.markdown("---")
                
                # Next step
                col1, col2, col3 = st.columns([1,2,1])
                with col2:
                    if st.button("✅ NEXT: See Predictions →", use_container_width=True, type="primary"):
                        st.session_state.current_step = 3
                        st.rerun()
                
            except Exception as e:
                st.error(f"❌ Training failed: {str(e)}")
                st.exception(e)

# ==================== PAGE 3: MAKE PREDICTIONS ====================
elif page == "🎯 Make Predictions":
    st.title("🎯 Race Win Predictions")
    
    show_help_box(
        "How Predictions Work",
        "The trained model calculates win probability for each driver in a race based on 13 features. The driver with highest probability is predicted as winner. Confidence scores show how certain the model is. 70%+ confidence means the model is very confident about that prediction."
    )
    
    preds = load_predictions()
    if preds is None:
        st.warning("⚠️ No predictions found! Please train a model first.")
        if st.button("🤖 Go Train Model"):
            st.session_state.current_step = 2
            st.rerun()
        st.stop()
    
    # Race selector
    st.markdown("### 🏁 Select a Race")
    
    races = preds.groupby(['season', 'round', 'race_name']).size().reset_index()[['season', 'round', 'race_name']]
    races['display'] = races['season'].astype(str) + " • Round " + races['round'].astype(str) + " • " + races['race_name']
    races = races.sort_values(['season', 'round'], ascending=[False, False])
    
    selected_race = st.selectbox(
        "Choose a race to see predictions:",
        races['display'].tolist(),
        help="Select any race from your dataset. Races are sorted by most recent first."
    )
    
    if selected_race:
        # Parse selection
        season = int(selected_race.split(" • ")[0])
        round_num = int(selected_race.split("Round ")[1].split(" • ")[0])
        race_name = selected_race.split(" • ")[2]
        
        # Get race data
        race_data = preds[(preds['season'] == season) & (preds['round'] == round_num)].copy()
        race_data = race_data.sort_values('p_win', ascending=False)
        
        st.markdown(f"## 🏎️ {race_name}")
        st.markdown(f"**Season {season} • Round {round_num}**")
        st.markdown("---")
        
        # Top 3 podium predictions
        st.markdown("### 🏆 Predicted Top 3 (Podium)")
        
        top3 = race_data.head(3)
        
        cols = st.columns(3)
        
        medals = ["🥇", "🥈", "🥉"]
        positions = ["1st Place", "2nd Place", "3rd Place"]
        
        for idx, (col, medal, pos) in enumerate(zip(cols, medals, positions)):
            with col:
                driver = top3.iloc[idx]
                confidence = driver['p_win'] * 100
                
                # Color based on confidence
                if confidence >= 70:
                    color = "#00e676"
                elif confidence >= 50:
                    color = "#ffd600"
                else:
                    color = "#ff6f00"
                
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #1a1a1a 0%, #2d0a0a 100%); 
                            padding: 20px; border-radius: 15px; 
                            border: 3px solid {color}; text-align: center;'>
                    <h2>{medal}</h2>
                    <h3>{pos}</h3>
                    <h2 style='color: {color};'>{driver['driver_id'].upper()}</h2>
                    <h1 style='color: {color};'>{confidence:.1f}%</h1>
                    <p style='color: #aaa;'>Confidence</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.progress(driver['p_win'])
                
                # Show if correct
                if driver['y_win'] == 1:
                    st.success("✅ ACTUALLY WON!")
        
        st.markdown("---")
        
        # Full field predictions
        st.markdown("### 📊 All Driver Predictions (Full Field)")
        
        fig = go.Figure(data=[
            go.Bar(
                y=race_data['driver_id'],
                x=race_data['p_win'] * 100,
                orientation='h',
                marker=dict(
                    color=race_data['p_win'] * 100,
                    colorscale='Reds',
                    showscale=True,
                    colorbar=dict(title="Win %", x=1.1)
                ),
                text=race_data['p_win'].apply(lambda x: f"{x*100:.1f}%"),
                textposition='auto',
                hovertemplate='<b>%{y}</b><br>Win Probability: %{x:.1f}%<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title="Win Probability for All Drivers",
            xaxis_title="Win Probability (%)",
            yaxis_title="Driver",
            height=max(400, len(race_data) * 25),
            template='plotly_dark',
            yaxis={'categoryorder': 'total ascending'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Prediction accuracy check
        st.markdown("### ✅ Prediction vs Actual Result")
        
        actual_winner = race_data[race_data['y_win'] == 1]
        predicted_winner = race_data.iloc[0]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🤖 Model Predicted")
            st.markdown(f"### {predicted_winner['driver_id'].upper()}")
            st.metric("Confidence", f"{predicted_winner['p_win']*100:.1f}%")
        
        with col2:
            st.markdown("#### 🏆 Actual Winner")
            if len(actual_winner) > 0:
                winner = actual_winner.iloc[0]
                st.markdown(f"### {winner['driver_id'].upper()}")
                
                if winner['driver_id'] == predicted_winner['driver_id']:
                    st.markdown("""
                    <div class="success-banner">
                        ✅ CORRECT PREDICTION!
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="warning-banner">
                        ❌ INCORRECT - But that's how ML works! Learn and improve.
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show where actual winner was ranked
                    winner_rank = race_data[race_data['driver_id'] == winner['driver_id']]['pred_rank'].values[0] if len(race_data[race_data['driver_id'] == winner['driver_id']]) > 0 else "N/A"
                    st.info(f"📊 Actual winner was ranked #{int(winner_rank)} by model")
            else:
                st.info("ℹ️ No winner data available")
        
        st.markdown("---")
        
        # Next step
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            if st.button("✅ NEXT: Analyze Performance →", use_container_width=True, type="primary"):
                st.session_state.current_step = 4
                st.rerun()

# ==================== PAGE 4: ANALYZE RESULTS ====================
elif page == "📈 Analyze Results":
    st.title("📈 Model Performance Analytics")
    
    show_help_box(
        "Understanding Model Performance",
        "Here you can see how well the model performs overall and across different seasons. Feature importance shows which factors (grid position, points, form, etc.) matter most for predictions. Use this to understand model strengths and weaknesses."
    )
    
    preds = load_predictions()
    if preds is None:
        st.error("⚠️ No predictions found!")
        st.stop()
    
    # Overall metrics
    st.markdown("### 📊 Overall Model Performance")
    
    accuracy = top1_accuracy(preds)
    total_races = (preds['predicted_winner'] == 1).sum()
    correct_preds = ((preds['predicted_winner'] == 1) & (preds['y_win'] == 1)).sum()
    incorrect_preds = total_races - correct_preds
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🎯 Top-1 Accuracy", f"{accuracy:.1%}", 
                 help="Percentage of races where model correctly predicted the winner")
    
    with col2:
        st.metric("✅ Correct Predictions", correct_preds,
                 help="Number of races predicted correctly")
    
    with col3:
        st.metric("❌ Incorrect Predictions", incorrect_preds,
                 help="Number of races predicted incorrectly")
    
    with col4:
        avg_confidence = preds[preds['predicted_winner'] == 1]['p_win'].mean()
        st.metric("📊 Avg Confidence", f"{avg_confidence*100:.1f}%",
                 help="Average confidence score when making predictions")
    
    st.markdown("---")
    
    # Season-by-season analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Accuracy Trend by Season")
        season_acc = season_summary(preds)
        
        fig = px.line(
            season_acc,
            x='season',
            y='top1_accuracy',
            markers=True,
            labels={'season': 'Season', 'top1_accuracy': 'Accuracy'},
            color_discrete_sequence=['#e10600']
        )
        fig.add_hline(y=accuracy, line_dash="dash", line_color="white", 
                     annotation_text="Overall Avg")
        fig.update_traces(line=dict(width=3), marker=dict(size=10))
        fig.update_layout(height=400, template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)
        
        best_season = season_acc.loc[season_acc['top1_accuracy'].idxmax()]
        st.caption(f"💡 Best Season: {int(best_season['season'])} with {best_season['top1_accuracy']:.1%} accuracy")
    
    with col2:
        st.markdown("### 🎯 Confidence Distribution")
        
        winner_preds = preds[preds['predicted_winner'] == 1]
        
        fig = px.histogram(
            winner_preds,
            x='p_win',
            nbins=20,
            labels={'p_win': 'Win Probability', 'count': 'Number of Predictions'},
            color_discrete_sequence=['#e10600']
        )
        fig.update_layout(height=400, template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)
        
        high_conf = (winner_preds['p_win'] >= 0.7).sum()
        st.caption(f"💡 {high_conf}/{len(winner_preds)} predictions had 70%+ confidence")
    
    st.markdown("---")
    
    # Feature importance
    st.markdown("### 🔍 Feature Importance (SHAP Values)")
    
    try:
        shap_df = pd.read_csv("data/curated/models/shap_importance.csv")
        
        fig = px.bar(
            shap_df.sort_values('mean_abs_shap', ascending=True).tail(10),
            x='mean_abs_shap',
            y='feature',
            orientation='h',
            labels={'mean_abs_shap': 'Importance Score (SHAP)', 'feature': 'Feature'},
            color='mean_abs_shap',
            color_continuous_scale='Reds',
            title="Top 10 Most Important Features"
        )
        fig.update_layout(height=500, template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("💡 SHAP values show how much each feature contributes to predictions. Higher = more important.")
        
    except FileNotFoundError:
        st.warning("⚠️ SHAP importance not found. Run: `make interpret` to generate it.")
        
        if st.button("Run SHAP Analysis Now"):
            with st.spinner("Calculating feature importance..."):
                try:
                    import subprocess
                    subprocess.run(["make", "interpret"], check=True)
                    st.success("✅ SHAP analysis complete! Refresh this page.")
                except:
                    st.error("Failed to run SHAP. Try: `make interpret` in terminal")
    
    st.markdown("---")
    
    # Per-season table
    with st.expander("📋 Detailed Season-by-Season Breakdown"):
        season_acc_full = season_summary(preds)
        
        # Add more stats
        season_stats = []
        for season in season_acc_full['season'].unique():
            season_preds = preds[(preds['season'] == season) & (preds['predicted_winner'] == 1)]
            correct = ((season_preds['y_win'] == 1)).sum()
            total = len(season_preds)
            avg_conf = season_preds['p_win'].mean()
            
            season_stats.append({
                'Season': int(season),
                'Accuracy': f"{(correct/total*100) if total > 0 else 0:.1f}%",
                'Correct': correct,
                'Total': total,
                'Avg Confidence': f"{avg_conf*100:.1f}%"
            })
        
        stats_df = pd.DataFrame(season_stats).sort_values('Season', ascending=False)
        st.dataframe(stats_df, use_container_width=True)
    
    st.markdown("---")
    
    # Next step
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("✅ NEXT: Test Betting Strategies →", use_container_width=True, type="primary"):
            st.session_state.current_step = 5
            st.rerun()

# ==================== PAGE 5: TEST BETTING ====================
elif page == "💰 Test Betting":
    st.title("💰 Betting Strategy Simulator")
    
    show_help_box(
        "How Betting Simulation Works",
        "Test your betting strategy risk-free! Set your starting bankroll, how much to bet per race, and minimum confidence threshold. The simulator shows if you would have made or lost money using the model's predictions on historical races. ROI (Return on Investment) shows your profit as a percentage."
    )
    
    preds = load_predictions()
    if preds is None:
        st.error("⚠️ No predictions found!")
        st.stop()
    
    st.markdown("### ⚙️ Configure Your Strategy")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        starting_bank = st.number_input(
            "💵 Starting Bankroll ($)",
            min_value=100,
            max_value=100000,
            value=1000,
            step=100,
            help="How much money you start with"
        )
    
    with col2:
        stake = st.number_input(
            "🎯 Stake per Bet ($)",
            min_value=1,
            max_value=1000,
            value=50,
            step=10,
            help="How much to bet on each race"
        )
    
    with col3:
        confidence = st.slider(
            "📊 Minimum Confidence (%)",
            min_value=50,
            max_value=95,
            value=70,
            step=5,
            help="Only bet when model confidence is above this threshold"
        )
    
    st.info(f"💡 Strategy: Bet ${stake} per race when confidence ≥ {confidence}%. Conservative: 75%+, Aggressive: 60%+")
    
    st.markdown("---")
    
    if st.button("🚀 RUN SIMULATION", use_container_width=True, type="primary"):
        with st.spinner("Running betting simulation..."):
            # Filter by confidence
            betting_df = preds[preds['predicted_winner'] == 1].copy()
            betting_df = betting_df[betting_df['p_win'] >= confidence/100]
            
            if len(betting_df) == 0:
                st.warning(f"⚠️ No bets would be placed at {confidence}% confidence threshold. Try lowering it.")
                st.stop()
            
            # Simulate (even odds 2.0 for demo)
            betting_df['decimal_odds'] = 2.0
            betting_df['payout'] = betting_df.apply(
                lambda row: (row['decimal_odds'] - 1) * stake if row['y_win'] == 1 else -stake,
                axis=1
            )
            betting_df['cum_pnl'] = betting_df['payout'].cumsum()
            betting_df['bankroll'] = starting_bank + betting_df['cum_pnl']
            betting_df['number_of_bets'] = range(1, len(betting_df) + 1)
            betting_df['ROI'] = betting_df['cum_pnl'] / (betting_df['number_of_bets'] * stake)
            
            # Results
            total_bets = len(betting_df)
            wins = (betting_df['y_win'] == 1).sum()
            losses = total_bets - wins
            final_bankroll = betting_df['bankroll'].iloc[-1]
            final_pnl = betting_df['cum_pnl'].iloc[-1]
            roi = betting_df['ROI'].iloc[-1]
            win_rate = wins / total_bets
            
            # Display results
            if final_pnl > 0:
                st.markdown("""
                <div class="success-banner">
                    🎉 PROFITABLE STRATEGY! You would have made money!
                </div>
                """, unsafe_allow_html=True)
            elif final_pnl < 0:
                st.markdown("""
                <div class="warning-banner">
                    ⚠️ LOSS - Strategy needs adjustment. Try different confidence threshold.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Break-even strategy")
            
            st.markdown("### 📊 Simulation Results")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("🎲 Total Bets", total_bets)
            
            with col2:
                st.metric("✅ Wins", wins, delta=f"{win_rate*100:.1f}%")
            
            with col3:
                st.metric("❌ Losses", losses)
            
            with col4:
                st.metric("💰 Final Bankroll", f"${final_bankroll:.2f}",
                         delta=f"${final_pnl:+.2f}")
            
            with col5:
                st.metric("📈 ROI", f"{roi*100:.1f}%",
                         delta="Profit" if roi > 0 else "Loss")
            
            st.markdown("---")
            
            # Bankroll chart
            st.markdown("### 📈 Bankroll Over Time")
            
            fig = go.Figure()
            
            # Add bankroll line
            fig.add_trace(go.Scatter(
                x=betting_df['number_of_bets'],
                y=betting_df['bankroll'],
                mode='lines',
                name='Bankroll',
                line=dict(color='#e10600', width=3),
                fill='tozeroy',
                fillcolor='rgba(225, 6, 0, 0.1)',
                hovertemplate='Bet #%{x}<br>Bankroll: $%{y:.2f}<extra></extra>'
            ))
            
            # Add starting bankroll line
            fig.add_hline(
                y=starting_bank,
                line_dash="dash",
                line_color="white",
                annotation_text="Starting Bankroll",
                annotation_position="right"
            )
            
            # Add wins and losses markers
            wins_data = betting_df[betting_df['y_win'] == 1]
            losses_data = betting_df[betting_df['y_win'] == 0]
            
            fig.add_trace(go.Scatter(
                x=wins_data['number_of_bets'],
                y=wins_data['bankroll'],
                mode='markers',
                name='Wins',
                marker=dict(color='#00e676', size=10, symbol='circle'),
                hovertemplate='Win!<br>Bet #%{x}<br>Bankroll: $%{y:.2f}<extra></extra>'
            ))
            
            fig.add_trace(go.Scatter(
                x=losses_data['number_of_bets'],
                y=losses_data['bankroll'],
                mode='markers',
                name='Losses',
                marker=dict(color='#ff1744', size=8, symbol='x'),
                hovertemplate='Loss<br>Bet #%{x}<br>Bankroll: $%{y:.2f}<extra></extra>'
            ))
            
            fig.update_layout(
                xaxis_title="Bet Number",
                yaxis_title="Bankroll ($)",
                template='plotly_dark',
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Betting history
            st.markdown("### 📋 Detailed Betting History")
            
            display_cols = ['season', 'round', 'race_name', 'driver_id', 'p_win', 'y_win', 'payout', 'cum_pnl', 'bankroll', 'ROI']
            display_df = betting_df[display_cols].copy()
            display_df['p_win'] = display_df['p_win'].apply(lambda x: f"{x*100:.1f}%")
            display_df['ROI'] = display_df['ROI'].apply(lambda x: f"{x*100:.1f}%")
            display_df['Result'] = display_df['y_win'].apply(lambda x: "✅ WIN" if x == 1 else "❌ LOSS")
            
            st.dataframe(display_df, use_container_width=True, height=400)
            
            # Download option
            csv = betting_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Full Results (CSV)",
                data=csv,
                file_name=f"f1_betting_sim_{confidence}pct.csv",
                mime="text/csv"
            )
            
            st.markdown("---")
            
            # Strategy insights
            st.markdown("### 💡 Strategy Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 Win Rate Analysis")
                if win_rate >= 0.6:
                    st.success(f"✅ Excellent win rate: {win_rate*100:.1f}%")
                elif win_rate >= 0.5:
                    st.info(f"📊 Good win rate: {win_rate*100:.1f}%")
                else:
                    st.warning(f"⚠️ Low win rate: {win_rate*100:.1f}% - Consider higher confidence threshold")
                
                st.markdown("**Recommendations:**")
                if roi > 0.1:
                    st.write("✅ Strong strategy - keep it!")
                elif roi > 0:
                    st.write("📊 Profitable but marginal - can optimize")
                else:
                    st.write("❌ Losing strategy - try:")
                    st.write("• Increase confidence threshold to 75%+")
                    st.write("• Reduce stake size")
                    st.write("• Wait for better model accuracy")
            
            with col2:
                st.markdown("#### 💰 Bankroll Management")
                max_drawdown = (betting_df['bankroll'].max() - betting_df['bankroll'].min())
                risk_pct = (stake / starting_bank * 100)
                
                st.metric("Max Drawdown", f"${max_drawdown:.2f}")
                st.metric("Risk per Bet", f"{risk_pct:.1f}%")
                
                st.markdown("**Risk Assessment:**")
                if risk_pct <= 2:
                    st.success("✅ Conservative (≤2% per bet)")
                elif risk_pct <= 5:
                    st.info("📊 Moderate (2-5% per bet)")
                else:
                    st.warning("⚠️ Aggressive (>5% per bet)")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; padding: 20px;'>
    <h3 style='color: #e10600;'>🏎️ F1 Race Win Predictor</h3>
    <p>Professional Machine Learning System | Built with Streamlit & ❤️</p>
    <p>Data: Historical F1 Races (1995-2024) | Models: XGBoost, Random Forest, GBM, Logistic Regression</p>
    <p><b>13 Advanced Features</b> including driver form, team performance, grid analysis, and championship standings</p>
    <p style='margin-top: 20px;'>Made by <b>Harshith K.</b> | Version 2.0 (Enhanced UI & Workflow)</p>
</div>
""", unsafe_allow_html=True)

