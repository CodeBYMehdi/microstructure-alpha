import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy import stats

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def create_signal_opportunity_dashboard():
    print("Generation of Interactive Dashboards for Signal Discovery...")
    
    # 1. Load Data
    try:
        df_trades = pd.read_csv(os.path.join(BASE_DIR, "trade_journal.csv"))
        df_decisions = pd.read_csv(os.path.join(BASE_DIR, "decision_log.csv"))
        df_market = pd.read_csv(os.path.join(BASE_DIR, "market_state_log.csv"))
    except FileNotFoundError as e:
        print(f"Error loading logs: {e}. Make sure you've run the backtest recently.")
        return

    # ------------------------------------------------------------------
    # Plot 1: Signal Quality & PnL Outcomes Explorer 
    # (Interactive version of the 3D trade outcome scatter)
    # ------------------------------------------------------------------
    # Filter valid trades
    df_valid_trades = df_trades[df_trades['net_pnl'].notna()].copy()
    
    # Create size mappings (absolute PnL) and discrete color mappings (Win/Loss)
    df_valid_trades['Abs PnL'] = df_valid_trades['net_pnl'].abs().clip(lower=0.1)
    df_valid_trades['Outcome'] = np.where(df_valid_trades['net_pnl'] > 0, 'Profit', 'Loss')
    
    fig1 = px.scatter_3d(
        df_valid_trades,
        x='transition_strength',
        y='regime_confidence',
        z='volatility_at_entry',
        color='Outcome',
        color_discrete_map={'Profit': '#00ff00', 'Loss': '#ff0000'},
        size='Abs PnL',
        hover_data=[
            'net_pnl', 'regime_id', 'hold_duration_windows', 
            'entry_price', 'exit_reason'
        ],
        title="Interactive Signal Validation (Hover to identify profitable structural conditions)",
        labels={
            'transition_strength': 'Transition Strength (Edge)',
            'regime_confidence': 'Confidence Score',
            'volatility_at_entry': 'Sigma at Entry'
        }
    )
    
    fig1.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
    fig1.write_html(os.path.join(BASE_DIR, "interactive_signal_quality.html"))
    print("-> Saved interactive_signal_quality.html")


    # ------------------------------------------------------------------
    # Plot 2: Opportunity Landscape & Signal Generation
    # (Shows the exact algorithm decisions vs the structural landscape)
    # ------------------------------------------------------------------
    # Filter out boring flats
    df_decisions_filtered = df_decisions[
        (df_decisions['result'].isin(['TRADE', 'REJECTED'])) | 
        (df_decisions['transition_strength'].abs() > 0.1)
    ].copy()

    # Need 'l2_liquidity_slope' if available, otherwise just use delta_sigma
    y_axis_name = 'l2_liquidity_slope' if 'l2_liquidity_slope' in df_decisions_filtered.columns else 'delta_sigma'

    fig2 = px.scatter_3d(
        df_decisions_filtered,
        x='transition_strength',
        y=y_axis_name,
        z='tail_risk',
        color='result',
        color_discrete_map={
            'TRADE': '#00bfff', 
            'REJECTED': '#ffa500', 
            'FLAT': '#d3d3d3'
        },
        hover_data=[
            'reason', 'action', 'regime_id', 'delta_entropy', 'delta_skew'
        ],
        title="Interactive Opportunity Landscape (Identify why signals trigger vs abort)",
        labels={
            'transition_strength': 'Signal Strength',
            y_axis_name: 'Market Context (Liq / Vol Shift)',
            'tail_risk': 'Fat Tail Risk (Alpha)'
        }
    )
    
    fig2.update_traces(marker=dict(size=4, line=dict(width=0.5, color='DarkSlateGrey')))
    fig2.write_html(os.path.join(BASE_DIR, "interactive_opportunity_landscape.html"))
    print("-> Saved interactive_opportunity_landscape.html")

    
    # ------------------------------------------------------------------
    # Plot 3: 3D Microstructure State Drift (Regime Shift Mapping)
    # ------------------------------------------------------------------
    if len(df_market) > 100:
        step = max(1, len(df_market) // 2000) # Subsample to prevent crash
        df_market_sample = df_market.iloc[::step].copy()
        
        # Color by regime to see actual mapping of volatility to regimes
        df_market_sample['Regime'] = "Reg " + df_market_sample['regime_id'].astype(str)

        fig3 = px.scatter_3d(
            df_market_sample,
            x='mu',
            y='sigma',
            z='timestamp',
            color='Regime',
            hover_data=['entropy', 'skew', 'kurtosis', 'book_slope'],
            title="Drift vs Volatility Interactive State Progression",
            labels={'mu': 'Drift (µ)', 'sigma': 'Volatility (σ)', 'timestamp': 'Time'}
        )
        
        # Draw path
        fig3.add_trace(go.Scatter3d(
            x=df_market_sample['mu'],
            y=df_market_sample['sigma'],
            z=df_market_sample['timestamp'],
            mode='lines',
            line=dict(color='rgba(150,150,150,0.5)', width=1),
            showlegend=False,
            hoverinfo='skip'
        ))

        fig3.write_html(os.path.join(BASE_DIR, "interactive_volatility_drift.html"))
        print("-> Saved interactive_volatility_drift.html")


    print("\nVisualization generation complete. Open the .html files in your browser to explore opportunities.")

if __name__ == "__main__":
    create_signal_opportunity_dashboard()
