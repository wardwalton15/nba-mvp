"""
Reusable UI components for the Streamlit dashboard.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict


def render_player_card(player_data: pd.Series, index: int):
    """
    Render a player card with MVP prediction and stats.

    Args:
        player_data: Series containing player information
        index: Player index for display
    """
    rank = player_data.get('Rank', index + 1)

    # Card container
    with st.container():
        # Player header
        col1, col2 = st.columns([3, 1])

        with col1:
            # Rank badge with different colors
            if rank == 1:
                badge_color = "🥇"
            elif rank == 2:
                badge_color = "🥈"
            elif rank == 3:
                badge_color = "🥉"
            else:
                badge_color = f"{rank}."

            st.markdown(f"### {badge_color} {player_data['Player']}")
            st.caption(f"Team: {player_data.get('Tm', 'N/A')} | "
                      f"Position: {player_data.get('Pos', 'N/A')} | "
                      f"Age: {player_data.get('Age', 'N/A')}")

        with col2:
            st.metric(
                label="Predicted Vote Share",
                value=f"{player_data['Predicted_Share']:.3f}",
                delta=None
            )

        # Stats row
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            pts = player_data.get('PTS', 0)
            st.metric("PPG", f"{pts:.1f}" if pd.notna(pts) else "N/A")

        with col2:
            ast = player_data.get('AST', 0)
            st.metric("APG", f"{ast:.1f}" if pd.notna(ast) else "N/A")

        with col3:
            trb = player_data.get('TRB', 0)
            st.metric("RPG", f"{trb:.1f}" if pd.notna(trb) else "N/A")

        with col4:
            team_win_pct = player_data.get('team_win_pct', 0)
            st.metric("Team Win %", f"{team_win_pct:.3f}" if pd.notna(team_win_pct) else "N/A")

        with col5:
            games = player_data.get('G', 0)
            st.metric("Games", f"{int(games)}" if pd.notna(games) else "N/A")

        st.markdown("---")


def render_top_predictions_chart(predictions: pd.DataFrame):
    """
    Render a bar chart of top predictions.

    Args:
        predictions: DataFrame with predictions
    """
    # Create bar chart
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=predictions['Player'],
        y=predictions['Predicted_Share'],
        marker_color='#FF6B35',
        text=predictions['Predicted_Share'].apply(lambda x: f'{x:.3f}'),
        textposition='outside'
    ))

    fig.update_layout(
        xaxis_title="Player",
        yaxis_title="Predicted Vote Share",
        height=400,
        showlegend=False,
        hovermode='x',
        margin=dict(t=20, b=100)
    )

    fig.update_xaxes(tickangle=-45)

    st.plotly_chart(fig, use_container_width=True)


def render_feature_importance_chart(feature_importance: pd.DataFrame):
    """
    Render a horizontal bar chart of feature importance.

    Args:
        feature_importance: DataFrame with feature importance scores
    """
    if len(feature_importance) == 0:
        st.info("Feature importance data not available.")
        return

    # Take top 10 features
    top_features = feature_importance.head(10)

    # Create horizontal bar chart
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=top_features['Importance'],
        y=top_features['Feature'],
        orientation='h',
        marker_color='#004E89',
        text=top_features['Importance'].apply(lambda x: f'{x:.4f}'),
        textposition='outside'
    ))

    fig.update_layout(
        xaxis_title="Importance Score",
        yaxis_title="Feature",
        height=400,
        showlegend=False,
        margin=dict(l=100, r=50, t=20, b=50)
    )

    # Reverse y-axis to show most important at top
    fig.update_yaxes(autorange="reversed")

    st.plotly_chart(fig, use_container_width=True)


def render_model_metrics(metadata: dict):
    """
    Render model performance metrics.

    Args:
        metadata: Dictionary with model metadata
    """
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Mean Squared Error",
            value=f"{metadata.get('mse', 'N/A'):.6f}" if isinstance(metadata.get('mse'), (int, float)) else "N/A"
        )

    with col2:
        st.metric(
            label="Mean Absolute Error",
            value=f"{metadata.get('mae', 'N/A'):.6f}" if isinstance(metadata.get('mae'), (int, float)) else "N/A"
        )

    with col3:
        st.metric(
            label="Training Period",
            value=metadata.get('years_range', 'N/A')
        )

    with col4:
        st.metric(
            label="Training Samples",
            value=f"{metadata.get('n_train_samples', 'N/A'):,}" if isinstance(metadata.get('n_train_samples'), int) else "N/A"
        )

    # Model saved date
    saved_at = metadata.get('saved_at', 'N/A')
    if saved_at != 'N/A':
        st.caption(f"Model last trained: {saved_at}")


def render_comparison_table(predictions: pd.DataFrame):
    """
    Render a comparison table of top candidates.

    Args:
        predictions: DataFrame with predictions
    """
    # Select relevant columns
    display_cols = ['Rank', 'Player', 'Tm', 'Predicted_Share', 'PTS', 'AST',
                   'TRB', 'team_win_pct', 'G']

    # Filter to existing columns
    available_cols = [col for col in display_cols if col in predictions.columns]

    display_df = predictions[available_cols].copy()

    # Format numeric columns
    if 'Predicted_Share' in display_df.columns:
        display_df['Predicted_Share'] = display_df['Predicted_Share'].apply(lambda x: f'{x:.3f}')
    if 'team_win_pct' in display_df.columns:
        display_df['team_win_pct'] = display_df['team_win_pct'].apply(lambda x: f'{x:.3f}')
    if 'PTS' in display_df.columns:
        display_df['PTS'] = display_df['PTS'].apply(lambda x: f'{x:.1f}')
    if 'AST' in display_df.columns:
        display_df['AST'] = display_df['AST'].apply(lambda x: f'{x:.1f}')
    if 'TRB' in display_df.columns:
        display_df['TRB'] = display_df['TRB'].apply(lambda x: f'{x:.1f}')
    if 'G' in display_df.columns:
        display_df['G'] = display_df['G'].apply(lambda x: f'{int(x)}')

    # Rename columns for display
    display_df = display_df.rename(columns={
        'Tm': 'Team',
        'Predicted_Share': 'Predicted Vote Share',
        'PTS': 'PPG',
        'AST': 'APG',
        'TRB': 'RPG',
        'team_win_pct': 'Team Win %',
        'G': 'Games'
    })

    st.dataframe(display_df, use_container_width=True, hide_index=True)


def render_shap_values(shap_values: Dict[str, pd.DataFrame], top_predictions: pd.DataFrame):
    """
    Render SHAP value explanations for top 5 players.

    Args:
        shap_values: Dictionary mapping player names to SHAP DataFrames
        top_predictions: DataFrame with top predictions (for ordering)
    """
    st.subheader("🔍 Prediction Explanations (SHAP Values)")
    st.markdown(
        "SHAP values show how each feature contributes to pushing the prediction "
        "higher (positive) or lower (negative) from the baseline."
    )

    # Get top 5 players in order
    top_5_players = top_predictions.head(5)['Player'].tolist()

    # Create tabs for each player
    tabs = st.tabs(top_5_players)

    for tab, player_name in zip(tabs, top_5_players):
        with tab:
            if player_name not in shap_values:
                st.warning(f"SHAP values not available for {player_name}")
                continue

            shap_df = shap_values[player_name].copy()

            # Take top 8 features by absolute SHAP value
            top_features = shap_df.head(8)

            # Create waterfall-style horizontal bar chart
            fig = go.Figure()

            # Color based on positive/negative SHAP value
            colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in top_features['SHAP_Value']]

            fig.add_trace(go.Bar(
                y=top_features['Feature'],
                x=top_features['SHAP_Value'],
                orientation='h',
                marker_color=colors,
                text=[f"{v:.4f}" for v in top_features['SHAP_Value']],
                textposition='outside',
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "SHAP Value: %{x:.4f}<br>"
                    "<extra></extra>"
                )
            ))

            fig.update_layout(
                xaxis_title="SHAP Value (Impact on Prediction)",
                yaxis_title="Feature",
                height=350,
                showlegend=False,
                margin=dict(l=120, r=80, t=20, b=50),
                xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='gray')
            )

            # Reverse y-axis to show most important at top
            fig.update_yaxes(autorange="reversed")

            st.plotly_chart(fig, use_container_width=True)

            # Show feature values table
            with st.expander("View Feature Values"):
                display_df = top_features[['Feature', 'Feature_Value', 'SHAP_Value']].copy()
                display_df.columns = ['Feature', 'Value', 'SHAP Impact']
                display_df['Value'] = display_df['Value'].apply(lambda x: f"{x:.3f}" if abs(x) < 100 else f"{x:.1f}")
                display_df['SHAP Impact'] = display_df['SHAP Impact'].apply(lambda x: f"{x:+.4f}")
                st.dataframe(display_df, use_container_width=True, hide_index=True)
