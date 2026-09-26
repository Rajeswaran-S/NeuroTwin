import textwrap
import numpy as np
import plotly.graph_objects as go
import streamlit as st


def safe_markdown_html(html_code: str):
    """Safely renders HTML in Streamlit without triggering Markdown code block formatting."""
    lines = [line.strip() for line in html_code.splitlines()]
    clean_html = "\n".join(lines).strip()
    st.markdown(clean_html, unsafe_allow_html=True)


def render_metric_card(title: str, value: str, subtitle: str = "", badge_text: str = None, badge_type: str = "cyan"):
    """Renders a sleek medical glassmorphic stat card."""
    badge_html = f'<span class="badge-{badge_type}">{badge_text}</span>' if badge_text else ''
    html_str = f"""
<div class="glass-card">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
        <div class="metric-label">{title}</div>
        {badge_html}
    </div>
    <div class="metric-val">{value}</div>
    <div class="metric-sub">{subtitle}</div>
</div>
"""
    safe_markdown_html(html_str)


def overlay_mask_on_image(
    base_img: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.55,
    show_et: bool = True,
    show_ed: bool = True,
    show_net: bool = True
) -> np.ndarray:
    """
    Overlays BraTS multi-compartment mask on grayscale base slice:
    - ET (Label 3): Vivid Crimson Red (#FF2E93 -> [255, 46, 147])
    - ED (Label 2): Luminous Solar Yellow (#F1C40F -> [241, 196, 15])
    - NET (Label 1): Cyan / Teal (#00D2D3 -> [0, 210, 211])
    """
    # Normalize base image to [0, 255] RGB
    base_norm = (base_img * 255.0).astype(np.uint8) if base_img.max() <= 1.0 else base_img.astype(np.uint8)
    rgb = np.stack([base_norm, base_norm, base_norm], axis=-1).astype(np.float32)

    # Color definitions
    colors_map = {
        1: np.array([0, 210, 211], dtype=np.float32),   # NET / Necrosis: Cyan
        2: np.array([241, 196, 15], dtype=np.float32),  # ED / Edema: Amber Yellow
        3: np.array([255, 46, 147], dtype=np.float32)   # ET / Enhancing: Vivid Crimson
    }

    overlay = rgb.copy()
    for lbl, color in colors_map.items():
        if lbl == 3 and not show_et:
            continue
        if lbl == 2 and not show_ed:
            continue
        if lbl == 1 and not show_net:
            continue

        lbl_mask = (mask == lbl)
        if np.any(lbl_mask):
            overlay[lbl_mask] = (1.0 - alpha) * rgb[lbl_mask] + alpha * color

    return np.clip(overlay, 0, 255).astype(np.uint8)


def create_radiomics_radar_chart(radiomics: dict) -> go.Figure:
    """Creates an interactive Radar Polar Chart of top normalized Radiomics features."""
    categories = [
        'Sphericity', 'Compactness', 'Homogeneity',
        'Entropy (Norm)', 'GLCM Energy', 'Contrast (Norm)', 'Infiltration Risk'
    ]
    
    # Normalized feature values (0.0 to 1.0)
    values = [
        radiomics.get('sphericity', 0.5),
        radiomics.get('compactness', 0.4),
        radiomics.get('homogeneity', 0.6),
        min(1.0, radiomics.get('entropy', 2.0) / 4.0),
        radiomics.get('glcm_energy', 0.4),
        min(1.0, radiomics.get('contrast', 1.0) / 3.0),
        min(1.0, radiomics.get('biological_aggressiveness_score', 50) / 100.0)
    ]
    # Close radar loop
    categories_closed = categories + [categories[0]]
    values_closed = values + [values[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill='toself',
        fillcolor='rgba(56, 189, 248, 0.25)',
        line=dict(color='#38BDF8', width=2),
        name='Patient Radiomic Phenotype'
    ))

    # Benchmark population baseline
    benchmark_baseline = [0.65, 0.50, 0.70, 0.45, 0.55, 0.35, 0.40, 0.65]
    fig.add_trace(go.Scatterpolar(
        r=benchmark_baseline,
        theta=categories_closed,
        fill='none',
        line=dict(color='rgba(148, 163, 184, 0.4)', dash='dash', width=1.5),
        name='BraTS Cohort Baseline'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], gridcolor='#1E293B', color='#94A3B8'),
            angularaxis=dict(gridcolor='#1E293B', color='#E2E8F0'),
            bgcolor='rgba(15, 23, 42, 0.4)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=40, b=30, t=30),
        legend=dict(
            font=dict(color='#E2E8F0', size=11),
            x=0.0, y=1.1,
            orientation='h'
        ),
        height=380
    )
    return fig


def create_brain_region_bar_chart(overlaps: dict) -> go.Figure:
    """Horizontal bar chart showing Brain Lobe / Anatomical Segment Involvement %."""
    sorted_items = sorted(overlaps.items(), key=lambda x: x[1], reverse=True)
    regions = [x[0] for x in sorted_items]
    percentages = [x[1] for x in sorted_items]
    
    # Color palette
    bar_colors = [
        '#38BDF8' if p > 20 else ('#818CF8' if p > 5 else '#334155')
        for p in percentages
    ]

    fig = go.Figure(go.Bar(
        x=percentages,
        y=regions,
        orientation='h',
        marker=dict(color=bar_colors, line=dict(color='rgba(255,255,255,0.1)', width=1)),
        text=[f"{p}%" if p > 0 else "" for p in percentages],
        textposition='auto',
        textfont=dict(color='white', size=11)
    ))

    fig.update_layout(
        xaxis=dict(title='Tumor Infiltration Overlap (%)', range=[0, 100], gridcolor='#1E293B', color='#94A3B8'),
        yaxis=dict(autorange="reversed", color='#E2E8F0'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, b=30, t=20),
        height=340
    )
    return fig


def create_growth_trajectory_chart(df) -> go.Figure:
    """Longitudinal tumor growth prediction line chart comparing treatment regimens over 1 Month (30 Days)."""
    fig = go.Figure()
    x_col = "Day" if "Day" in df.columns else "Month"
    x_title = "Time Post-Diagnosis (Days)" if x_col == "Day" else "Time Post-Diagnosis (Months)"

    fig.add_trace(go.Scatter(
        x=df[x_col],
        y=df["Untreated Natural Progression (cm³)"],
        mode='lines+markers',
        name='Untreated Natural Progression',
        line=dict(color='#EF4444', width=2.5, dash='dash')
    ))

    fig.add_trace(go.Scatter(
        x=df[x_col],
        y=df["Standard Chemoradiation Protocol (cm³)"],
        mode='lines+markers',
        name='Standard Chemoradiation (Stupp)',
        line=dict(color='#F59E0B', width=2.5)
    ))

    fig.add_trace(go.Scatter(
        x=df[x_col],
        y=df["Surgical Resection + Adjuvant (cm³)"],
        mode='lines+markers',
        name='Gross Total Resection (GTR) + Adjuvant',
        line=dict(color='#10B981', width=3)
    ))

    fig.update_layout(
        title=dict(text='Longitudinal Gompertzian Tumor Volume Trajectory Simulation (1 Month / 30 Days)', font=dict(color='#E2E8F0', size=14)),
        xaxis=dict(title=x_title, gridcolor='#1E293B', color='#94A3B8'),
        yaxis=dict(title='Estimated Tumor Volume (cm³)', gridcolor='#1E293B', color='#94A3B8'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15, 23, 42, 0.4)',
        legend=dict(font=dict(color='#E2E8F0'), x=0.05, y=0.95),
        margin=dict(l=30, r=20, b=30, t=50),
        height=380
    )
    return fig
