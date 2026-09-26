"""
UI Styles: Medical Dark-Mode Glassmorphic Theme with Cyan/Violet accents
"""

CUSTOM_CSS = """
<style>
/* Medical Cyber-Dark Theme Styling */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Background gradient */
.stApp {
    background: radial-gradient(circle at 15% 15%, #0d192e 0%, #080c16 55%, #05070d 100%) !important;
    color: #e2e8f0;
}

/* Hero Header Banner */
.hero-header {
    background: linear-gradient(135deg, rgba(14, 165, 233, 0.12) 0%, rgba(139, 92, 246, 0.15) 50%, rgba(236, 72, 153, 0.08) 100%);
    border: 1px solid rgba(56, 189, 248, 0.25);
    border-radius: 16px;
    padding: 24px 30px;
    margin-bottom: 24px;
    box-shadow: 0 12px 36px -8px rgba(0, 0, 0, 0.5), inset 0 1px 1px rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(12px);
}

.hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    line-height: 1.2;
}

.hero-sub {
    font-size: 1.0rem;
    color: #94a3b8;
    margin-top: 8px;
    font-weight: 400;
}

/* Glassmorphic Metric Cards */
.glass-card {
    background: rgba(15, 23, 42, 0.65);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 18px 20px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
    backdrop-filter: blur(16px);
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.glass-card:hover {
    border-color: rgba(56, 189, 248, 0.4);
    transform: translateY(-2px);
}

.metric-label {
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #94a3b8;
    margin-bottom: 6px;
}

.metric-val {
    font-size: 1.65rem;
    font-weight: 800;
    font-family: 'JetBrains Mono', monospace;
    color: #f8fafc;
    line-height: 1.1;
}

.metric-sub {
    font-size: 0.78rem;
    color: #64748b;
    margin-top: 4px;
}

/* Badges & Tags */
.badge-cyan {
    background: rgba(6, 182, 212, 0.18);
    border: 1px solid #06b6d4;
    color: #22d3ee;
    padding: 3px 10px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    display: inline-block;
}

.badge-purple {
    background: rgba(168, 85, 247, 0.18);
    border: 1px solid #a855f7;
    color: #c084fc;
    padding: 3px 10px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    display: inline-block;
}

.badge-red {
    background: rgba(239, 68, 68, 0.18);
    border: 1px solid #ef4444;
    color: #f87171;
    padding: 3px 10px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    display: inline-block;
}

.badge-green {
    background: rgba(34, 197, 94, 0.18);
    border: 1px solid #22c55e;
    color: #4ade80;
    padding: 3px 10px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    display: inline-block;
}

.badge-amber {
    background: rgba(245, 158, 11, 0.18);
    border: 1px solid #f59e0b;
    color: #fbbf24;
    padding: 3px 10px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    display: inline-block;
}

/* Pipeline Step Indicator */
.pipeline-step {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 16px;
    background: rgba(30, 41, 59, 0.5);
    border-left: 4px solid #38bdf8;
    border-radius: 0 10px 10px 0;
    margin-bottom: 12px;
}

.pipeline-num {
    background: #0284c7;
    color: white;
    font-weight: 700;
    font-size: 0.85rem;
    width: 26px;
    height: 26px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
}

/* Streamlit Tabs Customization */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(15, 23, 42, 0.5);
    padding: 6px;
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.06);
}

.stTabs [data-baseweb="tab"] {
    height: 44px;
    white-space: pre-wrap;
    background-color: transparent;
    border-radius: 8px;
    color: #94a3b8;
    font-weight: 600;
    font-size: 0.9rem;
    padding: 0 18px;
    border: none !important;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #0284c7 0%, #4f46e5 100%) !important;
    color: #ffffff !important;
    box-shadow: 0 4px 14px rgba(2, 132, 199, 0.4);
}

/* Buttons */
.stButton > button {
    border-radius: 10px;
    font-weight: 600;
    border: 1px solid rgba(56, 189, 248, 0.3);
    background: linear-gradient(135deg, rgba(2, 132, 199, 0.2) 0%, rgba(79, 70, 229, 0.2) 100%);
    color: #e0f2fe;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #0284c7 0%, #4f46e5 100%);
    border-color: #38bdf8;
    color: #ffffff;
    box-shadow: 0 6px 20px rgba(2, 132, 199, 0.4);
    transform: translateY(-1px);
}
</style>
"""
