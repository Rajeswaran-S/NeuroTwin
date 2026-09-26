"""
Module: 3D Digital Twin Representation Engine
Generates interactive 3D volumetric surfaces, multi-planar orthogonal slices (Axial, Coronal, Sagittal),
and real-time biometric telemetry for patient-centric digital twin visualization.
"""

import numpy as np
import plotly.graph_objects as go
from skimage.measure import marching_cubes
from scipy.ndimage import gaussian_filter


def generate_3d_digital_twin_figure(
    seg_mask: np.ndarray,
    brain_volume: np.ndarray = None,
    downsample_factor: int = 2,
    show_brain_shell: bool = True,
    show_et: bool = True,
    show_ed: bool = True,
    show_net: bool = True
) -> go.Figure:
    """
    Creates an interactive 3D Plotly mesh figure rendering the patient's brain tumor digital twin.
    Compartments:
      - Brain Outer Contour (Translucent Silver Wireframe / Shell from Real MRI or Geometric Envelope)
      - Enhancing Tumor (ET) - Vivid Crimson Red
      - Non-Enhancing Core (NET) - Emerald Cyan
      - Vasogenic Edema (ED) - Luminous Amber/Yellow Isosurface
    """
    fig = go.Figure()

    if seg_mask.ndim != 3:
        # Fallback for 2D inputs
        fig.add_annotation(text="3D Digital Twin requires volumetric 3D scan.", showarrow=False)
        return fig

    # Downsample for smooth interactive browser rendering
    vol = seg_mask[::downsample_factor, ::downsample_factor, ::downsample_factor]
    d, h, w = vol.shape

    # 1. Brain Shell Isosurface (Extracted from real MRI volume if provided)
    if show_brain_shell:
        if brain_volume is not None and brain_volume.ndim == 3:
            b_vol = brain_volume[::downsample_factor, ::downsample_factor, ::downsample_factor]
            # Threshold foreground brain parenchyma from real MRI scan
            brain_mask = (b_vol > 0.08).astype(float)
            smoothed_brain = gaussian_filter(brain_mask, sigma=1.2)
            level = 0.35
        else:
            # Elliptical brain contour enclosing the volume
            z, y, x = np.ogrid[:d, :h, :w]
            cz, cy, cx = d / 2.0, h / 2.0, w / 2.0
            brain_field = (
                ((z - cz) / (d * 0.44))**2 +
                ((y - cy) / (h * 0.44))**2 +
                ((x - cx) / (w * 0.42))**2
            )
            brain_mask = (brain_field <= 1.0).astype(float)
            smoothed_brain = gaussian_filter(brain_mask, sigma=1.0)
            level = 0.50
        
        try:
            verts, faces, _, _ = marching_cubes(smoothed_brain, level=level)
            fig.add_trace(go.Mesh3d(
                x=verts[:, 2] * downsample_factor,
                y=verts[:, 1] * downsample_factor,
                z=verts[:, 0] * downsample_factor,
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color='#8892B0',
                opacity=0.08,
                name='Brain Anatomical Boundary',
                hoverinfo='name',
                showlegend=True
            ))
        except Exception:
            pass

    # 2. Vasogenic Edema (ED = 2)
    if show_ed and np.sum(vol == 2) > 10:
        ed_binary = gaussian_filter((vol == 2).astype(float), sigma=0.8)
        try:
            verts, faces, _, _ = marching_cubes(ed_binary, level=0.4)
            fig.add_trace(go.Mesh3d(
                x=verts[:, 2] * downsample_factor,
                y=verts[:, 1] * downsample_factor,
                z=verts[:, 0] * downsample_factor,
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color='#F1C40F',  # Luminous Amber Yellow
                opacity=0.25,
                name='Vasogenic Edema (ED)',
                hoverinfo='name',
                showlegend=True
            ))
        except Exception:
            pass

    # 3. Non-Enhancing Core / Necrosis (NET = 1)
    if show_net and np.sum(vol == 1) > 10:
        net_binary = gaussian_filter((vol == 1).astype(float), sigma=0.8)
        try:
            verts, faces, _, _ = marching_cubes(net_binary, level=0.4)
            fig.add_trace(go.Mesh3d(
                x=verts[:, 2] * downsample_factor,
                y=verts[:, 1] * downsample_factor,
                z=verts[:, 0] * downsample_factor,
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color='#00D2D3',  # Cyan Emerald
                opacity=0.65,
                name='Necrotic / Non-Enhancing Core (NET)',
                hoverinfo='name',
                showlegend=True
            ))
        except Exception:
            pass

    # 4. Enhancing Active Tumor Rim (ET = 3)
    if show_et and np.sum(vol == 3) > 10:
        et_binary = gaussian_filter((vol == 3).astype(float), sigma=0.8)
        try:
            verts, faces, _, _ = marching_cubes(et_binary, level=0.4)
            fig.add_trace(go.Mesh3d(
                x=verts[:, 2] * downsample_factor,
                y=verts[:, 1] * downsample_factor,
                z=verts[:, 0] * downsample_factor,
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color='#FF2E93',  # Active Hypervascular Crimson
                opacity=0.85,
                name='Enhancing Tumor (ET)',
                hoverinfo='name',
                showlegend=True
            ))
        except Exception:
            pass

    # 3D Scene Layout Styling
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X (Sagittal Width)', showbackground=False, zerolinecolor='#2B3A4A', gridcolor='#1E293B'),
            yaxis=dict(title='Y (Coronal Depth)', showbackground=False, zerolinecolor='#2B3A4A', gridcolor='#1E293B'),
            zaxis=dict(title='Z (Axial Height)', showbackground=False, zerolinecolor='#2B3A4A', gridcolor='#1E293B'),
            bgcolor='rgba(10, 15, 29, 0.95)',
            camera=dict(
                eye=dict(x=1.6, y=1.6, z=1.2),
                up=dict(x=0, y=0, z=1)
            ),
            aspectmode='data'
        ),
        paper_bgcolor='rgba(10, 15, 29, 0.0)',
        plot_bgcolor='rgba(10, 15, 29, 0.0)',
        margin=dict(l=0, r=0, b=0, t=30),
        legend=dict(
            font=dict(color='#E2E8F0', size=11),
            bgcolor='rgba(15, 23, 42, 0.8)',
            bordercolor='#334155',
            borderwidth=1,
            x=0.02,
            y=0.98
        )
    )

    return fig


def get_orthogonal_slices(
    volume: np.ndarray,
    mask: np.ndarray,
    axial_idx: int,
    coronal_idx: int,
    sagittal_idx: int
) -> dict:
    """
    Extracts 2D Orthogonal Slices across the 3 anatomical planes:
    - Axial (Transverse): Z-slice (top-down)
    - Coronal: Y-slice (front-back)
    - Sagittal: X-slice (left-right)
    """
    if volume.ndim != 3:
        return {
            "axial_img": volume, "axial_mask": mask,
            "coronal_img": volume, "coronal_mask": mask,
            "sagittal_img": volume, "sagittal_mask": mask
        }

    D, H, W = volume.shape
    z = np.clip(axial_idx, 0, D - 1)
    y = np.clip(coronal_idx, 0, H - 1)
    x = np.clip(sagittal_idx, 0, W - 1)

    # 1. Axial View (Z slice)
    ax_img = volume[z, :, :]
    ax_mask = mask[z, :, :]

    # 2. Coronal View (Y slice)
    cor_img = volume[:, y, :]
    cor_mask = mask[:, y, :]

    # 3. Sagittal View (X slice)
    sag_img = volume[:, :, x]
    sag_mask = mask[:, :, x]

    return {
        "axial_img": ax_img, "axial_mask": ax_mask, "axial_idx": z,
        "coronal_img": cor_img, "coronal_mask": cor_mask, "coronal_idx": y,
        "sagittal_img": sag_img, "sagittal_mask": sag_mask, "sagittal_idx": x
    }


def compute_digital_twin_telemetry(measurements: dict, localization: dict, radiomics: dict) -> dict:
    """
    Synthesizes multi-module telemetry into a comprehensive patient digital twin status report.
    """
    vol_wt = measurements.get("volume_wt_cm3", 0.0)
    vol_ed = measurements.get("volume_ed_cm3", 0.0)
    vol_tc = measurements.get("volume_tc_cm3", 0.0)
    vol_net = measurements.get("volume_net_cm3", 0.0)

    # 1. Mass Effect Score (0 - 100) based on Whole Tumor volume and edema
    mass_effect = min(100.0, (vol_wt / 80.0) * 100.0)

    # 2. Midline Shift Estimation (in millimeters)
    centroid_x = localization.get("centroid_normalized", (0.5, 0.5, 0.5))[2]
    midline_deviation = abs(centroid_x - 0.5) * 2.0  # 0 to 1
    midline_shift_mm = round(midline_deviation * min(vol_wt * 0.18, 12.0), 1)

    # 3. Necrotic Core Burden
    necrosis_ratio = round(vol_net / (vol_tc + 1e-4), 2)

    # 4. Aggressiveness Category
    agg_score = radiomics.get("biological_aggressiveness_score", 50)
    if agg_score >= 80:
        agg_status = "CRITICAL / HIGH AGGRESSION"
        agg_color = "#FF3366"
    elif agg_score >= 50:
        agg_status = "MODERATE / INTERMEDIATE"
        agg_color = "#F5A623"
    else:
        agg_status = "STABLE / LOW AGGRESSION"
        agg_color = "#10B981"

    return {
        "mass_effect_index": round(mass_effect, 1),
        "midline_shift_mm": midline_shift_mm,
        "necrosis_ratio": necrosis_ratio,
        "edema_burden": round((vol_ed / (vol_wt + 1e-4)) * 100.0, 1),
        "aggressiveness_score": agg_score,
        "aggressiveness_status": agg_status,
        "aggressiveness_color": agg_color,
        "primary_region": localization.get("primary_region", "Brain Cortex"),
        "hemisphere": localization.get("hemisphere", "Unknown")
    }
