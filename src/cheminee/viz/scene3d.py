"""Interactive 3D scene (plotly): terrain coloured by the lift index, routes with altitude."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from ..config import Params
from ..domain import Route, Terrain
from ..planning import Planner

ROUTE_COLORS = ["#e6194b", "#1f77b4", "#f58231", "#911eb4", "#17a589"]
LIFT_COLORSCALE = [[0.0, "#8d9c8a"], [0.2, "#c9c79a"], [0.45, "#f2c14e"], [0.7, "#f07f3c"], [1.0, "#c81e1e"]]


def fmt_hour(h: float) -> str:
    m = int(round(h * 60))
    return f"{m // 60:02d}:{m % 60:02d}"


def scoring_points(route: Route) -> np.ndarray:
    """(k, 3) ground coordinates of the scoring polyline (closed for triangles)."""
    pts = route.xy[route.turnpoints]
    return np.vstack([pts, pts[:1]]) if route.kind in ("triangle", "fai") else pts


def build_scene(
    terrain: Terrain,
    planner: Planner,
    routes: list[Route],
    params: Params,
    hours: list[float] | None = None,
    stride: int = 3,
    vertical_exaggeration: float = 1.5,
) -> go.Figure:
    """Build the 3D figure.

    Parameters
    ----------
    hours  : local hours of the lift-field animation frames (default: every hour of the window).
    stride : DEM decimation factor (3 -> ~90 m cells), keeps the HTML file light.
    """
    hours = hours or list(np.arange(np.ceil(params.t_start), planner.t_end + 1e-6, 1.0))
    z = terrain.z[::stride, ::stride]
    rows, cols = np.arange(terrain.ny)[::stride], np.arange(terrain.nx)[::stride]
    xs, _ = terrain.rowcol_to_xy(0, cols)
    _, ys = terrain.rowcol_to_xy(rows, 0)
    fields = [np.round(planner.lift.field(h)[::stride, ::stride], 3) for h in hours]

    def surface(k):
        return go.Surface(
            x=xs / 1000,
            y=ys / 1000,
            z=z,
            surfacecolor=fields[k],
            cmin=0,
            cmax=0.8,
            colorscale=LIFT_COLORSCALE,
            colorbar=dict(title="lift index", len=0.5, x=0.0),
            lighting=dict(ambient=0.55, diffuse=0.9, specular=0.05, roughness=0.9),
            lightposition=dict(x=-2000, y=2000, z=6000),
            name="terrain",
            hovertemplate="x %{x:.1f} km<br>y %{y:.1f} km<br>alt %{z:.0f} m<br>lift %{surfacecolor:.2f}<extra></extra>",
        )

    k0 = len(hours) // 2
    traces = [surface(k0)]
    traces.append(
        go.Scatter3d(
            x=planner.x[1:] / 1000,
            y=planner.y[1:] / 1000,
            z=planner.zg[1:] + 30,
            mode="markers",
            marker=dict(size=2.5, color="black"),
            name="thermal nodes",
            hoverinfo="skip",
        )
    )

    ceiling = float(planner.ceil.max())
    for k, r in enumerate(routes):
        color = ROUTE_COLORS[k % len(ROUTE_COLORS)]
        group, visible = f"route{k}", (True if k == 0 else "legendonly")
        prof = r.profile
        label = f"{k + 1}: {r.label}, {r.distance_km:.1f} km, {r.points:.0f} pts, {r.duration_h:.1f} h"
        traces.append(
            go.Scatter3d(
                x=prof[:, 0] / 1000,
                y=prof[:, 1] / 1000,
                z=prof[:, 2],
                mode="lines",
                line=dict(color=color, width=5),
                name=label,
                legendgroup=group,
                visible=visible,
                text=[fmt_hour(h) for h in prof[:, 3]],
                hovertemplate="%{text}<br>alt %{z:.0f} m<extra>route " + str(k + 1) + "</extra>",
            )
        )
        tp = scoring_points(r)
        names = (
            ["start", "TP1", "TP2", "TP3", "end"]
            if r.kind == "free"
            else ["TP1", "TP2", "TP3", ""]
            if r.kind in ("triangle", "fai")
            else ["start", "goal"]
        )
        traces.append(
            go.Scatter3d(
                x=tp[:, 0] / 1000,
                y=tp[:, 1] / 1000,
                z=np.full(len(tp), ceiling),
                mode="lines+markers+text",
                line=dict(color=color, width=2, dash="dash"),
                marker=dict(size=5, color=color, symbol="diamond", line=dict(color="white", width=1)),
                text=names[: len(tp)],
                textposition="top center",
                legendgroup=group,
                showlegend=False,
                visible=visible,
                hoverinfo="skip",
            )
        )

    traces.append(
        go.Scatter3d(
            x=[planner.x[0] / 1000],
            y=[planner.y[0] / 1000],
            z=[planner.zg[0] + 40],
            mode="markers+text",
            marker=dict(size=8, color="yellow", line=dict(color="black", width=2)),
            text=["takeoff"],
            textposition="top center",
            name="takeoff",
            hoverinfo="skip",
        )
    )

    frames = [
        go.Frame(data=[go.Surface(surfacecolor=fields[k])], traces=[0], name=fmt_hour(h)) for k, h in enumerate(hours)
    ]
    slider = dict(
        active=k0,
        currentvalue=dict(prefix="lift field at "),
        pad=dict(t=10),
        len=0.6,
        x=0.2,
        steps=[
            dict(
                method="animate",
                label=fmt_hour(h),
                args=[
                    [fmt_hour(h)],
                    dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0)),
                ],
            )
            for h in hours
        ],
    )

    w = params.weather
    z_top = max(ceiling, float(z.max())) + 100
    fig = go.Figure(data=traces, frames=frames)
    fig.update_layout(
        title=dict(
            text=(
                f"XC flights - {planner.lift.name} lift - day {params.day_of_year} - wind "
                f"{w.wind_dir_deg:.0f}° {w.wind_speed_kmh:.0f} km/h - cloud cover "
                f"{100 * w.cloud_cover:.0f} % - ceiling {planner.ceil.min():.0f}-{ceiling:.0f} m"
            ),
            font=dict(size=14),
        ),
        scene=dict(
            xaxis_title="x east [km]",
            yaxis_title="y north [km]",
            zaxis_title="altitude [m]",
            zaxis=dict(range=[float(z.min()) - 50, z_top]),
            aspectmode="manual",
            aspectratio=dict(
                x=1.0,
                y=terrain.height / terrain.width,
                z=vertical_exaggeration * (z_top - float(z.min())) / terrain.width,
            ),
            camera=dict(eye=dict(x=-0.65, y=-0.95, z=0.7)),
        ),
        sliders=[slider],
        legend=dict(x=0.68, y=0.98, bgcolor="rgba(255,255,255,0.7)"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def write_html(fig: go.Figure, path: str, offline: bool = True) -> None:
    """Write a standalone HTML file. `offline=True` embeds plotly.js (works without network)."""
    fig.write_html(path, include_plotlyjs=True if offline else "cdn", auto_play=False)
    print(f"[3d] scene saved to {path}")
