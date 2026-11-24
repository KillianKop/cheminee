"""Pilot-oriented emagram: altitude in metres, temperature curves, rising parcel, wind by level."""

from __future__ import annotations

import numpy as np

from ..weather.sounding import DRY_LAPSE, Sounding


def plot_emagram(sounding: Sounding, hour_utc: float, path: str | None = None, max_alt_m: float = 4500.0):
    """Draw the emagram of one hour. Returns the matplotlib figure (saved to `path` if given)."""
    import matplotlib

    if path:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k = sounding.index(hour_utc)
    alt, temp, dew = sounding.profile(k)
    p_alt, p_temp = sounding.parcel(k)
    base, top, ceiling = sounding.cloud_base(k), sounding.thermal_top(k), sounding.ceiling(k)
    z_max = min(max_alt_m, alt[-1])
    keep = p_alt <= min(top, z_max)

    fig, (ax, axw) = plt.subplots(1, 2, figsize=(10, 7.5), sharey=True, gridspec_kw=dict(width_ratios=[4, 1]))
    t_lo, t_hi = (
        np.floor(min(temp[alt <= z_max].min(), np.nanmin(dew[alt <= z_max])) / 5) * 5 - 5,
        np.ceil((temp[0] + sounding.parcel_excess_c) / 5) * 5 + 5,
    )

    # dry adiabats: the slope a rising, non-condensing parcel follows
    for t0 in np.arange(t_lo - 40, t_hi + 40, 5.0):
        ax.plot(
            [t0, t0 - DRY_LAPSE * (z_max - sounding.elevation_m)],
            [sounding.elevation_m, z_max],
            color="#c9a66b",
            lw=0.5,
            alpha=0.6,
            zorder=1,
        )
    # convective layer usable by the pilot
    ax.axhspan(
        sounding.elevation_m,
        max(ceiling, sounding.elevation_m),
        color="#ffe082",
        alpha=0.5,
        zorder=0,
        label=f"usable layer (ceiling {ceiling:.0f} m)",
    )
    if base < top:  # cumulus between the cloud base and the thermal top
        ax.axhspan(base, min(top, z_max), color="#b0bec5", alpha=0.35, zorder=0, label=f"cloud (base {base:.0f} m)")
    ax.plot(temp, alt, "-o", color="#d32f2f", lw=2, ms=4, label="temperature", zorder=3)
    ax.plot(dew, alt, "-o", color="#1565c0", lw=2, ms=4, label="dew point", zorder=3)
    ax.plot(
        p_temp[keep],
        p_alt[keep],
        "--",
        color="k",
        lw=1.5,
        label=f"parcel (+{sounding.parcel_excess_c:.0f} °C at the ground)",
        zorder=4,
    )
    ax.axhline(sounding.elevation_m, color="#5d4037", lw=2)
    ax.set(
        xlim=(t_lo, t_hi),
        ylim=(sounding.elevation_m - 50, z_max + 200),
        xlabel="temperature [°C]",
        ylabel="altitude [m]",
        title=f"Emagram at {sounding.hours_utc[k]:02.0f}:00 UTC - model ground "
        f"{sounding.elevation_m:.0f} m - thermal top {top:.0f} m",
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8.5)

    # wind by level: arrow pointing where the air goes (north up), speed as text
    w_alt, w_speed, w_dir = sounding.wind_levels(k)
    axw.set(xlim=(-1, 1.6), title="wind [km/h]")
    axw.set_xticks([])
    span = z_max - sounding.elevation_m
    for a, s, d in zip(w_alt, w_speed, w_dir):
        if a > z_max:
            continue
        ang = np.radians(d)
        color = "#2e7d32" if s < 15 else "#f9a825" if s < 25 else "#c62828"
        dx, dy = -0.45 * np.sin(ang), -0.45 * np.cos(ang) * span / 12.0
        axw.annotate("", xy=(dx, a + dy), xytext=(-dx, a - dy), arrowprops=dict(arrowstyle="-|>", color=color, lw=2))
        axw.text(0.75, a, f"{s:.0f}", va="center", fontsize=9, color=color, weight="bold")
    axw.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print(f"[emagram] saved {path}")
    return fig
