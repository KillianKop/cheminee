"""Small matplotlib interface.

- Left click on the map  : move the takeoff and recompute.
- Right click on the map : set a goal and compute the fastest route to it.
- Sliders + "Compute"    : change weather / time parameters and recompute.
- "Map time" slider      : show the lift field at another time of the day (instant).
- Radio buttons          : objective of the optimiser, and route to highlight.
- "3D view"              : export the current result as an interactive HTML scene.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import numpy as np

from ..config import OBJECTIVES, Params
from ..domain import Terrain, sun_position
from ..lift import LiftModel
from ..planning import Planner, RouteOptimizer, default_takeoff
from .scene3d import ROUTE_COLORS, fmt_hour, scoring_points

LiftFactory = Callable[[Terrain, Params], LiftModel]

# (label, attribute, min, max, step); attributes prefixed with "w." belong to Params.weather
WEATHER_SLIDERS = [
    ("Wind from [deg]", "w.wind_dir_deg", 0, 360, 5),
    ("Wind [km/h]", "w.wind_speed_kmh", 0, 35, 1),
    ("Wind 3000 m [km/h]", "w.wind_aloft_speed_kmh", 0, 60, 1),
    ("Temperature [C]", "w.temp_c", 5, 38, 0.5),
    ("Dew point [C]", "w.dewpoint_c", -5, 22, 0.5),
    ("Inversion [m]", "w.inversion_alt_m", 1500, 4500, 50),
    ("Cloud cover", "w.cloud_cover", 0, 1, 0.05),
    ("Instability", "w.thermal_strength", 0.5, 1.4, 0.05),
]
OTHER_SLIDERS = [
    ("Breeze [km/h]", "w.breeze_kmh", 0, 20, 1),
    ("Snow line [m]", "w.snow_line_m", 800, 4000, 50),
    ("Takeoff time [h]", "t_start", 8, 16, 0.25),
    ("Landing time [h]", "t_end", 12, 21, 0.25),
    ("Day of year", "day_of_year", 1, 365, 1),
    ("Min climb [m/s]", "min_climb_ms", 0.2, 2.0, 0.1),
]


class App:
    def __init__(
        self,
        terrain: Terrain,
        params: Params,
        lift_factory: LiftFactory,
        takeoff_xy: tuple[float, float] | None = None,
        html_path: str = "xc_flights_3d.html",
    ):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider

        self.plt = plt
        self.t, self.p, self.lift_factory, self.html_path = terrain, params, lift_factory, html_path
        self.takeoff = takeoff_xy or default_takeoff(terrain, params, lift_factory(terrain, params))
        self.goal, self.routes, self.planner = None, [], None
        self.extent = [0, terrain.width / 1000, 0, terrain.height / 1000]

        self.fig = plt.figure(figsize=(16.5, 9))
        self.fig.canvas.manager.set_window_title("cheminee")
        self.ax = self.fig.add_axes([0.035, 0.07, 0.54, 0.88])

        self.sliders = {}
        for x0, specs in ((0.685, WEATHER_SLIDERS), (0.875, OTHER_SLIDERS)):
            y = 0.945
            for label, attr, lo, hi, step in specs:
                value = self._get(attr)
                if value is None:  # unset wind aloft: same as the low-level wind
                    value = params.weather.wind_speed_kmh
                ax_s = self.fig.add_axes([x0, y, 0.085, 0.02])
                self.sliders[attr] = Slider(ax_s, label, lo, hi, valinit=float(np.clip(value, lo, hi)), valstep=step)
                self.sliders[attr].label.set_fontsize(8.5)
                self.sliders[attr].valtext.set_fontsize(8.5)
                y -= 0.036
        self.map_time = Slider(
            self.fig.add_axes([0.875, 0.945 - 0.036 * 6.5, 0.085, 0.02]),
            "Map time [h]",
            6,
            21,
            valinit=0.5 * (params.t_start + params.t_end),
            valstep=0.25,
            color="#f0b429",
        )
        self.map_time.label.set_fontsize(8.5)
        self.map_time.on_changed(lambda _: self.draw())

        y = 0.60
        self.button = Button(
            self.fig.add_axes([0.60, y, 0.075, 0.04]), "Compute", color="#dfe9f5", hovercolor="#b9d2ee"
        )
        self.button.on_clicked(lambda _: self.compute())
        self.button3d = Button(
            self.fig.add_axes([0.60, y - 0.05, 0.075, 0.04]), "3D view", color="#e8f3e3", hovercolor="#c5e2b8"
        )
        self.button3d.on_clicked(lambda _: self.export_3d(open_browser=True))
        self.check = CheckButtons(
            self.fig.add_axes([0.60, y - 0.11, 0.09, 0.05], frameon=False),
            ["cast shadows"],
            [params.weather.cast_shadows],
        )

        self.fig.text(0.70, y + 0.045, "Objective", fontsize=9, weight="bold")
        self.objective = RadioButtons(
            self.fig.add_axes([0.695, y - 0.11, 0.09, 0.155], frameon=False),
            list(OBJECTIVES),
            active=OBJECTIVES.index(params.objective),
        )
        self.fig.text(0.80, y + 0.045, "Highlight", fontsize=9, weight="bold")
        self.radio = RadioButtons(
            self.fig.add_axes([0.795, y - 0.11, 0.09, 0.155], frameon=False),
            ["all"] + [f"route {i + 1}" for i in range(params.n_routes)],
        )
        self.highlight = 0
        self.radio.set_active(1)  # highlight the best route by default
        self.radio.on_clicked(self.on_radio)

        self.text_ax = self.fig.add_axes([0.595, 0.03, 0.40, 0.42])
        self.text_ax.axis("off")

        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.hillshade = terrain.hillshade()
        self.compute()

    # ---------------------------------------------------------------------- #
    def _get(self, attr: str):
        return getattr(self.p.weather, attr[2:]) if attr.startswith("w.") else getattr(self.p, attr)

    def read_params(self) -> Params:
        top = {a: s.val for a, s in self.sliders.items() if not a.startswith("w.")}
        wx = {a[2:]: s.val for a, s in self.sliders.items() if a.startswith("w.")}
        top["day_of_year"] = int(top["day_of_year"])
        top["t_end"] = max(top["t_end"], top["t_start"] + 0.5)
        # Hourly series (forecast) win over scalars: drop those the user just overrode with a slider.
        changed = {k for k in wx if wx[k] != self.sliders[f"w.{k}"].valinit}
        if changed & {"temp_c", "dewpoint_c", "inversion_alt_m"}:
            changed.add("ceiling_m")
        hourly = {k: v for k, v in self.p.weather.hourly.items() if k not in changed}
        if changed & {"wind_dir_deg", "wind_speed_kmh", "wind_aloft_speed_kmh"}:
            wx["wind_profile"] = {}  # the sliders replace the sounding wind profile
        weather = dataclasses.replace(
            self.p.weather, **wx, hourly=hourly, cast_shadows=bool(self.check.get_status()[0])
        )
        return dataclasses.replace(self.p, weather=weather, objective=self.objective.value_selected, **top)

    def compute(self):
        self.p = self.read_params()
        w = self.p.weather
        print(
            f"[gui] computing: takeoff=({self.takeoff[0]:.0f}, {self.takeoff[1]:.0f}) m, wind "
            f"{w.wind_dir_deg:.0f} deg / {w.wind_speed_kmh:.0f} km/h, objective {self.p.objective}"
        )
        self.planner = Planner(self.t, self.p, self.takeoff, self.lift_factory(self.t, self.p))
        optimizer = RouteOptimizer(self.planner)
        self.routes = optimizer.to_goal(self.goal) if self.goal else optimizer.optimise()
        self.draw()

    def on_click(self, event):
        if event.inaxes is not self.ax or getattr(self.fig.canvas.toolbar, "mode", ""):
            return
        point = (event.xdata * 1000.0, event.ydata * 1000.0)
        if event.button == 1:
            self.takeoff, self.goal = point, None
        elif event.button == 3:
            self.goal = point
        else:
            return
        self.compute()

    def on_radio(self, label: str):
        self.highlight = None if label == "all" else int(label.split()[1]) - 1
        self.draw()

    def export_3d(self, open_browser: bool = False):
        from .scene3d import build_scene, write_html

        write_html(build_scene(self.t, self.planner, self.routes, self.p), self.html_path)
        if open_browser:
            import pathlib
            import webbrowser

            webbrowser.open(pathlib.Path(self.html_path).resolve().as_uri())

    # ---------------------------------------------------------------------- #
    def draw(self):
        ax, t, p, pl, w = self.ax, self.t, self.p, self.planner, self.p.weather
        hour = self.map_time.val
        solar = pl.solar(hour)
        ax.clear()

        # Terrain: altitude tint modulated by hillshade; snow in white.
        ax.imshow(t.z, cmap="gist_earth", extent=self.extent, vmin=t.z.min() - 600, vmax=t.z.max() + 300)
        ax.imshow(self.hillshade, cmap="gray", extent=self.extent, alpha=0.45, vmin=0, vmax=1)
        if (t.z > w.snow_line_m).any():
            snow = np.zeros(t.z.shape + (4,))
            snow[t.z > w.snow_line_m] = (1, 1, 1, 0.55)
            ax.imshow(snow, extent=self.extent)

        # Lift field at the selected map time (transparent where there is no lift).
        lift = pl.lift.field(hour)
        rgba = self.plt.cm.YlOrRd(np.clip(lift / 0.8, 0, 1))
        rgba[..., 3] = np.clip((lift - 0.5 * p.lift_min) / 0.4, 0, 0.75)
        ax.imshow(rgba, extent=self.extent)

        # Thermal nodes: filled if usable at the map time.
        active = pl.climb[pl.bin_of(hour), 1:] >= p.min_climb_ms
        ax.scatter(pl.x[1:][active] / 1000, pl.y[1:][active] / 1000, s=14, c="k", zorder=3)
        ax.scatter(
            pl.x[1:][~active] / 1000,
            pl.y[1:][~active] / 1000,
            s=14,
            facecolors="none",
            edgecolors="k",
            linewidths=0.6,
            zorder=3,
        )

        # Routes.
        lines = []
        for k, r in enumerate(self.routes):
            dim = self.highlight is not None and self.highlight != k and len(self.routes) > 1
            color = ROUTE_COLORS[k % len(ROUTE_COLORS)]
            ax.plot(
                r.xy[:, 0] / 1000,
                r.xy[:, 1] / 1000,
                "-",
                color=color,
                lw=1.0 if dim else 2.2,
                alpha=0.25 if dim else 0.95,
                zorder=4 + (not dim),
            )
            if not dim:
                tp = scoring_points(r)
                ax.plot(tp[:, 0] / 1000, tp[:, 1] / 1000, "--", color=color, lw=1.2, zorder=5)
                if r.kind in ("triangle", "fai"):  # scored triangle B1-B2-B3
                    ax.fill(tp[:, 0] / 1000, tp[:, 1] / 1000, color=color, alpha=0.12, zorder=4)
                marks = r.xy[r.turnpoints[1:4]] if r.kind == "free" else r.xy[r.turnpoints]
                ax.scatter(
                    marks[:, 0] / 1000, marks[:, 1] / 1000, marker="^", s=70, color=color, edgecolors="w", zorder=6
                )
                ax.scatter(*r.xy[-1, :2] / 1000, marker="s", s=45, color=color, edgecolors="w", zorder=6)
            lines.append(
                f"{k + 1}: {r.label:<13} {r.distance_km:5.1f} km {r.points:6.1f} pts  "
                f"{r.duration_h:4.1f} h {r.speed_kmh:5.1f} km/h  "
                f"{fmt_hour(r.times[0])}-{fmt_hour(r.times[-1])}"
            )
        ax.scatter(
            self.takeoff[0] / 1000, self.takeoff[1] / 1000, marker="*", s=320, c="yellow", edgecolors="k", zorder=7
        )
        if self.goal:
            ax.scatter(
                self.goal[0] / 1000, self.goal[1] / 1000, marker="X", s=160, c="#00e5ff", edgecolors="k", zorder=7
            )

        # Wind (low level in blue, aloft in grey) and sun compass.
        cx, cy, rad = 0.075, 0.90, 0.055
        box = dict(fc="w", ec="none", alpha=0.7, pad=1.5)
        for alt, color, lw in ((w.wind_aloft_alt_m, "#607d8b", 1.5), (w.wind_alt_m, "#1565c0", 2.5)):
            v = w.wind_vec(alt, solar)
            speed = float(np.hypot(*v))
            if speed > 0.1:
                u = v / speed * rad
                ax.annotate(
                    "",
                    xy=(cx + u[0], cy + u[1]),
                    xytext=(cx - u[0], cy - u[1]),
                    xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw),
                    zorder=8,
                )
        low, high = w.wind_vec(w.wind_alt_m, solar), w.wind_vec(w.wind_aloft_alt_m, solar)
        ax.text(
            cx,
            cy - 0.085,
            f"wind {np.hypot(*low) * 3.6:.0f} km/h (aloft {np.hypot(*high) * 3.6:.0f})",
            transform=ax.transAxes,
            ha="center",
            fontsize=8,
            color="#1565c0",
            bbox=box,
            zorder=9,
        )
        az, el = (float(v) for v in sun_position(p.day_of_year, p.to_utc(hour), t.lat_mid, t.lon_mid))
        if el > 0:
            sa = np.radians(az)
            ax.scatter(
                cx + 0.21 + rad * np.sin(sa),
                cy + rad * np.cos(sa),
                s=160,
                c="#ffb300",
                edgecolors="k",
                transform=ax.transAxes,
                zorder=8,
            )
            ax.scatter(cx + 0.21, cy, s=12, c="k", transform=ax.transAxes, zorder=8)
        ax.text(
            cx + 0.21,
            cy - 0.085,
            f"sun az {az:.0f}° el {el:.0f}°",
            transform=ax.transAxes,
            ha="center",
            fontsize=8,
            bbox=box,
            zorder=9,
        )

        lon, lat = t.xy_to_lonlat(*self.takeoff)
        ax.set_title(
            f"{pl.lift.name} lift at {fmt_hour(hour)}  |  ceiling {w.ceiling(solar):.0f} m  |  "
            f"takeoff {lat:.4f}N {lon:.4f}E ({pl.zg[0]:.0f} m)",
            fontsize=10,
        )
        ax.set_xlabel("x east [km]   (left click: takeoff, right click: goal)")
        ax.set_ylabel("y north [km]")
        ax.set_xlim(self.extent[:2])
        ax.set_ylim(self.extent[2:])

        self.text_ax.clear()
        self.text_ax.axis("off")
        kw = dict(va="top", family="monospace", fontsize=8.5, transform=self.text_ax.transAxes)
        mode = "goal flight" if self.goal else f"objective: {p.objective}"
        self.text_ax.text(
            0,
            1.0,
            f"Routes ({mode})   window {fmt_hour(p.t_start)}-{fmt_hour(pl.t_end)}   "
            f"cloud base {w.cloud_base(solar):.0f} m\n"
            " #  type          distance  points   time   speed   takeoff-landing",
            **kw,
        )
        for k, line in enumerate(lines):
            self.text_ax.text(
                0,
                0.88 - 0.065 * k,
                line,
                color=ROUTE_COLORS[k % len(ROUTE_COLORS)],
                weight="bold" if self.highlight == k else "normal",
                **kw,
            )
        if not lines:
            self.text_ax.text(
                0,
                0.88,
                "No flight found with this weather: no usable thermal at the\n"
                "takeoff and none within glide range, or objective not reachable.",
                **kw,
            )
        self.text_ax.text(
            0,
            0.88 - 0.065 * max(len(lines), 2),
            "\n*  takeoff   ^  turnpoints   []  landing   X  goal\no  thermal node (filled = usable at map time)",
            **kw,
        )
        self.fig.canvas.draw_idle()
