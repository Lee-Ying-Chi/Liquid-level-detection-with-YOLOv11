# telemetry.py
import os
import time
from dataclasses import dataclass
from typing import Optional
import sys


@dataclass
class RansacTelemetry:
    """Latest telemetry snapshot."""
    y_vertex: Optional[float] = None
    y_center: Optional[float] = None
    inliers_count: Optional[int] = None

    # --- NEW: gap + ml ---
    gap_px: Optional[float] = None
    ml: Optional[float] = None
    avg_ml: Optional[float] = None

    # optional extras (kept for debugging / future use)
    x_vertex: Optional[float] = None
    a: Optional[float] = None
    b: Optional[float] = None
    c: Optional[float] = None
    ts: Optional[float] = None


class TelemetryTable:
    """
    Terminal telemetry:
      - refresh a small table in-place (ANSI clear + cursor home)
      - also append to CSV
    """

    def __init__(self, csv_path: str = "logs/ransac_vertex_log.csv", refresh_hz: float = 10.0):
        self.csv_path = csv_path
        self.refresh_hz = max(0.1, float(refresh_hz))
        self._min_dt = 1.0 / self.refresh_hz
        self._last_render_t = 0.0
        self.state = RansacTelemetry()
        self._rendered_once = False

        # ensure dir exists
        d = os.path.dirname(self.csv_path)
        if d:
            os.makedirs(d, exist_ok=True)

        # write header if file not exist
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", encoding="utf-8") as f:
                f.write("ts,y_vertex,y_center,inliers_count,gap_px,ml,x_vertex,a,b,c\n")

    def update(
        self,
        y_vertex: Optional[float],
        y_center: Optional[float],
        inliers_count: Optional[int],
        # --- NEW ---
        gap_px: Optional[float] = None,
        ml: Optional[float] = None,
        avg_ml: Optional[float] = None,
        # ---
        x_vertex: Optional[float] = None,
        coeff: Optional[tuple] = None,
        ts: Optional[float] = None,
        write_csv: bool = True,
    ):
        if ts is None:
            ts = time.time()

        self.state.ts = ts
        self.state.y_vertex = y_vertex
        self.state.y_center = y_center
        self.state.inliers_count = inliers_count

        self.state.gap_px = gap_px
        self.state.ml = ml
        self.state.avg_ml = avg_ml

        self.state.x_vertex = x_vertex

        if coeff is not None:
            self.state.a, self.state.b, self.state.c = coeff
        else:
            self.state.a = self.state.b = self.state.c = None

        if write_csv:
            self._append_csv()

    def maybe_render(self):
        now = time.time()
        if now - self._last_render_t < self._min_dt:
            return
        self._last_render_t = now
        self._render_table()

    def _append_csv(self):
        s = self.state

        def fmt(x):
            if x is None:
                return ""
            if isinstance(x, (int, float)):
                return f"{float(x):.4f}"
            return str(x)

        line = (
            f"{fmt(s.ts)},"
            f"{fmt(s.y_vertex)},"
            f"{fmt(s.y_center)},"
            f"{'' if s.inliers_count is None else int(s.inliers_count)},"
            f"{fmt(s.gap_px)},"
            f"{fmt(s.ml)},"
            f"{fmt(s.x_vertex)},"
            f"{fmt(s.a)},"
            f"{fmt(s.b)},"
            f"{fmt(s.c)}\n"
        )
        with open(self.csv_path, "a", encoding="utf-8") as f:
            f.write(line)

    def _render_table(self):
        s = self.state

        def cell(name: str, val, width=16):
            if val is None:
                txt = "-"
            elif isinstance(val, float):
                txt = f"{val:.2f}"
            else:
                txt = str(val)
            return f"{name:<12}: {txt:<{width}}"
        lines = [
            "RANSAC Telemetry (vertex as target height)",
            "-" * 72,
            cell("y_vertex", s.y_vertex) + " | " + cell("y_center", s.y_center),
            cell("bottom_gap_px", s.gap_px) + " | " + cell("ml", s.ml),
            cell("avg_ml", s.avg_ml),
            cell("inliers", s.inliers_count) + " | " + cell("x_vertex", s.x_vertex),
            (cell("a", s.a) + " | " + cell("b", s.b) + " | " + cell("c", s.c)) if s.a is not None
            else (cell("a", None) + " | " + cell("b", None) + " | " + cell("c", None)),
            "-" * 72,
            f"csv: {self.csv_path}",
        ]

        if self._rendered_once:
            sys.stdout.write(f"\033[{len(lines)}F")  # 游標上移到表格起點
        else:
            self._rendered_once = True

        sys.stdout.write("\n".join(lines) + "\n")
        sys.stdout.flush()
