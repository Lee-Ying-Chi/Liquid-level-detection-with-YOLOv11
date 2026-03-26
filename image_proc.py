import numpy as np
import cv2
from config import *

class ImageProcessor:
    def __init__(self):
        pass

    def depth_to_colormap(self, depth_frame, depth_scale, min_d=MIN_D, max_d=MAX_D):
        depth_u16 = np.asanyarray(depth_frame.get_data()).astype(np.float32)
        depth_m = depth_u16 * depth_scale

        valid = depth_u16 > 0
        depth_m[~valid] = max_d
        depth_m = np.clip(depth_m, min_d, max_d)

        norm = (depth_m - min_d) / (max_d - min_d)
        norm = np.clip(norm, 0, 1)
        depth_norm = (255 * (1.0 - norm)).astype(np.uint8)

        depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
        return depth_color

    def process_line_detect(self, color_bgr, clahe_clip=2.0, clahe_tile=(8, 8), sobel_ksize=3):
        gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
        # clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_tile)
        # gray_clahe = clahe.apply(gray)
        gray_clahe = gray

        sobel_y = cv2.Sobel(gray_clahe, cv2.CV_32F, 0, 1, ksize=sobel_ksize)
        sobel_y_abs = np.abs(sobel_y)
        sobel_vis = cv2.normalize(sobel_y_abs, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        clahe_bgr = cv2.cvtColor(gray_clahe, cv2.COLOR_GRAY2BGR)
        sobel_bgr = cv2.cvtColor(sobel_vis, cv2.COLOR_GRAY2BGR)
        return clahe_bgr, sobel_bgr

    def sobel_y_gray(self, color_bgr, clahe_clip=2.0, clahe_tile=(8, 8), sobel_ksize=3):
        """Return Sobel(Y) as uint8 gray image (0..255)."""
        gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
        # clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_tile)
        # gray_clahe = clahe.apply(gray)
        gray_clahe = gray

        sobel_y = cv2.Sobel(gray_clahe, cv2.CV_32F, 0, 1, ksize=sobel_ksize)
        sobel_y_abs = np.abs(sobel_y)
        sobel_vis = cv2.normalize(sobel_y_abs, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return sobel_vis

    def apply_on_roi_fullframe(self, color_bgr, roi_xyxy):
        """Return (clahe_full, sobel_full) same size as input; outside ROI is black."""
        h, w = color_bgr.shape[:2]
        x1, y1, x2, y2, conf, cls = roi_xyxy

        out1 = np.zeros((h, w, 3), dtype=np.uint8)
        out2 = np.zeros((h, w, 3), dtype=np.uint8)

        roi = color_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            return out1, out2

        a, b = self.process_line_detect(
            roi,
            clahe_clip=CLAHE_CLIP,
            clahe_tile=CLAHE_TILE,
            sobel_ksize=SOBEL_KSIZE
        )
        out1[y1:y2, x1:x2] = a
        out2[y1:y2, x1:x2] = b
        return out1, out2

    def smooth_1d(self, prof, k=7):
        """Simple moving average; k should be odd."""
        prof = np.asarray(prof, dtype=np.float32)
        k = int(k)
        if k <= 1:
            return prof
        if k % 2 == 0:
            k += 1
        pad = k // 2
        p = np.pad(prof, (pad, pad), mode="edge")
        kernel = np.ones((k,), dtype=np.float32) / float(k)
        return np.convolve(p, kernel, mode="valid")

    def render_profile_plot(self, prof, width=480, height=360, title="1D profile"):
        """Render 1D profile into a BGR image."""
        prof = np.asarray(prof, dtype=np.float32)
        canvas = np.zeros((height, width, 3), dtype=np.uint8)

        if prof.size < 2:
            cv2.putText(canvas, "empty profile", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            return canvas

        mn = float(np.min(prof))
        mx = float(np.max(prof))
        denom = (mx - mn) if (mx > mn) else 1.0

        ys = (height - 1) - (prof - mn) / denom * (height - 1)
        ys = np.clip(ys, 0, height - 1).astype(np.int32)

        xs = np.linspace(0, width - 1, num=prof.size).astype(np.int32)

        for i in range(1, len(xs)):
            cv2.line(canvas, (xs[i - 1], ys[i - 1]), (xs[i], ys[i]), (0, 255, 0), 2)

        cv2.putText(canvas, title, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(canvas, f"min={mn:.1f}, max={mx:.1f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        return canvas

    # =========================
    # RANSAC quadratic fitting
    # =========================
    @staticmethod
    def _fit_quadratic_ls(x, y):
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        A = np.stack([x * x, x, np.ones_like(x)], axis=1)  # [a,b,c]
        coeff, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        return coeff  # (a,b,c)

    
    def ransac_quadratic(
        self,
        pts,
        n_iter=250,
        inlier_th=2.5,
        min_inliers=40,
        seed=None,
        # Image coordinate note: y grows downward, so "concave up" in image is a < 0.
        prefer_concave_up=True,   # prefer concave-up in image coordinates (a < 0)
        fallback_allow_any=True   # if not found, allow any sign as fallback
    ):
        """
        pts: (N,2) int/float array, columns [x,y]
        model: y = a x^2 + b x + c

        Returns:
          best_coeff (a,b,c) or None,
          best_inlier_mask (N,) bool or None
        """
        if pts is None or len(pts) < 3:
            return None, None

        pts = np.asarray(pts, dtype=np.float32)
        xs = pts[:, 0]
        ys = pts[:, 1]

        rng = np.random.default_rng(seed)

        best_coeff = None
        best_inliers = None
        best_score = -1  # score = #inliers (can extend later)

        # Two-stage search: first enforce concavity sign, then fallback if enabled.
        def run_stage(require_a_pos: bool):
            nonlocal best_coeff, best_inliers, best_score

            N = pts.shape[0]
            for _ in range(n_iter):
                idx = rng.choice(N, size=3, replace=False)
                x3 = xs[idx]
                y3 = ys[idx]

                # solve least squares for [a,b,c]
                A = np.stack([x3**2, x3, np.ones_like(x3)], axis=1)
                # avoid singular
                if np.linalg.matrix_rank(A) < 3:
                    continue

                coeff, *_ = np.linalg.lstsq(A, y3, rcond=None)
                a, b, c = coeff

                if require_a_pos and not (a < 0):
                    continue

                # compute residuals
                y_hat = a * xs * xs + b * xs + c
                err = np.abs(ys - y_hat)

                inliers = err <= inlier_th
                nin = int(np.sum(inliers))
                if nin < min_inliers:
                    continue

                # score: inlier count (你也可以之後改成 nin - lambda*RMSE)
                if nin > best_score:
                    best_score = nin
                    best_coeff = (float(a), float(b), float(c))
                    best_inliers = inliers.copy()

        if prefer_concave_up:
            run_stage(require_a_pos=True)

        if best_coeff is None and fallback_allow_any:
            run_stage(require_a_pos=False)

        return best_coeff, best_inliers
    
    def render_profile_plot_rightward(self, prof_1d, width, height,
                                      x_max=PROFILE_X_MAX,
                                      bg=(0, 0, 0),
                                      line_color=(0, 255, 0)):
        """
        Render 1D profile as a curve:
          - X axis = intensity (to the RIGHT is larger)
          - Y axis = y index (top->bottom)
        IMPORTANT: fixed x_max, no per-frame normalization.
        Values beyond x_max are clipped at right edge.
        """
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:] = bg

        if prof_1d is None or len(prof_1d) < 2:
            return canvas

        p = np.asarray(prof_1d, dtype=np.float32)

        n = p.shape[0]
        if n < 2:
            return canvas

        x_max = float(max(1e-6, x_max))

        # Map y (0..n-1) -> display y (0..height-1)
        # Map x (value) -> display x using fixed scale: x = value/x_max*(width-1)
        pts = []
        for i in range(n):
            y = int(round(i * (height - 1) / max(1, (n - 1))))
            x = int(round((p[i] / x_max) * (width - 1)))
            # allow overflow: clip at border (value > x_max will stick to right edge)
            x = 0 if x < 0 else (width - 1 if x >= width else x)
            pts.append((x, y))

        # Draw polyline
        for i in range(1, len(pts)):
            cv2.line(canvas, pts[i - 1], pts[i], line_color, 1, cv2.LINE_AA)

        return canvas
