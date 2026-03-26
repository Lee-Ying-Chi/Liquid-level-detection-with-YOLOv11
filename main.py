# main.py (refactored: background compute always-on; views are display only)

import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d

from config import *
from yolo_roi import YoloRoi
from image_proc import ImageProcessor
from telemetry import TelemetryTable
# from aruco_pose import ArucoPoseEstimator
from Surface_Height_Client import SurfaceWSClient


# -----------------------------
# 3D loop stays as-is
# -----------------------------
def show_3d_loop(pipeline, align):
    """
    3D: show point cloud with RGB texture (viewer-like) using Open3D.
    Close the Open3D window to return to 2D.
    """
    print("[3D] RGB-textured point cloud. Rotate/zoom with mouse. Close the 3D window to return to 2D.")

    pc = rs.pointcloud()

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="RealSense 3D PointCloud (RGB)", width=960, height=540)

    opt = vis.get_render_option()
    opt.point_size = 1.0
    opt.background_color = np.asarray([0, 0, 0])

    pcd = o3d.geometry.PointCloud()
    added = False

    try:
        while True:
            if not vis.poll_events():
                return "to2d"

            try:
                frames = pipeline.wait_for_frames(timeout_ms=FRAME_TIMEOUT_MS)
            except RuntimeError:
                continue
            aligned = align.process(frames)

            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            pc.map_to(color_frame)
            points = pc.calculate(depth_frame)

            vtx = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
            tex = np.asanyarray(points.get_texture_coordinates()).view(np.float32).reshape(-1, 2)

            mask = vtx[:, 2] > 0
            vtx = vtx[mask]
            tex = tex[mask]
            if vtx.shape[0] == 0:
                continue

            z = vtx[:, 2]
            mask2 = (z >= MIN_D) & (z <= MAX_D)
            vtx = vtx[mask2]
            tex = tex[mask2]
            if vtx.shape[0] == 0:
                continue

            color_img = np.asanyarray(color_frame.get_data())  # BGR
            h, w, _ = color_img.shape

            u = np.clip(tex[:, 0], 0.0, 0.999999)
            v = np.clip(tex[:, 1], 0.0, 0.999999)
            x = (u * w).astype(np.int32)
            y = (v * h).astype(np.int32)

            bgr = color_img[y, x, :].astype(np.float32) / 255.0
            rgb = bgr[:, ::-1]

            pcd.points = o3d.utility.Vector3dVector(vtx.astype(np.float64))
            pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64))

            if not added:
                vis.add_geometry(pcd)
                added = True
                vis.reset_view_point(True)

                ctr = vis.get_view_control()
                bbox = pcd.get_axis_aligned_bounding_box()
                center = bbox.get_center()
                ctr.set_lookat(center)
                ctr.set_front([0, 0, -1.0])
                ctr.set_up([0, -1, 0])
                ctr.set_zoom(0.5)
            else:
                vis.update_geometry(pcd)

            vis.update_renderer()

    finally:
        vis.destroy_window()


def make_blank_profile_panels():
    blank = np.zeros((PROFILE_PLOT_H, PROFILE_PLOT_W, 3), dtype=np.uint8)
    return {
        "sobel_left": blank.copy(),
        "plot": blank.copy(),
        "sobel_peaks": blank.copy(),
        "sobel_ransac": blank.copy(),
        "combo": np.hstack((blank, blank.copy(), blank.copy(), blank.copy())),
    }


def compute_profile_state(color_bgr, roi, imageProc: ImageProcessor, roi_mask=None, liquid_mask=None):
    state = {
        "roi": roi,
        "valid_roi": False,
        "has_liquid": False,
        "sobel_gray": None,
        "H": 0,
        "W": 0,
        "left": 0,
        "right": 0,
        "prof": None,
        "peak_idx": np.array([], dtype=np.int32),
        "peak_vals": np.array([], dtype=np.float32),
        "y0": None,
        "y_vertex": None,
        "y_center": None,
        "x_vertex": None,
        "inliers_count": 0,
        "coeff": None,
    }

    if roi is None:
        return state

    x1, y1, x2, y2, conf, cls = roi
    roi_bgr = color_bgr[y1:y2, x1:x2]
    if roi_bgr.size == 0:
        return state
    state["valid_roi"] = True

    if roi_mask is not None:
        mask_roi = roi_mask[y1:y2, x1:x2]
        if mask_roi.size > 0:
            if (mask_roi.shape[0] != roi_bgr.shape[0]) or (mask_roi.shape[1] != roi_bgr.shape[1]):
                mask_roi = cv2.resize(mask_roi, (roi_bgr.shape[1], roi_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
            roi_bgr = cv2.bitwise_and(roi_bgr, roi_bgr, mask=mask_roi.astype(np.uint8))

    if liquid_mask is not None:
        liquid_roi = liquid_mask[y1:y2, x1:x2]
        if liquid_roi.size > 0:
            if (liquid_roi.shape[0] != roi_bgr.shape[0]) or (liquid_roi.shape[1] != roi_bgr.shape[1]):
                liquid_roi = cv2.resize(liquid_roi, (roi_bgr.shape[1], roi_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
            state["has_liquid"] = bool(np.count_nonzero(liquid_roi) > 0)

    sobel_gray = imageProc.sobel_y_gray(
        roi_bgr,
        clahe_clip=CLAHE_CLIP,
        clahe_tile=CLAHE_TILE,
        sobel_ksize=SOBEL_KSIZE,
    )
    H, W = sobel_gray.shape[:2]
    band_w = int(W * center_ratio)
    band_w = max(1, min(W, band_w))
    left = (W - band_w) // 2
    right = left + band_w

    sobel_band = sobel_gray[:, left:right]
    prof = sobel_band.astype(np.float32).mean(axis=1)
    prof = imageProc.smooth_1d(prof, PROFILE_SMOOTH_K)

    p = prof.astype(np.float32)
    if p.shape[0] >= 3:
        mid = p[1:-1]
        left_nb = p[:-2]
        right_nb = p[2:]
        is_peak = (mid > left_nb) & (mid >= right_nb)
        peak_idx_all = np.where(is_peak)[0] + 1
    else:
        peak_idx_all = np.array([], dtype=np.int32)

    peak_vals_all = p[peak_idx_all] if peak_idx_all.size > 0 else np.array([], dtype=np.float32)
    keep = peak_vals_all > PEAK_TH
    peak_idx = peak_idx_all[keep]
    peak_vals = peak_vals_all[keep]

    y0 = None
    if state["has_liquid"] and peak_idx.size > 0:
        y0 = int(np.max(peak_idx))

    coeff = None
    inliers_count = 0
    x_vertex = None
    y_vertex = None
    y_center = None

    if state["has_liquid"] and (y0 is not None):
        y_lo = max(0, y0 - RANSAC_Y_WIN)
        y_hi = min(H - 1, y0 + RANSAC_Y_WIN)

        sub = sobel_gray[y_lo:y_hi + 1, left:right]
        ys, xs = np.where(sub >= RANSAC_EDGE_TH)
        xs = xs.astype(np.int32) + left
        ys = ys.astype(np.int32) + y_lo
        pts = np.stack([xs, ys], axis=1)

        coeff, inliers = imageProc.ransac_quadratic(
            pts,
            n_iter=RANSAC_NITER,
            inlier_th=RANSAC_INLIER_TH,
            min_inliers=RANSAC_MIN_INLIERS,
            seed=None,
            prefer_concave_up=True,
            fallback_allow_any=True
        )

        if coeff is not None:
            a, b, c = coeff
            if abs(a) > 1e-9:
                x_vertex = float(-b / (2.0 * a))
                y_vertex = float(a * x_vertex * x_vertex + b * x_vertex + c)
            x_center_roi = float((W - 1) * 0.5)
            y_center = float(a * x_center_roi * x_center_roi + b * x_center_roi + c)
            if inliers is not None:
                inliers_count = int(np.sum(inliers))

    state.update({
        "sobel_gray": sobel_gray,
        "H": H,
        "W": W,
        "left": left,
        "right": right,
        "prof": prof,
        "peak_idx": peak_idx,
        "peak_vals": peak_vals,
        "y0": y0,
        "y_vertex": y_vertex,
        "y_center": y_center,
        "x_vertex": x_vertex,
        "inliers_count": inliers_count,
        "coeff": coeff,
    })
    return state


def render_profile_panels(profile_state, imageProc: ImageProcessor):
    if (profile_state is None) or (not profile_state["valid_roi"]):
        return make_blank_profile_panels()

    roi = profile_state["roi"]
    x1, y1, x2, y2, conf, cls = roi
    sobel_gray = profile_state["sobel_gray"]
    H = profile_state["H"]
    W = profile_state["W"]
    left = profile_state["left"]
    right = profile_state["right"]
    prof = profile_state["prof"]
    peak_idx = profile_state["peak_idx"]
    has_peak = peak_idx.size > 0
    has_liquid = profile_state["has_liquid"]
    y0 = profile_state["y0"]
    coeff = profile_state["coeff"]

    plot = imageProc.render_profile_plot_rightward(
        prof,
        width=W,
        height=H,
        x_max=PROFILE_X_MAX
    )
    x_ref_px = int(round((float(PEAK_TH) / max(1e-6, float(PROFILE_X_MAX))) * (W - 1)))
    x_ref_px = int(np.clip(x_ref_px, 0, W - 1))
    cv2.line(plot, (x_ref_px, 0), (x_ref_px, H - 1), (60, 60, 60), 1)

    sobel_left = cv2.cvtColor(sobel_gray, cv2.COLOR_GRAY2BGR)
    cv2.line(sobel_left, (left, 0), (left, H - 1), (255, 255, 255), 1)
    cv2.line(sobel_left, (right - 1, 0), (right - 1, H - 1), (255, 255, 255), 1)
    cv2.putText(sobel_left, f"cls={cls} conf={conf:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(sobel_left, f"X band: [{left}:{right}) / W={W}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    sobel_peaks = cv2.cvtColor(sobel_gray, cv2.COLOR_GRAY2BGR)
    x_center = W // 2
    cv2.line(sobel_peaks, (x_center, 0), (x_center, H - 1), (80, 80, 80), 1)
    if has_peak:
        for yi in peak_idx:
            y_dot = int(round(int(yi) * (H - 1) / max(1, (H - 1))))
            cv2.circle(sobel_peaks, (x_center, y_dot), 6, (0, 0, 255), -1)
    if not has_liquid:
        cv2.putText(sobel_peaks, "No liquid detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    sobel_ransac = cv2.cvtColor(sobel_gray, cv2.COLOR_GRAY2BGR)
    if has_liquid and (y0 is not None):
        y_lo = max(0, y0 - RANSAC_Y_WIN)
        y_hi = min(H - 1, y0 + RANSAC_Y_WIN)
        cv2.rectangle(sobel_ransac, (left, y_lo), (right - 1, y_hi), (255, 255, 0), 1)
    else:
        cv2.putText(sobel_ransac, "No liquid -> skip RANSAC", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    if coeff is not None:
        a, b, c = coeff
        xs_curve = np.arange(0, W, dtype=np.float32)
        ys_curve = a * xs_curve * xs_curve + b * xs_curve + c
        pts_curve = []
        for xx, yy in zip(xs_curve, ys_curve):
            yy_i = int(round(float(yy)))
            xx_i = int(round(float(xx)))
            if 0 <= yy_i < H:
                pts_curve.append([xx_i, yy_i])
        if len(pts_curve) >= 2:
            pts_curve = np.array(pts_curve, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(sobel_ransac, [pts_curve], False, (0, 0, 255), 2)

    panels = {
        "sobel_left": sobel_left,
        "plot": plot,
        "sobel_peaks": sobel_peaks,
        "sobel_ransac": sobel_ransac,
    }
    panels["combo"] = np.hstack((sobel_left, plot, sobel_peaks, sobel_ransac))
    return panels


def ml_from_bottom_gap_px(gap_px):
    """
    Temporary test mapping:
      bottle bottom to liquid surface gap in pixels -> ml
    Uses piecewise-linear interpolation from config.BOTTON_GAP_PX_TO_ML.
    """
    if gap_px is None:
        return None

    pairs = [(float(px), float(ml)) for px, ml in BOTTOM_GAP_PX_TO_ML]
    if len(pairs) < 2:
        return None

    xs = np.asarray([p[0] for p in pairs], dtype=np.float32)
    ys = np.asarray([p[1] for p in pairs], dtype=np.float32)
    x = float(gap_px)

    if x <= xs[0]:
        return float(ys[0])
    if x >= xs[-1]:
        return float(ys[-1])

    idx = int(np.searchsorted(xs, x) - 1)
    idx = int(np.clip(idx, 0, len(xs) - 2))
    x0, x1 = float(xs[idx]), float(xs[idx + 1])
    y0, y1 = float(ys[idx]), float(ys[idx + 1])

    if abs(x1 - x0) < 1e-9:
        return y0
    t = (x - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)


def main():
    print("[Keys] '1'=2D, '2'=3D, '3'=LINE, '4'=PROFILE, 'q'/ESC=quit")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    align = rs.align(rs.stream.color)

    yolo = YoloRoi(YOLO_WEIGHTS, conf=YOLO_CONF, padding=ROI_PADDING, target_class=None)
    imageProc = ImageProcessor()
    telemetry = TelemetryTable(csv_path=TELEMETRY_CSV, refresh_hz=TELEMETRY_HZ)

    # aruco = ArucoPoseEstimator(
    #     marker_length_m=0.020,
    #     world_ids=(),
    #     cup_id=0
    # )
    
    ws_client = SurfaceWSClient(
        uri="ws://localhost:9090",
        send_hz=10.0,
        only_send_on_change=True,
        change_eps=0.05,
    )
    ws_client.start()
    
    # color_profile = profile.get_stream(rs.stream.color)
    # intr = color_profile.as_video_stream_profile().get_intrinsics()
    # aruco.set_camera_intrinsics(intr)

    view = "2d"  # "2d" / "line" / "profile"

    mls = []
    avg_ml = None
    panels = make_blank_profile_panels()
    
    try:
        while True:
            # ---------------------------
            # Grab one frame (single-loop architecture)
            # ---------------------------
            try:
                frames = pipeline.wait_for_frames(timeout_ms=FRAME_TIMEOUT_MS)
            except RuntimeError:
                continue
            frames = align.process(frames)

            cf = frames.get_color_frame()
            df = frames.get_depth_frame()
            if not cf:
                continue

            color = np.asanyarray(cf.get_data())  # BGR

            # ---------------------------
            # (A) Background: YOLO ROI (always)
            # ---------------------------
            roi = None
            roi_mask = None
            liquid_mask = None
            result = yolo.infer(color)
            roi, roi_mask, liquid_mask = yolo.extract_bottle_and_liquid(result, color.shape)

            # cup_roi_xyxy = None
            # if roi is not None:
            #     x1, y1, x2, y2, conf, cls = roi
            #     cup_roi_xyxy = (x1, y1, x2, y2)
            #
            # ---------------------------
            # (B) Background: ArUco update (disabled for quick ml test)
            # ---------------------------
            # aruco.update(color, cup_roi_xyxy=cup_roi_xyxy)

            # ---------------------------
            # (C) Background: PROFILE/RANSAC (always)
            # (still runs even if you are viewing 2D/LINE)
            # ---------------------------
            profile_state = compute_profile_state(
                color, roi, imageProc, roi_mask=roi_mask, liquid_mask=liquid_mask
            )
            y_vertex = profile_state["y_vertex"]
            y_center = profile_state["y_center"]
            x_vertex = profile_state["x_vertex"]
            inliers_count = profile_state["inliers_count"]
            coeff = profile_state["coeff"]

            # ---------------------------
            # (D) Background: liquid surface y -> bottle-bottom gap_px -> ml
            # ---------------------------
            gap_px = None
            ml = None
            has_liquid = (liquid_mask is not None) and (np.count_nonzero(liquid_mask) > 0)
            if has_liquid and (roi is not None) and (y_vertex is not None):
                x1, y1, x2, y2, conf, cls = roi
                y_full_px = float(y1) + float(y_vertex)
                gap_px = max(0.0, float(y2) - y_full_px)
                ml = ml_from_bottom_gap_px(gap_px)
            else:
                # No liquid detection: do not send values, and clear averaging buffer.
                mls.clear()
                avg_ml = None
                ws_client.set_ml(None)
            
            # ws_client.set_ml(ml)
            # ml is the surface height 

            # 2026/2/9: Modify the original messaging mechanism. Instead of sending message in every iteration, we 
            # take every 20 readings and take the average to smooth out the noises. The average is then sent to C++ client.
            if ml is not None:
                mls.append(ml)
            
            if len(mls) >= 20:
                avg_ml = sum(mls) / len(mls)
                mls.clear()
                ws_client.set_ml(avg_ml)
            

            telemetry.update(
                y_vertex=y_vertex,
                y_center=y_center,
                inliers_count=inliers_count,
                gap_px=gap_px,
                ml=ml,
                avg_ml=avg_ml,
                x_vertex=x_vertex,
                coeff=coeff,
                write_csv=(coeff is not None and gap_px is not None),
            )
            telemetry.maybe_render()

            # ---------------------------
            # (E) Key handling (view only, not trigger compute)
            # ---------------------------
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            if key == ord('1'):
                view = "2d"
            elif key == ord('3'):
                view = "line"
            elif key == ord('4'):
                view = "profile"
            # elif key == ord('5'):
            #     view = "aruco"
            elif key == ord('2'):
                out = show_3d_loop(pipeline, align)
                if out == "to2d":
                    view = "2d"

            # ---------------------------
            # (F) Display only
            # ---------------------------
            if view == "2d":
                rgb = color.copy()
                cv2.putText(rgb, "RGB", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv2.imshow("2D View (RGB)", rgb)

            elif view == "line":
                vis1 = color.copy()
                if liquid_mask is not None:
                    # Overlay liquid segmentation on original image (semi-transparent blue).
                    overlay = np.zeros_like(vis1)
                    overlay[:, :, 0] = 255
                    m3 = np.repeat((liquid_mask > 0)[:, :, None], 3, axis=2)
                    vis1[m3] = cv2.addWeighted(vis1, 0.55, overlay, 0.45, 0)[m3]
                    cv2.putText(vis1, "Liquid seg: ON", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
                else:
                    cv2.putText(vis1, "Liquid seg: none", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 255), 2)
                if roi is None:
                    vis2 = np.zeros_like(color)
                    vis3 = np.zeros_like(color)
                    cv2.putText(vis1, "YOLO: no detection", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                else:
                    x1, y1, x2, y2, conf, cls = roi
                    cv2.rectangle(vis1, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(vis1, f"cls={cls} conf={conf:.2f}", (x1, max(20, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    line_input = color
                    if roi_mask is not None:
                        line_input = cv2.bitwise_and(color, color, mask=roi_mask.astype(np.uint8))
                    vis2, vis3 = imageProc.apply_on_roi_fullframe(line_input, roi)

                cv2.putText(vis2, "Gray (ROI only)", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.putText(vis3, "Sobel(Y) (ROI only)", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                vis = np.hstack((vis1, vis2, vis3))
                cv2.imshow("LINE+YOLO", vis)

            elif view == "profile":
                panels = render_profile_panels(profile_state, imageProc)
                cv2.imshow("PROFILE (Sobel ROI | 1D | Peaks | RANSAC)", panels["combo"])

            # elif view == "aruco":
            #     vis = color.copy()
            #     vis = aruco.draw_overlay(vis, draw_axes=True)
            #     if ml is not None and gap_px is not None:
            #         cv2.putText(vis, f"gap_px={gap_px:.1f}  ml={ml:.1f}", (10, 120),
            #                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            #     cv2.imshow("ARUCO POSE", vis)

    finally:
        try:
            ws_client.stop()
        except Exception:
            pass
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
