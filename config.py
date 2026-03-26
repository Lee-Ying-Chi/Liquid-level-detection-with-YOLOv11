# ---- 2D Depth visualization range (meters) ----
MIN_D = 0.1
MAX_D = 3.0

# ---- YOLO ----
YOLO_WEIGHTS = "./Yolo/greenBottle0319/best.pt"
YOLO_CONF = 0.6
ROI_PADDING = 10

# ---- Line detect params ----
CLAHE_CLIP = 2.0
CLAHE_TILE = (8, 8)
SOBEL_KSIZE = 3

# ---- RealSense wait_for_frames timeout (ms) ----
FRAME_TIMEOUT_MS = 15000

# ---- 1D profile display ----
PROFILE_SMOOTH_K = 9      # odd number; 1 disables smoothing
PROFILE_PLOT_W = 360
PROFILE_PLOT_H = 480

# ---- PROFILE fixed X scale (do NOT auto-normalize by max per frame) ----
PROFILE_X_REF = 25.0        # the reference value to draw as a vertical line
PROFILE_X_MAX = 60.0        # fixed display max; makes x=25 sit at center (25/50=0.5)

# ---- 範圍控制 ----
center_ratio = 0.80
PEAK_TH = 20.0       # 1D profile 峰值門檻
TOP2_PEAK_DY_SWITCH = 50  # top-2 peaks: if |dy| > this, choose upper peak; else choose lower peak
TOPK = 4

# 用 ROI 高度的上下 margin 近似「杯口~杯底」範圍
    # 例如 top 12% 當作杯口附近，bottom 8% 當作杯底附近，排除它們，只在中間找液面
CUP_TOP_MARGIN_RATIO = 0.12
CUP_BOTTOM_MARGIN_RATIO = 0.12

# RANSAC params
RANSAC_Y_WIN = 13          # y 初始原點範圍
RANSAC_EDGE_TH = 100        # 候選邊緣點門檻(0-255)
RANSAC_NITER = 250         # RANSAC 迭代次數
RANSAC_INLIER_TH = 2.5     # 殘差容忍，越大通常 inlier 越多
RANSAC_MIN_INLIERS = 40    # 最小 inlier 數量，否則放棄本次結果

# ---- Temporary ml mapping for quick testing (bottle bottom -> liquid surface gap in pixels) ----
# gap_px = bottle_bbox_bottom_y - liquid_surface_y
# px -> ml
BOTTOM_GAP_PX_TO_ML = [
    (0.0, 0.0),
    (34.0, 10.0),
    (42.0, 20.0),
    (50.0, 30.0),
    (59.0, 40.0),
    (67.0, 50.0),
    (76.0, 60.0),
    (83.0, 70.0),
    (104.0, 100.0),
    (140.0, 150.0),
    (178.0, 200.0),
    (218.0, 250.0),
    (250.0, 300.0),
    (500.0, 350.0),
]

# ---- Telemetry (terminal table + CSV) ----
TELEMETRY_CSV = "logs/ransac_vertex_log.csv"
TELEMETRY_HZ = 10  # refresh rate cap (Hz)
