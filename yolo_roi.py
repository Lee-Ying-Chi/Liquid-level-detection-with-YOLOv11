import numpy as np
import cv2
from ultralytics import YOLO

class YoloRoi:
    def __init__(self, weights, conf=0.7, padding=10, target_class=None):
        self.model = YOLO(weights)
        self.conf = conf
        self.padding = padding
        self.target_class = target_class

    def infer(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        r = self.model.predict(rgb, conf=self.conf, verbose=False)[0]
        return r

    @staticmethod
    def _mask_to_u8(mask_data, h, w, thr=0.5):
        m = mask_data.detach().cpu().numpy()
        if m.shape[0] != h or m.shape[1] != w:
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        return (m > float(thr)).astype(np.uint8) * 255

    @staticmethod
    def _label_ids(result, label_name):
        names = result.names if hasattr(result, "names") else None
        if names is None:
            return []

        label_l = str(label_name).strip().lower()
        target_ids = []
        if isinstance(names, dict):
            for k, v in names.items():
                if str(v).strip().lower() == label_l:
                    target_ids.append(int(k))
        else:
            for i, v in enumerate(names):
                if str(v).strip().lower() == label_l:
                    target_ids.append(int(i))
        return target_ids

    def pick_roi(self, result, frame_shape):
        h, w = frame_shape[:2]
        if result.boxes is None or len(result.boxes) == 0:
            return None

        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        clss = result.boxes.cls.cpu().numpy().astype(int)

        idxs = np.arange(len(confs))
        if self.target_class is not None:
            idxs = idxs[clss == int(self.target_class)]
            if len(idxs) == 0:
                return None

        best_i = idxs[np.argmax(confs[idxs])]
        x1, y1, x2, y2 = boxes_xyxy[best_i]

        x1 = max(0, int(x1) - self.padding)
        y1 = max(0, int(y1) - self.padding)
        x2 = min(w - 1, int(x2) + self.padding)
        y2 = min(h - 1, int(y2) + self.padding)

        return (x1, y1, x2, y2, float(confs[best_i]), int(clss[best_i]))

    def pick_roi_by_label(self, result, frame_shape, label_name):
        h, w = frame_shape[:2]
        if result.boxes is None or len(result.boxes) == 0:
            return None

        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        clss = result.boxes.cls.cpu().numpy().astype(int)

        target_ids = self._label_ids(result, label_name)
        if len(target_ids) == 0:
            return None

        idxs = np.where(np.isin(clss, np.asarray(target_ids, dtype=np.int32)))[0]
        if len(idxs) == 0:
            return None

        best_i = idxs[np.argmax(confs[idxs])]
        x1, y1, x2, y2 = boxes_xyxy[best_i]

        x1 = max(0, int(x1) - self.padding)
        y1 = max(0, int(y1) - self.padding)
        x2 = min(w - 1, int(x2) + self.padding)
        y2 = min(h - 1, int(y2) + self.padding)

        return (x1, y1, x2, y2, float(confs[best_i]), int(clss[best_i]))

    def pick_roi_and_mask(self, result, frame_shape):
        """
        Returns:
          roi: (x1,y1,x2,y2,conf,cls) or None
          seg_mask_u8: full-frame uint8 mask in {0,255}, or None if model has no masks
        """
        roi = self.pick_roi(result, frame_shape)
        if roi is None:
            return None, None

        h, w = frame_shape[:2]
        seg_mask_u8 = None

        if (result.masks is not None) and (result.masks.data is not None):
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            clss = result.boxes.cls.cpu().numpy().astype(int)

            idxs = np.arange(len(confs))
            if self.target_class is not None:
                idxs = idxs[clss == int(self.target_class)]
                if len(idxs) == 0:
                    return roi, None

            best_i = idxs[np.argmax(confs[idxs])]
            if best_i < len(result.masks.data):
                m = result.masks.data[best_i].detach().cpu().numpy()
                if m.shape[0] != h or m.shape[1] != w:
                    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                seg_mask_u8 = (m > 0.5).astype(np.uint8) * 255

        return roi, seg_mask_u8

    def mask_by_label(self, result, frame_shape, label_name, thr=0.5):
        """
        Return merged full-frame mask (uint8 0/255) for a given class label name.
        If no such class or no segmentation exists, return None.
        """
        h, w = frame_shape[:2]
        if (result.boxes is None) or (len(result.boxes) == 0):
            return None
        if (result.masks is None) or (result.masks.data is None):
            return None

        target_ids = self._label_ids(result, label_name)
        if len(target_ids) == 0:
            return None

        clss = result.boxes.cls.cpu().numpy().astype(int)
        mask_data = result.masks.data
        if len(mask_data) == 0:
            return None

        out = np.zeros((h, w), dtype=np.uint8)
        for i, cls_id in enumerate(clss):
            if cls_id not in target_ids:
                continue
            if i >= len(mask_data):
                continue
            m = mask_data[i].detach().cpu().numpy()
            if m.shape[0] != h or m.shape[1] != w:
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            out = np.maximum(out, ((m > float(thr)).astype(np.uint8) * 255))

        if np.count_nonzero(out) == 0:
            return None
        return out

    def union_mask_by_labels(self, result, frame_shape, label_names, thr=0.5):
        out = np.zeros(frame_shape[:2], dtype=np.uint8)
        has_any = False
        for label_name in label_names:
            m = self.mask_by_label(result, frame_shape, label_name, thr=thr)
            if m is None:
                continue
            out = np.maximum(out, m.astype(np.uint8))
            has_any = True
        if not has_any or np.count_nonzero(out) == 0:
            return None
        return out

    def extract_bottle_and_liquid(self, result, frame_shape, mask_thr=0.5):
        """
        Single-pass parse of a YOLO result.
        Returns:
          bottle_roi: (x1,y1,x2,y2,conf,cls) or None
          bottle_mask: full-frame uint8 mask or None
          liquid_mask: full-frame uint8 mask or None
        """
        h, w = frame_shape[:2]
        if result.boxes is None or len(result.boxes) == 0:
            return None, None, None

        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        clss = result.boxes.cls.cpu().numpy().astype(int)
        mask_data = result.masks.data if (result.masks is not None and result.masks.data is not None) else None

        bottle_ids = self._label_ids(result, "bottle")
        liquid_ids = self._label_ids(result, "Liquid")

        bottle_best_i = None
        if len(bottle_ids) > 0:
            bottle_idxs = np.where(np.isin(clss, np.asarray(bottle_ids, dtype=np.int32)))[0]
            if len(bottle_idxs) > 0:
                bottle_best_i = int(bottle_idxs[np.argmax(confs[bottle_idxs])])

        bottle_roi = None
        bottle_mask = None
        if bottle_best_i is not None:
            x1, y1, x2, y2 = boxes_xyxy[bottle_best_i]
            x1 = max(0, int(x1) - self.padding)
            y1 = max(0, int(y1) - self.padding)
            x2 = min(w - 1, int(x2) + self.padding)
            y2 = min(h - 1, int(y2) + self.padding)
            bottle_roi = (x1, y1, x2, y2, float(confs[bottle_best_i]), int(clss[bottle_best_i]))

            if (mask_data is not None) and (bottle_best_i < len(mask_data)):
                bottle_mask = self._mask_to_u8(mask_data[bottle_best_i], h, w, thr=mask_thr)
                if np.count_nonzero(bottle_mask) == 0:
                    bottle_mask = None

        liquid_mask = None
        if (mask_data is not None) and (len(liquid_ids) > 0):
            out = np.zeros((h, w), dtype=np.uint8)
            for i, cls_id in enumerate(clss):
                if cls_id not in liquid_ids:
                    continue
                if i >= len(mask_data):
                    continue
                out = np.maximum(out, self._mask_to_u8(mask_data[i], h, w, thr=mask_thr))
            if np.count_nonzero(out) > 0:
                liquid_mask = out

        return bottle_roi, bottle_mask, liquid_mask
