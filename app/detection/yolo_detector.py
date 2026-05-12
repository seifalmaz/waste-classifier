from ultralytics import YOLO

import cv2


class YOLODetector:

    def __init__(
        self,
        model_name="yolov8n.pt",
        confidence_threshold=0.25
    ):

        self.model = YOLO(model_name)

        self.conf_threshold = confidence_threshold

        # Only ignore humans
        self.blocked_classes = [

            "person"
        ]

    # =====================================================
    # DETECT OBJECTS
    # =====================================================

    def detect(self, frame):

        results = self.model(
            frame,
            verbose=False
        )

        detections = []

        frame_h, frame_w = frame.shape[:2]

        for result in results:

            boxes = result.boxes

            for box in boxes:

                confidence = float(
                    box.conf[0]
                )

                if confidence < self.conf_threshold:
                    continue

                cls_id = int(box.cls[0])

                label = result.names[cls_id]

                # ==========================================
                # IGNORE HUMANS ONLY
                # ==========================================

                if label in self.blocked_classes:
                    continue

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                w = x2 - x1
                h = y2 - y1

                # ==========================================
                # FILTER VERY TINY OBJECTS ONLY
                # ==========================================

                if w < 20 or h < 20:
                    continue

                # ==========================================
                # SAFE BOUNDS
                # ==========================================

                x1 = max(0, x1)
                y1 = max(0, y1)

                x2 = min(frame_w, x2)
                y2 = min(frame_h, y2)

                roi = frame[
                    y1:y2,
                    x1:x2
                ]

                detections.append({

                    "label": label,

                    "confidence": confidence,

                    "bbox": (
                        x1,
                        y1,
                        w,
                        h
                    ),

                    "roi": roi
                })

        return detections