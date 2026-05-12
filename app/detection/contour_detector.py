import cv2
import numpy as np


class ContourDetector:

    def __init__(
        self,
        min_area=5000
    ):

        self.min_area = min_area

    # =====================================================
    # MAIN ROI DETECTION
    # =====================================================

    def detect(self, frame):

        original = frame.copy()

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_RGB2GRAY
        )

        # =====================================================
        # PREPROCESSING
        # =====================================================

        blurred = cv2.GaussianBlur(
            gray,
            (5, 5),
            0
        )

        edges = cv2.Canny(
            blurred,
            50,
            150
        )

        kernel = np.ones((5, 5), np.uint8)

        edges = cv2.dilate(
            edges,
            kernel,
            iterations=2
        )

        # =====================================================
        # FIND CONTOURS
        # =====================================================

        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:
            return None

        # =====================================================
        # SMART CONTOUR FILTERING
        # =====================================================

        best_contour = None

        best_score = 0

        frame_area = (
            frame.shape[0] *
            frame.shape[1]
        )

        for contour in contours:

            area = cv2.contourArea(contour)

            # Skip tiny contours
            if area < self.min_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            # Prevent division by zero
            if w == 0:
                continue

            # Aspect ratio
            aspect_ratio = h / float(w)

            # Reject wide background objects
            if aspect_ratio < 1.0:
                continue

            # Reject giant contours
            if area > frame_area * 0.7:
                continue

            # Solidity
            hull = cv2.convexHull(contour)

            hull_area = cv2.contourArea(hull)

            if hull_area == 0:
                continue

            solidity = area / float(hull_area)

            # Reject noisy contours
            if solidity < 0.5:
                continue

            # =====================================================
            # CONTOUR SCORE
            # =====================================================

            score = (
                area *
                aspect_ratio *
                solidity
            )

            if score > best_score:

                best_score = score

                best_contour = contour

        # =====================================================
        # NO VALID CONTOUR
        # =====================================================

        if best_contour is None:
            return None

        # =====================================================
        # FINAL BOUNDING BOX
        # =====================================================

        x, y, w, h = cv2.boundingRect(best_contour)

        # Smart padding
        padding = 15

        x1 = max(x - padding, 0)
        y1 = max(y - padding, 0)

        x2 = min(
            x + w + padding,
            original.shape[1]
        )

        y2 = min(
            y + h + padding,
            original.shape[0]
        )

        # ROI crop
        roi = original[
            y1:y2,
            x1:x2
        ]

        # Updated bbox
        x, y, w, h = (
            x1,
            y1,
            x2 - x1,
            y2 - y1
        )

        return {

            "roi": roi,

            "bbox": (x, y, w, h),

            "area": cv2.contourArea(best_contour)
        }