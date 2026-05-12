from collections import deque, Counter

import numpy as np


class PredictionSmoother:

    def __init__(
        self,
        history_size=10
    ):

        self.history_size = history_size

        self.class_history = deque(
            maxlen=history_size
        )

        self.conf_history = deque(
            maxlen=history_size
        )

    # =====================================================
    # UPDATE HISTORY
    # =====================================================

    def update(
        self,
        predicted_class,
        confidence
    ):

        self.class_history.append(
            predicted_class
        )

        self.conf_history.append(
            confidence
        )

    # =====================================================
    # GET SMOOTHED RESULT
    # =====================================================

    def get_smoothed_prediction(self):

        if len(self.class_history) == 0:

            return None, 0.0

        # Majority voting
        most_common = Counter(
            self.class_history
        ).most_common(1)[0][0]

        # Average confidence
        avg_conf = np.mean(
            self.conf_history
        )

        return most_common, float(avg_conf)