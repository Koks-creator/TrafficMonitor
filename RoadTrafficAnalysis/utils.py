from dataclasses import dataclass
from collections import deque
from datetime import datetime
from math import hypot
from typing import Tuple, Union, List
import cv2
import numpy as np

from RoadTrafficAnalysis.kalman_filter import KalmanFilter


@dataclass
class ObjectData:
    object_id: int
    center: Tuple[int, int]
    bbox: Tuple[int, int, int, int]
    move_history: deque
    start_time: Union[bool, float] = False
    end_time: Union[bool, float] = False
    frame_count: int = 0
    counted: bool = False
    speed: Union[bool, float] = False
    saved: bool = False
    __req_length: int = 5

    def __post_init__(self) -> None:
        self.move_history.append(self.center)
        color_raw = tuple(np.random.choice(range(256), size=3))
        self.color = [int(i) for i in color_raw]
        self.kalman_filter = KalmanFilter()

    @staticmethod
    def get_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        return hypot(p2[0] - p1[0], p2[1] - p1[1])

    def update_move_history(self, center: Tuple[int, int]) -> None:

        if self.get_distance(self.move_history[-1], center) < 40:
            self.move_history.append(center)
        else:
            # https://stackoverflow.com/questions/10003143/how-to-slice-a-deque
            # list comprehension is faster in this case since it small deque, in larger ones use itertools.islice
            start_index = self.move_history.index(self.move_history[-1])+1
            self.move_history = deque([self.move_history[i] for i in range(start_index, len(self.move_history))])
            self.move_history.append(center)

    def draw_move_history(self, image: np.array):
        if self.move_history:
            for i in range(len(self.move_history)):
                if i != len(self.move_history) - 1:
                    cv2.line(image, self.move_history[i], self.move_history[i+1], self.color, 4)

    def check_if_in_area(self, area: list) -> float:
        result = cv2.pointPolygonTest(np.array(area, np.int32), self.center, False)
        return result

    def get_speed(self, distance_in_meters: int, fps: int) -> float:
        a_speed_ms = distance_in_meters / (self.end_time - self.start_time)

        if fps == 0:
            fps = 1

        a_speed_kh = (a_speed_ms * 3.6) * (int(distance_in_meters*2 // fps))
        self.speed = a_speed_kh

        return a_speed_kh

    def control_speed(self, threshold: float, save_dir_path: str = False, image: np.array = False) -> bool:
        if self.speed > threshold:
            self.color = (0, 0, 255)

            if save_dir_path and image is not False and self.saved is False:
                x1, y1, x2, y2 = self.bbox[0], self.bbox[1], self.bbox[2], self.bbox[3]
                car = image[y1:y2, x1:x2]

                filename = f"{datetime.now().strftime('%d%m%Y-%H%M%S')}_{int(self.speed)}.jpg"

                cv2.imwrite(fr"{save_dir_path}\{filename}", car)
                self.saved = True

                return True
        return False

    def draw_prediction_pointer(self, image: np.array):
        points_history = self.move_history

        for index, ppoint in enumerate(points_history):
            # Predictions on previous points
            first_prediction = self.kalman_filter.Estimate(ppoint[0], ppoint[1])
            if index == len(points_history) - 1:
                p_prediction = first_prediction

                # Drawing extended prediction line
                prediction_line_points = []
                for i in range(self.__req_length):
                    new_prediction = self.kalman_filter.Estimate(p_prediction[0], p_prediction[1])
                    if len(points_history) > self.__req_length + 1:
                        cv2.line(image, p_prediction, new_prediction, (255, 255, 120), 3)

                    prediction_line_points.append(new_prediction)
                    p_prediction = new_prediction

                if len(points_history) > self.__req_length + 1:
                    # Pointer
                    pointer_p1 = prediction_line_points[-2][0] - 8, prediction_line_points[-2][1]
                    pointer_p2 = prediction_line_points[-2][0] + 8, prediction_line_points[-2][1]
                    cv2.line(image, prediction_line_points[-1], pointer_p1, (255, 0, 255), 3)
                    cv2.line(image, prediction_line_points[-1], pointer_p2, (255, 0, 255), 3)
                else:
                    cv2.putText(image, f"Making pointer...", (self.bbox[0], self.bbox[1]-5), cv2.FONT_HERSHEY_PLAIN, 0.8, self.color, 1)


@dataclass
class Area:
    name: str
    area_points_list: List[Tuple[int, int]]
    colors_list: List[Tuple[int, int, int]]
    state = 0

    def draw_area(self, image: np.array):
        m = cv2.moments(np.array(self.area_points_list, np.int32))
        cv2.fillPoly(image, [np.array(self.area_points_list, np.int32)], color=self.colors_list[self.state])

        if m['m00'] != 0:
            cx = int(m['m10'] / m['m00'])
            cy = int(m['m01'] / m['m00'])

            cv2.putText(image, self.name, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)
