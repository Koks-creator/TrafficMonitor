from collections import deque
import cv2

from VehiclesDetector.RoadTrafficAnalysis.detector import Detector
from VehiclesDetector.RoadTrafficAnalysis.sortalg import *
from VehiclesDetector.RoadTrafficAnalysis.utils import ObjectData, Area
from VehiclesDetector.RoadTrafficAnalysis.VideoConfigs import video_config
from VehiclesDetector.RoadTrafficAnalysis.config import Config

config = Config()
detector = Detector(
    weights_file_path=rf"{config.MODELS_FOLDER}\{config.MODEL_WEIGHTS}",
    config_file_path=rf"{config.MODELS_FOLDER}\{config.MODEL_CONFIG}",
    classes_file_path=rf"{config.MODELS_FOLDER}\{config.MODEL_CLASSES}",
    confidence_threshold=config.CONF_THRESH,
    nms_threshold=config.NMS_THRESH
)
cap = cv2.VideoCapture(rf"{config.VIDEOS_FOLDER}\{config.VIDEO}")
mot_tracker = Sort(max_age=config.MAX_AGE)

start_areas = [Area(area_points_list=area, colors_list=video_config[config.VIDEO]["ColorsList"], name=f"startA{i+1}")
               for i, area in enumerate(video_config[config.VIDEO]["StartAreas"])]

end_areas = [Area(area_points_list=area, colors_list=video_config[config.VIDEO]["ColorsList"], name=f"endA{i+1}")
             for i, area in enumerate(video_config[config.VIDEO]["EndAreas"])]

overspeed_counters = [0 for _ in range(len(video_config[config.VIDEO]["MaxSpeeds"]))]
detected_objects = {}
cars_count = 0
p_time = 0

if not os.path.exists(config.FULL_PATH):
    os.makedirs(config.FULL_PATH)

while cap.isOpened():
    success, img = cap.read()
    img = cv2.resize(img, (1366, 768))
    overlay = img.copy()

    # Info panel
    cv2.rectangle(overlay, (0, 0), config.PANEL_SIZE, (0, 0, 0), -1)

    for start_area in start_areas:
        start_area.draw_area(image=overlay)

    for end_area in end_areas:
        end_area.draw_area(image=overlay)

    final_img = cv2.addWeighted(overlay, config.ALPHA, img, 1 - config.ALPHA, 0)

    detections_ar = np.empty((0, 5))
    detections = detector.detect(img)

    detection_info = []

    c_time = time.time()
    fps = int(1 / (c_time - p_time))
    p_time = c_time

    for detection in detections:
        x1, y1 = detection.x, detection.y
        x2, y2 = detection.x + detection.w, detection.y + detection.h

        curr_arr = np.array([x1, y1, x2, y2, detection.detections_conf])

        detection_info.append((detection.class_name, round(detection.detections_conf, 2), detection.color))
        detections_ar = np.vstack((detections_ar, curr_arr))

    tracker_results = mot_tracker.update(detections_ar)

    current_ids = []
    for res in zip(tracker_results, detection_info):
        tr_res, det_info = res
        class_name, conf, color = det_info

        x1, y1, x2, y2, vehicle_id = tr_res
        vehicle_id = int(vehicle_id)
        current_ids.append(vehicle_id)
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w = x2 - x1
        h = y2 - y1

        cx, cy = x1 + w//2, y1 + h//2

        if vehicle_id not in list(detected_objects.keys()):
            detected_objects[vehicle_id] = ObjectData(
                object_id=vehicle_id,
                center=(cx, cy),
                bbox=(x1, y1, x2, y2),
                move_history=deque(maxlen=config.MAX_HISTORY)
            )
        else:
            detected_objects[vehicle_id].center = (cx, cy)
            detected_objects[vehicle_id].bbox = (x1, y1, x2, y2)
            detected_objects[vehicle_id].update_move_history(center=(cx, cy))
            detected_objects[vehicle_id].frame_count = 0

        detected_objects[vehicle_id].draw_move_history(image=final_img)

        for start_area in start_areas:
            in_start_area = detected_objects[vehicle_id].check_if_in_area(start_area.area_points_list)
            if in_start_area >= 0:
                if not detected_objects[vehicle_id].counted:
                    detected_objects[vehicle_id].counted = True
                    detected_objects[vehicle_id].start_time = time.time()
                    cars_count += 1
                    start_area.state = 1
                else:
                    start_area.state = 0

        for index, end_area in enumerate(end_areas):
            in_end_area = detected_objects[vehicle_id].check_if_in_area(end_area.area_points_list)
            if in_end_area >= 0 and detected_objects[vehicle_id].counted:
                detected_objects[vehicle_id].end_time = time.time()
                end_area.state = 1

                if not detected_objects[vehicle_id].speed:
                    detected_objects[vehicle_id].get_speed(distance_in_meters=video_config[config.VIDEO]["StartEndDistances"][index],
                                                           fps=fps)
                    overspeed = detected_objects[vehicle_id].control_speed(threshold=video_config[config.VIDEO]["MaxSpeeds"][index],
                                                                           save_dir_path=config.FULL_PATH, image=img)
                    if overspeed:
                        overspeed_counters[index] += 1
            else:
                end_area.state = 0

        detected_objects[vehicle_id].draw_prediction_pointer(image=final_img)
        object_color = detected_objects[vehicle_id].color
        cv2.circle(final_img, (cx, cy), 4, object_color, -1)

        if int(detected_objects[vehicle_id].speed) > 0:
            cv2.putText(final_img, f"{int(detected_objects[vehicle_id].speed)} km/h",
                        (x1, y1 - 5), cv2.FONT_HERSHEY_PLAIN, 1, object_color, 2)

        cv2.putText(final_img, f"{vehicle_id}", (cx, cy), cv2.FONT_HERSHEY_PLAIN, 1.5, object_color, 2)
        cv2.rectangle(final_img, (x1, y1), (x2, y2), object_color, 2)

    diff = np.setdiff1d(np.array([list(detected_objects.keys())]), np.array([current_ids]))
    for veh_id in diff:
        detected_objects[veh_id].frame_count += 1

        if detected_objects[veh_id].frame_count >= config.MAX_AGE:
            del detected_objects[veh_id]

    cv2.putText(final_img, f"FPS: {fps}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, config.INTERFACE_COLOR, 2)
    cv2.putText(final_img, f"Counter: {cars_count}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, .8, config.INTERFACE_COLOR, 2)

    start_y = 120
    for index, overspeed_data in enumerate(zip(video_config[config.VIDEO]["MaxSpeeds"], overspeed_counters)):
        max_speed, counter = overspeed_data
        cv2.putText(final_img, f"A{index+1} Limit: {max_speed}km/h", (20, start_y), cv2.FONT_HERSHEY_SIMPLEX, .8, config.INTERFACE_COLOR, 2)
        cv2.putText(final_img, f"A{index+1} Overspeed Counter: {counter}", (20, start_y + 40), cv2.FONT_HERSHEY_SIMPLEX, .8,
                    config.INTERFACE_COLOR, 2)

        start_y += 80

    cv2.rectangle(final_img, (0, 0), (config.PANEL_SIZE[0]+config.PANEL_PADDING,
                                      config.PANEL_SIZE[1]+config.PANEL_PADDING,), config.INTERFACE_COLOR, 2)

    cv2.imshow("res", final_img)
    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
