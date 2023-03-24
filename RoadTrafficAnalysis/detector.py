from dataclasses import dataclass
import cv2

from VehiclesDetector.sortalg import *


@dataclass
class DetectionData:
    x: int
    y: int
    w: int
    h: int
    class_name: str
    detections_conf: float
    color: list


@dataclass
class Detector:
    weights_file_path: str
    config_file_path: str
    classes_file_path: str
    image_width: int = 416
    image_height: int = 416
    confidence_threshold: float = 0.3
    nms_threshold: float = 0.3

    def __post_init__(self):
        self.net = cv2.dnn.readNet(self.weights_file_path, self.config_file_path)
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        with open(self.classes_file_path) as f:
            self.classes = f.read().splitlines()

        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def detect(self, img: np.array):
        """
        :param img: input img
        :return: list of tuples containing the following data: x, y, w, h, class_name, confidence, class_color
        """
        bbox = []
        class_ids = []
        confs = []

        height, width, _ = img.shape

        blob = cv2.dnn.blobFromImage(img, 1/255, (self.image_width, self.image_height), (0, 0, 0), True, crop=False)

        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                # print(class_id)
                confidence = scores[class_id]

                if confidence > self.confidence_threshold:
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int((detection[0] * width) - w/2)
                    y = int((detection[1] * height) - h/2)

                    bbox.append([x, y, w, h])
                    class_ids.append(class_id)
                    confs.append(float(confidence))

        indexes = cv2.dnn.NMSBoxes(bbox, confs, self.confidence_threshold, self.nms_threshold)

        detections_list = []
        for i in indexes:
            i = i[0]

            box = bbox[i]
            x, y, w, h = box
            class_name = self.classes[class_ids[i]].capitalize()
            conf = confs[i]
            class_color = [int(c) for c in self.colors[class_ids[i]]]

            detections_list.append(DetectionData(x, y, w, h, class_name, conf, class_color))

        return detections_list


if __name__ == '__main__':
    detector = Detector(
        weights_file_path=r"RoadTrafficAnalysis/models/yolov3_training_final.weights",
        config_file_path=r"RoadTrafficAnalysis/models/yolov3_testing.cfg",
        classes_file_path=r"RoadTrafficAnalysis/models/classes.txt",
        confidence_threshold=.1,
        nms_threshold=.1
    )

    image_mode = False
    if image_mode:
        img = cv2.imread("medias/3.jpg")

        detections = detector.detect(img)
        for detection in detections:
            x1, y1 = detection.x, detection.y
            x2, y2 = detection.x + detection.w, detection.y + detection.h

            cv2.rectangle(img, (x1, y1), (x2, y2), detection.color, 2)
            cv2.putText(img, f"{detection.class_name} {int(round(detection.detections_conf, 2) * 100)}%",
                        (x1, y1 - 15), cv2.FONT_HERSHEY_PLAIN, 1.5, detection.color, 2)

        cv2.imshow("resImage", img)
        cv2.waitKey(0)
    else:
        cap = cv2.VideoCapture(r"medias\testVideos\vid2.mp4")
        p_time = 0

        mot_tracker = Sort(max_age=20)

        while cap.isOpened():
            success, frame = cap.read()
            detectionsar = np.empty((0, 5))

            detections = detector.detect(frame)

            for detection in detections:
                x1, y1 = detection.x, detection.y
                x2, y2 = detection.x + detection.w, detection.y + detection.h

                curr_arr = np.array([x1, y1, x2, y2, detection.detections_conf])
                detectionsar = np.vstack((detectionsar, curr_arr))

            restr = mot_tracker.update(detectionsar)

            for d in restr:
                x1, y1, x2, y2, idd = d
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0 ,255), 2)
                cv2.putText(frame, f"{idd}",
                            (x1, y1 - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255,0 ,255), 2)
            #
            # c_time = time()
            # fps = int(1 / (c_time - p_time))
            # p_time = c_time
            #
            # cv2.putText(frame, f"FPS: {fps}", (10, 35), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

            cv2.imshow("res", frame)
            key = cv2.waitKey(1)
            if key == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
