import cv2 as cv
from ultralytics import YOLO
from utils import calculate_best_count, draw_text_with_background


class CardDetector:
    MODEL_PATH = "yolov8s_playing_cards.pt"
    CARD_MIN_CONFIDENCE = 0.5

    def __init__(self, video_source=0):
        self.model = YOLO(self.MODEL_PATH)
        self.remaining_cards = list(self.model.names.values())
        self.cap = cv.VideoCapture(video_source)

    def process_detections(self, frame):
        predictions_parsed = {}
        results = self.model(frame)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy.tolist()[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cls_id = int(box.cls)
                cls_conf = box.conf[0]
                cls_name = self.model.names[cls_id]

                if cls_conf < self.CARD_MIN_CONFIDENCE:
                    continue

                if cls_name in self.remaining_cards:
                    self.remaining_cards.remove(cls_name)

                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                width = x2 - x1
                height = y2 - y1

                new_bbox = {
                    'x': center_x,
                    'y': center_y,
                    'width': width,
                    'height': height,
                    'conf': cls_conf
                }

                if cls_name not in predictions_parsed:
                    predictions_parsed[cls_name] = {
                        'bbox': new_bbox
                    }
                else:
                    existing_bbox = predictions_parsed[cls_name]['bbox']
                    merged_bbox = self.merge_bboxes(existing_bbox, new_bbox)
                    predictions_parsed[cls_name]['bbox'] = merged_bbox

        return predictions_parsed

    def merge_bboxes(self, existing_bbox, new_bbox):
        top = min(new_bbox['y'] - new_bbox['height'] / 2, existing_bbox['y'] - existing_bbox['height'] / 2)
        bottom = max(new_bbox['y'] + new_bbox['height'] / 2, existing_bbox['y'] + existing_bbox['height'] / 2)
        left = min(new_bbox['x'] - new_bbox['width'] / 2, existing_bbox['x'] - existing_bbox['width'] / 2)
        right = max(new_bbox['x'] + new_bbox['width'] / 2, existing_bbox['x'] + existing_bbox['width'] / 2)
        max_conf = max(existing_bbox['conf'], new_bbox['conf'])

        merged_bbox = {
            'x': (left + right) / 2,
            'y': (top + bottom) / 2,
            'width': right - left,
            'height': bottom - top,
            'conf': max_conf
        }

        return merged_bbox

    def draw_detections_and_count(self, frame, predictions_parsed):
        detected_cards = []
        for cls_name, bbox_info in predictions_parsed.items():
            bbox = bbox_info['bbox']
            x1 = int(bbox['x'] - bbox['width'] / 2)
            y1 = int(bbox['y'] - bbox['height'] / 2)
            x2 = int(bbox['x'] + bbox['width'] / 2)
            y2 = int(bbox['y'] + bbox['height'] / 2)
            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            conf_text = f"{bbox['conf']:.2f}"
            draw_text_with_background(frame, conf_text, (x1 - 10, y1 - 20))
            detected_cards.append(cls_name)

        count = calculate_best_count(detected_cards)

        count_text = f"Count: {count}"
        draw_text_with_background(frame, count_text, (0, 25))

    def process_video(self):
        frame_counter = 0
        while self.cap.isOpened():
            ret, frame = self.cap.read()

            if frame_counter % 10 == 0:
                predictions_parsed = self.process_detections(frame)
                self.draw_detections_and_count(frame, predictions_parsed)
                print(len(self.remaining_cards))
                cv.imshow("Detections", frame)

            if cv.waitKey(1) == ord('q'):
                break

            frame_counter += 1

        self.cap.release()
        cv.destroyAllWindows()
