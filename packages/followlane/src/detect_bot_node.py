#!/usr/bin/env python3

import cv2
import rospy
import numpy as np
import os
import yaml

from cv_bridge import CvBridge
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import Twist2DStamped
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Float64
from ultralytics import YOLO


class FilteredResults:
    def __init__(self, names):
        self.names = names
        self.boxes = []
        self.nearest = None  # Store the nearest box here (not a list)


class DetectDuckieNode(DTROS):
    def __init__(self, node_name):
        # Initialize the DTROS parent class
        super(DetectDuckieNode, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)
        
        self._model = YOLO("packages/followlane/assets/model_detectDuckie.pt")  # YOLO model path

        self._vehicle_name = os.environ['VEHICLE_NAME']
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        self.sub_image = rospy.Subscriber(self._camera_topic, CompressedImage, self.cbDetectObjects)

        self._yolo_topic = f"/{self._vehicle_name}/detect/bot/image"
        self.pup_image = rospy.Publisher(self._yolo_topic, Image, queue_size=1)

        self._duckie_topic = f"/{self._vehicle_name}/detect/bot"
        self.pup_duckie = rospy.Publisher(self._duckie_topic, Float64, queue_size=1)

        twist_topic = f"/{self._vehicle_name}/car_cmd_switch_node/cmd"
        self.pub_cmd_vel = rospy.Publisher(twist_topic, Twist2DStamped, queue_size=1)

        with open('packages/followlane/config/detect_bot.yaml', 'r') as f:
            self.conf = yaml.safe_load(f)

        self._bridge = CvBridge()

        self._window = "bot-detection"

    def cbDetectObjects(
            self,
            image_msg
        ):
        # 1. Convert CompressedImage message to OpenCV image
        # np_arr = np.frombuffer(image_msg.data, np.uint8)
        # cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        cv_image = self._bridge.compressed_imgmsg_to_cv2(image_msg)

        # # 2. Create an empty mask with the same size as the image
        mask = np.zeros(cv_image.shape[:2], dtype=np.uint8)

        # # 3. Define polygon points for the area of interest
        polygon = self.get_polygon()

        # # 4. Draw the polygon on the mask
        cv2.fillPoly(mask, [polygon], 255)

        # # 5. Apply the mask to the image
        masked_image = cv2.bitwise_and(cv_image, cv_image, mask=mask)
        cv2.polylines(cv_image, [polygon], isClosed=True, color=(0, 255, 0), thickness=2)

        # 6. Run YOLO model on the masked image
        results = self._model(cv_image, conf=0.5, iou=0.5, agnostic_nms=True, verbose=False)
        
        # 7. Filter detected duckies inside polygon mask and find nearest one
        filtered_results, nearestDuckie = self.get_filteredResults_and_nearestDuckie(
            results,
            mask,
            filtered_results=[],
            y_lowest=0
        )

        # 8. Draw bounding boxes around detected duckies
        img_bb = self.draw_bounding_boxes(filtered_results, cv_image)

        # 9. Show polygon overlay
        cv2.imshow(self._window, img_bb)
        cv2.waitKey(1)

        # 10. Stop if duckie is nearer than value in config
        if nearestDuckie is not None:
            flag_stopped = self.stop_duckie(nearestDuckie)


    def get_polygon(self):
        return np.array([
            [self.conf['bot_image']['top_left_x'], self.conf['bot_image']['top_left_y']],
            [self.conf['bot_image']['top_right_x'], self.conf['bot_image']['top_right_y']],
            [self.conf['bot_image']['bottom_right_x'], self.conf['bot_image']['bottom_right_y']],
            [self.conf['bot_image']['bottom_left_x'], self.conf['bot_image']['bottom_left_y']],
        ], dtype=np.int32)


    def get_filteredResults_and_nearestDuckie(
            self,
            results,
            mask,
            filtered_results=[],
            y_lowest=0
        ):
        for result in results:
            filtered = FilteredResults(names=result.names)

            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                if mask[cy, cx] == 255:
                    filtered.boxes.append(box)
                    if y2 >= y_lowest:
                        filtered.nearest = box
                        y_lowest = y2

            # Only add if at least one box found
            if filtered.boxes:
                filtered_results.append(filtered)

        return filtered_results, filtered.nearest


    def draw_bounding_boxes(
            self,
            results_list,
            img
        ):
        for result in results_list:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0]) if hasattr(box, 'cls') else 0
                label = result.names.get(cls_id, "object")

                # Draw bounding box in blue
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            # If nearest box exists, highlight it in green
            if result.nearest is not None:
                box = result.nearest

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0]) if hasattr(box, 'cls') else 0
                label = f"{result.names.get(cls_id, 'object')} nearest"

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                if False:
                    print("nearest Duckie:")
                    print(f"    x1: {x1}")
                    print(f"    y1: {y1}")
                    print(f"    x2: {x2}")
                    print(f"    y2: {y2}")

        return img
    

    def stop_duckie(
            self,
            nearestDuckie
    ):
        y_min4stop = self.conf['min_distance_nearestBot']

        x1, y1, x2, y2 = map(int, nearestDuckie.xyxy[0])
        if y2 <= y_min4stop:
            rospy.loginfo("Bot in the way. Sending stop command...")
            stop_msg = Twist2DStamped(v=0.0, omega=0.0)
            
            for _ in range(5):
                rospy.sleep(0.05)
                self.pub_cmd_vel.publish(stop_msg)


if __name__ == '__main__':
    node = DetectDuckieNode(node_name='detect_bot_node')
    rospy.spin()
