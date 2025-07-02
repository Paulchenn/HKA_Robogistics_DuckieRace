#!/usr/bin/env python3

import cv2
import rospy
import numpy as np
import os
import yaml

from cv_bridge import CvBridge
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Float64MultiArray
from ultralytics import YOLO


class FilteredResults:
    def __init__(self, names):
        self.names = names
        self.boxes = []
        self.nearest = None


class DetectDuckieNode(DTROS):
    def __init__(self, node_name):
        super(DetectDuckieNode, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)
        self._vehicle_name = os.environ['VEHICLE_NAME']
        self.debug = False  # Nur wenn True wird Bild angezeigt

        self._model = YOLO("packages/followlane/assets/model_detectDuckie.pt")

        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        self.sub_image = rospy.Subscriber(self._camera_topic, CompressedImage, self.cbDetectObjects, queue_size=1)

        self.pub_image = rospy.Publisher(f"/{self._vehicle_name}/detect/duckie/image", Image, queue_size=1)
        self.pup_duckieNearestBB = rospy.Publisher(f"/{self._vehicle_name}/detect/duckie/nearestBB", Float64MultiArray, queue_size=1)
        self.pup_duckieNearestBB_right = rospy.Publisher(f"/{self._vehicle_name}/detect/duckie/nearestBBright", Float64MultiArray, queue_size=1)

        with open('packages/followlane/config/detect_duckie.yaml', 'r') as f:
            self.conf = yaml.safe_load(f)

        self._bridge = CvBridge()
        self.frame_count = 0
        self.latest_img = None
        self._window = "Detect Duckie"

    def cbDetectObjects(self, image_msg):
        self.frame_count += 1
        if self.frame_count % 5 != 0:
            return

        cv_image = self._bridge.compressed_imgmsg_to_cv2(image_msg)
        mask_center = np.zeros(cv_image.shape[:2], dtype=np.uint8)
        polygon_center = self.get_polygon()
        cv2.fillPoly(mask_center, [polygon_center], 255)
        masked_center = cv2.bitwise_and(cv_image, cv_image, mask=mask_center)
        cv2.polylines(cv_image, [polygon_center], isClosed=True, color=(0, 255, 0), thickness=2)

        # Rechte Maske
        mask_right = np.zeros(cv_image.shape[:2], dtype=np.uint8)
        polygon_right = self.get_right_polygon()
        cv2.fillPoly(mask_right, [polygon_right], 255)
        masked_right = cv2.bitwise_and(cv_image, cv_image, mask=mask_right)
        cv2.polylines(cv_image, [polygon_right], isClosed=True, color=(255, 255, 0), thickness=2)

        # YOLO ausführen für beide Masken
        results_center = self._model(masked_center, conf=0.5, iou=0.5, agnostic_nms=True, verbose=False)
        results_right = self._model(masked_right, conf=0.5, iou=0.5, agnostic_nms=True, verbose=False)

        filtered_center, nearest_center = self.get_filteredResults_and_nearestDuckie(results_center, mask_center)
        filtered_right, nearest_right = self.get_filteredResults_and_nearestDuckie(results_right, mask_right)

        img_bb = self.draw_bounding_boxes(filtered_center + filtered_right, cv_image)
        self.latest_img = img_bb

        if img_bb is not None:
            img_msg = self._bridge.cv2_to_imgmsg(img_bb, encoding='bgr8')
            self.pub_image.publish(img_msg)

        # Publish nearest duckie (Zentrum)
        if nearest_center is not None:
            x1, y1, x2, y2 = map(int, nearest_center.xyxy[0])
            self.pup_duckieNearestBB.publish(Float64MultiArray(data=[x1, y1, x2, y2]))
        else:
            self.pup_duckieNearestBB.publish(Float64MultiArray(data=[]))

        # Publish nearest duckie (Rechts)
        if nearest_right is not None:
            x1, y1, x2, y2 = map(int, nearest_right.xyxy[0])
            self.pup_duckieNearestBB_right.publish(Float64MultiArray(data=[x1, y1, x2, y2]))
        else:
            self.pup_duckieNearestBB_right.publish(Float64MultiArray(data=[]))

    def get_polygon(self):
        return np.array([
            [self.conf['lane_image']['top_left_x'], self.conf['lane_image']['top_left_y']],
            [self.conf['lane_image']['top_right_x'], self.conf['lane_image']['top_right_y']],
            [self.conf['lane_image']['bottom_right_x'], self.conf['lane_image']['bottom_right_y']],
            [self.conf['lane_image']['bottom_left_x'], self.conf['lane_image']['bottom_left_y']],
        ], dtype=np.int32)

    def get_right_polygon(self, x_offset=150):
        # Versetzt das zentrale Polygon um x_offset Pixel nach rechts
        base = self.get_polygon()
        shifted = base.copy()
        shifted[:, 0] = base[:, 0] + x_offset
        return shifted

    def get_filteredResults_and_nearestDuckie(self, results, mask):
        filtered_results = []
        y_lowest = 0
        nearest = None

        for result in results:
            filtered = FilteredResults(names=result.names)
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                if 0 <= cy < mask.shape[0] and 0 <= cx < mask.shape[1] and mask[cy, cx] == 255:
                    filtered.boxes.append(box)
                    if y2 >= y_lowest:
                        filtered.nearest = box
                        nearest = box
                        y_lowest = y2
            if filtered.boxes:
                filtered_results.append(filtered)

        return filtered_results, nearest

    def draw_bounding_boxes(self, results_list, img):
        for result in results_list:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0]) if hasattr(box, 'cls') else 0
                label = result.names.get(cls_id, "object")
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            if result.nearest is not None:
                box = result.nearest
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0]) if hasattr(box, 'cls') else 0
                label = f"{result.names.get(cls_id, 'object')} nearest"
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        return img

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.debug and self.latest_img is not None:
                cv2.imshow(self._window, self.latest_img)
                cv2.waitKey(1)
            rate.sleep()


if __name__ == '__main__':
    node = DetectDuckieNode(node_name='detect_duckie_node')
    node.run()
