#!/usr/bin/env python3

import cv2
import rospy
import numpy as np
import os
import yaml
import threading

from cv_bridge import CvBridge
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import Twist2DStamped
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Float64, Float64MultiArray
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
        self._vehicle_name = os.environ['VEHICLE_NAME']
        
        self._model = YOLO("packages/followlane/assets/model_detectDuckie.pt")  # YOLO model path

        # Subscriber for camera images
        # The camera topic is constructed using the vehicle name from the environment variable
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        self.sub_image = rospy.Subscriber(self._camera_topic, CompressedImage, self.cbDetectObjects, queue_size=1)

        # Publisher for image with bounding boxes
        self._yolo_topic = f"/{self._vehicle_name}/detect/duckie/image"
        self.pup_image = rospy.Publisher(self._yolo_topic, Image, queue_size=1)

        # Publisher for nearest duckie coordinates
        # Duckie nearest Bounding Box (BB) coordinates (x1, y1, x2, y2)
        self._duckieNearestBB_topic = f"/{self._vehicle_name}/detect/duckie/nearestBB"
        self.pup_duckieNearestBB = rospy.Publisher(self._duckieNearestBB_topic, Float64MultiArray, queue_size=1)

        with open('packages/followlane/config/detect_duckie.yaml', 'r') as f:
            self.conf = yaml.safe_load(f)

        self._bridge = CvBridge()
        self.frame_count = 0

    def cbDetectObjects(
            self,
            image_msg
        ):
        self.frame_count += 1
        if self.frame_count % 5 != 0:  # Nur jedes 5. Bild verarbeiten
            return

        # 1. Convert CompressedImage message to OpenCV image
        #np_arr = np.frombuffer(image_msg.data, np.uint8)
        #cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        cv_image = self._bridge.compressed_imgmsg_to_cv2(image_msg)#, desired_encoding='bgr8')
        

        # 2. Create an empty mask with the same size as the image
        mask = np.zeros(cv_image.shape[:2], dtype=np.uint8)

        # 3. Define polygon points for the area of interest
        polygon = self.get_polygon()

        # 4. Draw the polygon on the mask
        cv2.fillPoly(mask, [polygon], 255)

        # 5. Apply the mask to the image
        masked_image = cv2.bitwise_and(cv_image, cv_image, mask=mask)
        cv2.polylines(cv_image, [polygon], isClosed=True, color=(0, 255, 0), thickness=2)

        # 6. Run YOLO model on the masked image
        results = self._model(masked_image, conf=0.5, iou=0.5, agnostic_nms=True, verbose=False)
        
        # 7. Filter detected duckies inside polygon mask and find nearest one
        filtered_results, nearestDuckie = self.get_filteredResults_and_nearestDuckie(
            results,
            mask,
            filtered_results=[],
            y_lowest=0
        )

        # 8. Draw bounding boxes around detected duckies
        img_bb = self.draw_bounding_boxes(filtered_results, cv_image)
        self.latest_img = img_bb  # Nur das aktuellste Bild speichern

        # 9.1. Publish the image with bounding boxes
        if img_bb is not None:
            img_msg = self._bridge.cv2_to_imgmsg(img_bb, encoding='bgr8')
            self.pup_image.publish(img_msg)
        # 9.2. Publish the nearest duckie bounding box coordinates
        if nearestDuckie is not None:
            x1, y1, x2, y2 = map(int, nearestDuckie.xyxy[0])
            nearest_bb_msg = Float64MultiArray(data=[x1, y1, x2, y2])
            self.pup_duckieNearestBB.publish(nearest_bb_msg)


    def get_polygon(self):
        return np.array([
            [self.conf['lane_image']['top_left_x'], self.conf['lane_image']['top_left_y']],
            [self.conf['lane_image']['top_right_x'], self.conf['lane_image']['top_right_y']],
            [self.conf['lane_image']['bottom_right_x'], self.conf['lane_image']['bottom_right_y']],
            [self.conf['lane_image']['bottom_left_x'], self.conf['lane_image']['bottom_left_y']],
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

    def display_loop(self):
        while True:
            if self.latest_img is not None:
                cv2.imshow(self._window, self.latest_img)
                cv2.waitKey(1)


if __name__ == '__main__':
    node = DetectDuckieNode(node_name='detect_duckie_node')
    rospy.spin()
