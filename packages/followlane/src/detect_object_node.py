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


class DetectParkingSlotNode(DTROS):
    def __init__(self, node_name):
        # Initialize the DTROS parent class
        super(DetectParkingSlotNode, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)
        self._vehicle_name = os.environ['VEHICLE_NAME']
        
        self._model = YOLO("packages/followlane/assets/model_detectDuckieBotSlot.pt")  # YOLO model path

        # Subscriber for camera images
        # The camera topic is constructed using the vehicle name from the environment variable
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        self.sub_image = rospy.Subscriber(self._camera_topic, CompressedImage, self.cbDetectObjects, queue_size=1)

        # Publisher for image with bounding boxes
        self._yolo_topic = f"/{self._vehicle_name}/detect/object/image"
        self.pup_image = rospy.Publisher(self._yolo_topic, Image, queue_size=1)

        # Publisher for nearest duckie coordinates
        # Duckie nearest Bounding Box (BB) coordinates (x1, y1, x2, y2)
        self._duckieNearestBB_topic = f"/{self._vehicle_name}/detect/object/duckieNearestBB"
        self.pup_duckieNearestBB = rospy.Publisher(self._duckieNearestBB_topic, Float64MultiArray, queue_size=1)

        # Publisher for nearest bot coordinates
        # Duckie nearest Bounding Box (BB) coordinates (x1, y1, x2, y2)
        self._botNearestBB_topic = f"/{self._vehicle_name}/detect/object/botNearestBB"
        self.pup_botNearestBB = rospy.Publisher(self._botNearestBB_topic, Float64MultiArray, queue_size=1)

        # Publisher for nearest parking coordinates
        # Duckie nearest Bounding Box (BB) coordinates (x1, y1, x2, y2)
        self._parkingBB_topic = f"/{self._vehicle_name}/detect/object/parkingBB"
        self.pup_parkingBB = rospy.Publisher(self._parkingBB_topic, Float64MultiArray, queue_size=1)

        with open('packages/followlane/config/detect_duckieBotSlot.yaml', 'r') as f:
            self.conf = yaml.safe_load(f)

        self._bridge = CvBridge()
        self.frame_count = 0
        

    def cbDetectObjects(self, image_msg):
        self.frame_count += 1
        if self.frame_count % 5 != 0:  # Take only every fifth picture
            return

        # Convert CompressedImage message to OpenCV image
        cv_image = self._bridge.compressed_imgmsg_to_cv2(image_msg)
        
        # Create an empty mask with the same size as the image
        mask = np.zeros(cv_image.shape[:2], dtype=np.uint8)
        # Define polygon points for the area of interest
        polygon = self.get_polygon()
        # Draw the polygon on the mask
        cv2.fillPoly(mask, [polygon], 255)

        # Run YOLO model on the image
        results = self._model(cv_image, conf=0.65, iou=0.5, agnostic_nms=True, verbose=False)

        # Plot results on the image
        annotated_frame = results[0].plot()

        # Publish image mit Bounding Boxes
        if annotated_frame is not None:
            img_msg = self._bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')
            self.pup_image.publish(img_msg)

        # Filter detected duckies inside polygon mask and find nearest one
        filteredResults_duckie, nearest_duckie = self.get_filteredResults_and_nearest(
            results,
            "duckie",
            mask,
            filtered_results=[],
            y_lowest=0
        )
        # Filter detected bots inside polygon mask and find nearest one
        filteredResults_bot, nearest_bot = self.get_filteredResults_and_nearest(
            results,
            "bot",
            mask,
            filtered_results=[],
            y_lowest=0
        )
        # Filter detected free Slots inside polygon mask and find nearest one
        filteredResults_freeSlot, nearest_freeSlot = self.get_filteredResults_and_nearest(
            results,
            "freeSlot",
            mask,
            filtered_results=[],
            y_lowest=0
        )
        # Filter detected bots inside polygon mask and find nearest one
        filteredResults_occSlot, nearest_occSlot = self.get_filteredResults_and_nearest(
            results,
            "occupiedSlot",
            mask,
            filtered_results=[],
            y_lowest=0
        )
        
        # === Publish ===
        if nearest_duckie is not None:
            x1, y1, x2, y2 = map(int, nearest_duckie.xyxy[0])
            msg_duckieNearestBB = Float64MultiArray(data=[x1, y1, x2, y2])
            self.pup_duckieNearestBB.publish(msg_duckieNearestBB)
        if nearest_bot is not None:
            x1, y1, x2, y2 = map(int, nearest_bot.xyxy[0])
            msg_botNearestBB = Float64MultiArray(data=[x1, y1, x2, y2])
            self.pup_botNearestBB.publish(msg_botNearestBB)
        if nearest_freeSlot is not None:
            x1, y1, x2, y2 = map(int, nearest_freeSlot.xyxy[0])
            msg_parkingNearestBB = Float64MultiArray(data=[x1, y1, x2, y2])
            self.pup_parkingBB.publish(msg_parkingNearestBB)
        
        
        if self.conf['debugPrints']:
            print('==========')
            print('Duckies:')
            print(' ', filteredResults_duckie)
            print(' ', nearest_duckie)
            print('----------')
            print('Bots:')
            print(' ', filteredResults_bot)
            print(' ', nearest_bot)
            print('----------')
            print('free Slot:')
            print(' ', filteredResults_freeSlot)
            print(' ', nearest_freeSlot)
            print('----------')
            print('occupied Slot:')
            print(' ', filteredResults_occSlot)
            print(' ', nearest_occSlot)
                
                
    def get_polygon(self):
        return np.array([
            [self.conf['mask']['top_left_x'], self.conf['mask']['top_left_y']],
            [self.conf['mask']['top_right_x'], self.conf['mask']['top_right_y']],
            [self.conf['mask']['bottom_right_x'], self.conf['mask']['bottom_right_y']],
            [self.conf['mask']['bottom_left_x'], self.conf['mask']['bottom_left_y']],
        ], dtype=np.int32)
        
        
    def get_filteredResults_and_nearest(
            self,
            results,
            name,
            mask,
            filtered_results=[],
            y_lowest=0
        ):
        myClass = self.conf['classes'][name]
        
        for result in results:
            filtered = FilteredResults(names=name)

            for box in result.boxes:
                if box.cls[0] == myClass:
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


if __name__ == '__main__':
    node = DetectParkingSlotNode(node_name='detect_duckieBotSlot_node')
    rospy.spin()
