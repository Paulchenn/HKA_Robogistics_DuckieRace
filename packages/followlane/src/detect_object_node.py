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

        with open('packages/followlane/config/detect_duckie.yaml', 'r') as f:
            self.conf = yaml.safe_load(f)

        self._bridge = CvBridge()
        self.frame_count = 0

        self._window1 = "Camera Feed"
        self._window2 = "YOLOv8 Detection"

    def cbDetectObjects(self, image_msg):
        self.frame_count += 1
        if self.frame_count % 5 != 0:  # Nur jedes 5. Bild verarbeiten
            return

        # 1. Convert CompressedImage message to OpenCV image
        cv_image = self._bridge.compressed_imgmsg_to_cv2(image_msg)

        # 2. Run YOLO model on the image
        results = self._model(cv_image, conf=0.5, iou=0.5, agnostic_nms=True, verbose=False)

        # 2. Plot results on the image
        annotated_frame = results[0].plot()

        # 4. Publish image mit Bounding Boxes
        if annotated_frame is not None:
            img_msg = self._bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')
            self.pup_image.publish(img_msg)

        # Optional: nearest Duckie weiterverarbeiten, falls benötigt
        # ...


    def draw_bounding_boxes_all(self, results, img):
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0]) if hasattr(box, 'cls') else 0
                label = result.names.get(cls_id, "object")

                # Bounding Box zeichnen (blau)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                # Label über der Box zeichnen
                cv2.putText(img, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        return img

    def display_loop(self):
        while True:
            if self.latest_img is not None:
                cv2.imshow(self._window, self.latest_img)
                cv2.waitKey(1)


if __name__ == '__main__':
    node = DetectParkingSlotNode(node_name='detect_duckieBotSlot_node')
    rospy.spin()
