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


class BypassDuckieNode(DTROS):
    def __init__(self, node_name):
        # Initialize the DTROS parent class
        super(BypassDuckieNode, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)
        self._vehicle_name = os.environ['VEHICLE_NAME']

        # Subscriber for nearest duckie coordinates
        # Duckie nearest Bounding Box (BB) coordinates (x1, y1, x2, y2)
        self._duckieNearestBB_topic = f"/{self._vehicle_name}/detect/duckie/nearestBB"
        self.sub_duckieNearestBB = rospy.Subscriber(self._duckieNearestBB_topic, Float64MultiArray, self.bypassDuckie, queue_size=1)
        
        # Publisher for driving commands
        self._cmd_topic = f"/{self._vehicle_name}/detect/duckie/bypass_cmd"
        self.pub_bypassCmd_vel = rospy.Publisher(self._cmd_topic, Twist2DStamped, queue_size=1)

        with open('packages/followlane/config/detect_duckie.yaml', 'r') as f:
            self.conf = yaml.safe_load(f)
        
        self.bypass_mode = 0  # 0 = normal, 1 = going onto left lane, 2 = on left lane, 3 = going back onto right lane
        self.timestamp_lastDuckie = rospy.Time.now()
        
        
    def bypassDuckie(
            self,
            nearestDuckie
        ):
        """ Callback function to process the nearest duckie coordinates.
        :param nearestDuckie: The incoming message containing the nearest duckie coordinates.
        """
        if nearestDuckie.data:
            # Extract the coordinates of the nearest duckie
            x1, y1, x2, y2 = map(int, nearestDuckie.data)
            
            # # Print the coordinates of the nearest duckie
            # print(f"Nearest Duckie coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

            # Stop the duckie if it is too close
            self.stop_duckie(nearestDuckie)
            
                    
    def stop_duckie(
            self,
            nearestDuckie
    ):
        """ Stop the duckie if it is too close.
        :param nearestDuckie: The incoming message containing the nearest duckie coordinates.
        """
        y_min4stop = self.conf['min_distance_nearestDuckie']

        # Extract the coordinates of the nearest duckie
        # x1 ist the left x-coordinate, y1 is the top y-coordinate,
        # x2 is the right x-coordinate, y2 is the bottom y-coordinate
        x1, y1, x2, y2 = map(int, nearestDuckie.data)
        if y2 <= y_min4stop:
            if duckie erkannt und bypass_mode == 0:
                bypass_mode = 1  # Ausweichen starten

            if bypass_mode == 1:
                # Fahre auf Gegenspur, dann
                bypass_mode = 2  # Bleibe auf Gegenspur

            if bypass_mode == 2 und duckie nicht mehr erkannt:
                bypass_mode = 3
                timestamp_lastDuckie = jetzt

            if bypass_mode == 3 und (jetzt - timestamp_lastDuckie) > 5 Sekunden:
                bypass_mode = 0  # Zurück auf normale Spur


if __name__ == '__main__':
    node = BypassDuckieNode(node_name='bypass_duckie_node')
    rospy.spin()
