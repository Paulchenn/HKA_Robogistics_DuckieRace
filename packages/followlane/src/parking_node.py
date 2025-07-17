#!/usr/bin/env python3

import rospy
import numpy as np
import os
import yaml
import time
import itertools

from duckietown.dtros import DTROS, NodeType
from std_msgs.msg import Float64MultiArray
from duckietown_msgs.msg import Twist2DStamped
from sensor_msgs.msg import Image


class ParkingNode(DTROS):
    def __init__(self, node_name):
        # Initialize the DTROS parent class
        super(ParkingNode, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)
        self._vehicle_name = os.environ['VEHICLE_NAME']

        # Subscriber for image data with bounding boxes
        self._yolo_topic = f"/{self._vehicle_name}/detect/object/image"
        self.sub_image = rospy.Subscriber(self._yolo_topic, Image, self.cbPark, queue_size=1)

        # Subscriber for the nearest parking coordinates (Float64MultiArray)
        self._parkingBB_topic = f"/{self._vehicle_name}/detect/object/parkingBB"
        self.sub_parkingBB = rospy.Subscriber(self._parkingBB_topic, Float64MultiArray, self.cbPark, queue_size=1)

        # Publisher for the SwitchControlNode
        lane_cmd_topic = f"/{self._vehicle_name}/car_cmd_switch_node/cmd"
        self.pub_lane_twist = rospy.Publisher(lane_cmd_topic, Twist2DStamped, queue_size=1)

        with open('packages/followlane/config/detect_duckieBotSlot.yaml', 'r') as f:
            self.conf = yaml.safe_load(f)

        twist = Twist2DStamped()
        twist.header.stamp = rospy.Time.now()
        twist.v = 0
        twist.omega = 0
        self.pub_lane_twist.publish(twist)
        rospy.on_shutdown(self.fnShutDown)

        self.mode = 0
        self.ctr = 0
        self.time_lastBB = time.time()
        self.xyxy = [0, 0, 0, 0]


    def cbPark(self, msg):
        twist = Twist2DStamped()
        twist.header.stamp = rospy.Time.now()

        if isinstance(msg, Float64MultiArray):
            # Process the Float64MultiArray for parking coordinates
            self.time_lastBB = time.time()
            self.xyxy = msg.data

        if time.time()-self.time_lastBB > 1.5:
            self.xyxy = [0, 0, 0, 0]

        rospy.loginfo("==========")

        if self.xyxy[3] > self.conf["nearestSlot_y2forStartParking"]:
            rospy.loginfo(f"Spot free and near")
            self.mode = 1
            # print(msg)
        else:
            rospy.loginfo(f"No near Spot.")
            #self.mode = 0
            # print(msg)

        if self.mode == 1 and self.xyxy[1] < self.conf["nearestSlot_y1tillDriveSlow"]:
            self.mode = 2
            rospy.loginfo(f"drive slowly")
            twist.v = 0.05
            twist.omega = 0
        else:
            twist.v = 0
            twist.omega = 0

        if self.mode == 2 and sum(self.xyxy)==0:
            if self.ctr == 0:
                self.time_startPark = time.time()
                self.ctr = 1
            for i in itertools.count():
                delta = time.time()-self.time_startPark
                print(delta)
                if delta<3:
                    print("In Parking")
                    twist.v = -0.08
                    twist.omega = 1.5
                    self.pub_lane_twist.publish(twist)
                else:
                    twist.v = 0
                    twist.omega = 0
                    self.pub_lane_twist.publish(twist)
                    self.ctr = 1
                    #break
            self.mode = 0
            for _ in range(3):  # Repeat Stop message 3 times
                twist.v = 0
                twist.omega = 0
                self.pub_lane_twist.publish(twist)
                rospy.sleep(0.1)  # Short delay between stop messages

        self.pub_lane_twist.publish(twist)

        rospy.loginfo(self.mode)
        rospy.loginfo("==========")


    def fnShutDown(self):
        stop_msg = Twist2DStamped(v=0.0, omega=0.0)
        for _ in range(5):
            self.pub_lane_twist.publish(stop_msg)
            rospy.sleep(0.1)


if __name__ == '__main__':
    node = ParkingNode(node_name='parking_node')
    rospy.spin()
