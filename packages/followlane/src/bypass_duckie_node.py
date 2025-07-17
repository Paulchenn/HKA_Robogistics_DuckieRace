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
        self._duckieNearestBB_topic = f"/{self._vehicle_name}/detect/object/duckieNearestBB"
        self.sub_duckieNearestBB = rospy.Subscriber(self._duckieNearestBB_topic, Float64MultiArray, self.bypassDuckie, queue_size=1)
        
        # Publisher for driving commands
        self._cmd_topic = f"/{self._vehicle_name}/detect/duckie/bypass_cmd"
        self.pub_bypassCmd_vel = rospy.Publisher(self._cmd_topic, Twist2DStamped, queue_size=1)

        with open('packages/followlane/config/detect_duckie.yaml', 'r') as f:
            self.conf = yaml.safe_load(f)
        
        self.bypass_mode = 0  # 0 = normal, 1 = going onto left lane, 2 = on left lane, 3 = going back onto right lane
        self.timestamp_lastDuckie = rospy.Time.now()
        
        # Flags to indicate whether we are driving to the left lane or already on the left lane
        # and whether we are driving on the left lane
        # These flags are used to control the bypass behavior
        # and to ensure that we only switch lanes when necessary
        self.drive_to_left_lane     = False
        self.drive_on_left_lane     = False 
        self.drive_to_right_lane    = False
        self.drive_on_right_lane    = True  # Initially, we are on the right lane
        
        
    def bypassDuckie(
            self,
            nearestDuckie
        ):
        """ Callback function to process the nearest duckie coordinates.
        :param nearestDuckie: The incoming message containing the nearest duckie coordinates.
        """
        y_min4stop = self.conf['min_distance_nearestDuckie']
        
        if nearestDuckie.data:
            # Extract the coordinates of the nearest duckie
            x1, y1, x2, y2 = map(int, nearestDuckie.data)
        else:
            # If no duckie is detected, set coordinates to None
            x1, y1, x2, y2 = None, None, None, None
        
        rospy.loginfo(f"bypass_mode: {self.bypass_mode}")
        rospy.loginfo(f"y_min4stop: {y_min4stop}")
        rospy.loginfo(f"Nearest Duckie BB: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
        if y2 is not None and y2 <= y_min4stop and self.bypass_mode == 0:
            # If the duckie is too close, switch to bypass mode 1 to go onto the left lane
            self.bypass_mode = 1
            self.drive_to_left_lane = True
            self.driving_to_left_lane()
            
            while self.bypass_mode != 0:
                rospy.loginfo(f"bypass_mode: {self.bypass_mode}")
                rospy.loginfo(f"y_min4stop: {y_min4stop}")
                rospy.loginfo(f"Nearest Duckie BB: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
                
                if self.bypass_mode == 1 and self.drive_to_left_lane == False and self.drive_on_left_lane == True:
                    # If we are in bypass mode 1 and have already switched to the left lane,
                    # switch to bypass mode 2 to indicate we are on the left lane
                    self.bypass_mode = 2
                    self.drive_on_left_lane = True
                    self.driving_on_left_lane()
                    
                elif self.bypass_mode == 2 and self.drive_to_left_lane == False and self.drive_on_left_lane == True:
                    # If we are in bypass mode 2 and have already switched to the left lane and
                    # there is no duckie in sight,
                    # switch to bypass mode 3 to go back onto the right lane
                    self.bypass_mode = 3
                    self.drive_to_right_lane = True
                    self.driving_to_right_lane()
                    self.stop_duckie()
                    
                else:
                    break  # Exit the loop if no further action is needed
        else:
            self.bypass_mode = 0
            self.stop_duckie()
            

        
    def driving_to_left_lane(self):
        """ Function to drive the vehicle to the left lane.
        This function is called when the vehicle needs to switch to the left lane
        to bypass a duckie that is too close.
        
        v (Speed): e.g. 0.2 (drive slowly)
        omega (Steering): e.g. +3.0 (steer strongly to the left)
        
        after 1 second,
        
        v (Speed): e.g. 0.2 (drive slowly)
        omega (Steering): e.g. -3.0 (steer strongly to the right)
        
        This function will set the flag `drive_to_left_lane` to False when the vehicle
        has finished driving to the left lane.
        """
        msg = Twist2DStamped()
        msg.header.stamp = rospy.Time.now()
        
        # Set the speed and steering angle to drive to the left lane
        msg.v = 0.2  # Speed (m/s)
        msg.omega = 3.0  # Steering angle (rad/s)
        
        # Publish the driving command
        self.pub_bypassCmd_vel.publish(msg)
        
        rospy.loginfo("Driving to left lane...")
        
        # Wait for a short duration to simulate driving time
        rospy.sleep(1.0)  # Adjust the sleep time as needed
        
        msg.v = 0.2  # Continue driving slowly
        msg.omega = -3.0  # Steer strongly to the right to align with the left lane
        
        # Publish the updated driving command
        self.pub_bypassCmd_vel.publish(msg)
        
        rospy.loginfo("Driving on left lane...")
        
        # Wait for a short duration to simulate driving time
        rospy.sleep(1.0)  # Adjust the sleep time as needed
        
        # After driving to the left lane, set the flags
        self.drive_to_left_lane = False
        self.drive_on_left_lane = True
        
    
    def driving_on_left_lane(self):
        """ Function to drive the vehicle on the left lane.
        This function is called when the vehicle is already on the left lane
        and needs to continue driving until it can switch back to the right lane.
        
        v (Speed): e.g. 0.2 (drive slowly)
        omega (Steering): e.g. 0.0 (drive straight)
        
        This function will set the flag `driving_on_left_lane` to False when the vehicle
        has finished driving on the left lane.
        """
        msg = Twist2DStamped()
        msg.header.stamp = rospy.Time.now()
        
        # Set the speed and steering angle to drive on the left lane
        msg.v = 0.2  # Speed (m/s)
        msg.omega = 0.0  # Steering angle (rad/s) - drive straight
        
        # Publish the driving command
        self.pub_bypassCmd_vel.publish(msg)
        
        rospy.loginfo("Driving on left lane...")
        
        # Wait for a short duration to simulate driving time
        rospy.sleep(0.1)  # Adjust the sleep time as needed
    
    
    def driving_to_right_lane(self):
        """ Function to drive the vehicle back to the right lane.
        This function is called when the vehicle has finished driving on the left lane
        and needs to switch back to the right lane.
        
        v (Speed): e.g. 0.2 (drive slowly)
        omega (Steering): e.g. -3.0 (steer strongly to the right)
        
        after 1 second,
        
        v (Speed): e.g. 0.2 (drive slowly)
        omega (Steering): e.g. 3.0 (steer strongly to the left)
        
        This function will set the flags `drive_to_left_lane` and `drive_on_left_lane` to False
        when the vehicle has finished driving back to the right lane.
        """
        msg = Twist2DStamped()
        msg.header.stamp = rospy.Time.now()
        
        # Set the speed and steering angle to drive back to the right lane
        msg.v = 0.2  # Speed (m/s)
        msg.omega = -3.0  # Steering angle (rad/s) - steer strongly to the right
        
        # Publish the driving command
        self.pub_bypassCmd_vel.publish(msg)
        
        rospy.loginfo("Driving to right lane...")
        
        # Wait for a short duration to simulate driving time
        rospy.sleep(1.0)  # Adjust the sleep time as needed
        
        msg.v = 0.2  # Continue driving slowly
        msg.omega = 3.0  # Steer strongly to the left to align with the right lane
        
        # Publish the updated driving command
        self.pub_bypassCmd_vel.publish(msg)
        
        rospy.loginfo("Driving back on right lane...")
        
        # Wait for a short duration to simulate driving time
        rospy.sleep(1.0)  # Adjust the sleep time as needed
        
        # After driving back to the right lane, reset the flags
        self.drive_to_right_lane = False
        
        # Set the flags to indicate that we are no longer driving on the left lane
        # and that we have finished the bypass maneuver
        self.drive_to_left_lane = False
        self.drive_on_left_lane = False
        
        # Log the completion of the bypass maneuver
        rospy.loginfo("Bypass maneuver completed. Back on right lane.")
        
        
    def stop_duckie(self):
        """ Function to stop the duckie detection and bypass commands.
        This function is called to stop the duckie detection and bypass commands
        when the vehicle is no longer in bypass mode.
        """
        msg = Twist2DStamped()
        msg.header.stamp = rospy.Time.now()
        
        # Set speed and steering angle to zero to stop the vehicle
        msg.v = 0.0  # Speed (m/s)
        msg.omega = 0.0  # Steering angle (rad/s)
        
        # Publish the stop command
        self.pub_bypassCmd_vel.publish(msg)
        
        rospy.loginfo("Stopping duckie detection and bypass commands.")
        


if __name__ == '__main__':
    node = BypassDuckieNode(node_name='bypass_duckie_node')
    rospy.spin()
