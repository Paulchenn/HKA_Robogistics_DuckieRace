#!/usr/bin/env python3

import cv2
import numpy as np
import rospy
import os
import json
from std_msgs.msg import String, ColorRGBA, Int32
from geometry_msgs.msg import Point, Twist
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
from duckietown_msgs.msg import LEDPattern
import copy
import time
from cv_bridge import CvBridge

class RedLineListener(DTROS):

    def __init__(self, node_name):
        super(RedLineListener, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)

        self.debug = False  # Setze auf True für Debug-Ausgaben

        self._vehicle_name = os.environ['VEHICLE_NAME']
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"

        self.direction_redline_topic = f"/{self._vehicle_name}/red_line_info"
        self.direction_redline = rospy.Subscriber(self.direction_redline_topic, String, self.callback, queue_size=10)
        self.sub_image = rospy.Subscriber(self._camera_topic, CompressedImage, self.process_image, queue_size=1)

        self.cmd_pub = rospy.Publisher(f"/{self._vehicle_name}/zielpunkt", Int32, queue_size=1)
        self.left_point = None
        self.right_point = None

        self.led_pub = rospy.Publisher(f"/{self._vehicle_name}/led_emitter_node/led_pattern", LEDPattern, queue_size=1)

        self.blink_on = True
        self.blink_duration = 10.0
        self.blink_start_time = time.time()
        self.blink_timer = rospy.Timer(rospy.Duration(0.5), self.blink_all_leds)

        if self.debug:
            rospy.loginfo("Red Line Listener Node Initialized")

    def callback(self, msg):
        data = json.loads(msg.data)
        direction = data["richtung"]
        left = data["left"]
        right = data["right"]
        self.left_point = (int(left[0]), int(left[1]))
        self.right_point = (int(right[0]), int(right[1]))
        self.publish_turn_direction(direction)
        if self.debug:
            rospy.loginfo_throttle(1, f"Empfangen: {direction} | links: {self.left_point} | rechts: {self.right_point}")

    def process_image(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if self.left_point and self.right_point:
            self.ellipseForTurning(frame, self.left_point, self.right_point)

        if self.debug:
            cv2.imshow("Abbiegelinien", frame)
            cv2.waitKey(1)

    def ellipseForTurning(self, frame, left, right):
        left_axes = (150, 300)
        right_axes = (350, 80)
        color = (255, 0, 0)
        thickness = 2

        cv2.ellipse(frame, center=left, axes=left_axes, angle=90, startAngle=200, endAngle=270, color=color, thickness=thickness)
        cv2.ellipse(frame, center=right, axes=right_axes, angle=0, startAngle=180, endAngle=250, color=color, thickness=thickness)

        height, width = frame.shape[:2]
        third_height, half_width = height // 3, width // 2
        cv2.line(frame, (half_width, 400), (half_width, third_height), (0, 255, 0), 2)

    def publish_turn_direction(self, direction):
        self.pattern_on = LEDPattern()
        self.pattern_on.frequency = 2.0

        if direction == "links":
            self.pattern_on.color_mask = [True, False, False, True]
            self.pattern_on.frequency_mask = [True, False, False, True]
            self.pattern_on.rgb_vals = [
                ColorRGBA(1.0, 1.0, 0.0, 1.0),
                ColorRGBA(0, 0, 0, 1),
                ColorRGBA(0, 0, 0, 1),
                ColorRGBA(1.0, 1.0, 0.0, 1.0)
            ]
            zielpunkt_x = 42
        elif direction == "rechts":
            self.pattern_on.color_mask = [False, True, True, False]
            self.pattern_on.frequency_mask = [False, True, True, False]
            self.pattern_on.rgb_vals = [
                ColorRGBA(0, 0, 0, 1),
                ColorRGBA(1.0, 1.0, 0.0, 1.0),
                ColorRGBA(1.0, 1.0, 0.0, 1.0),
                ColorRGBA(0, 0, 0, 1)
            ]
            zielpunkt_x = 414
        elif direction == "geradeaus":
            zielpunkt_x = 223

        self.cmd_pub.publish(zielpunkt_x)
        if self.debug:
            rospy.loginfo(f"Zielpunkt publiziert: {zielpunkt_x}")

    def blink_all_leds(self, event):
        elapsed = time.time() - self.blink_start_time
        if elapsed > self.blink_duration:
            if self.debug:
                rospy.loginfo("Blinken beendet nach 6 Sekunden")
            self.blink_timer.shutdown()
            self.turn_off_leds()
            return

        if not hasattr(self, 'pattern_on'):
            return

        pattern_for_publish = LEDPattern()

        if self.blink_on:
            pattern_for_publish = copy.deepcopy(self.pattern_on)
        else:
            pattern_for_publish.color_mask = [False] * 4
            pattern_for_publish.frequency = 0
            pattern_for_publish.frequency_mask = [False] * 4
            pattern_for_publish.rgb_vals = [ColorRGBA(0, 0, 0, 1)] * 4

        if self.debug:
            rospy.loginfo_throttle(3, f"Jetzt wird gepublisht! {self.pattern_on}")

        self.led_pub.publish(pattern_for_publish)
        self.blink_on = not self.blink_on

    def turn_off_leds(self):
        off_pattern = LEDPattern()
        off_pattern.color_mask = [False] * 4
        off_pattern.frequency_mask = [False] * 4
        off_pattern.rgb_vals = [ColorRGBA(0, 0, 0, 1)] * 4
        self.led_pub.publish(off_pattern)

if __name__ == "__main__":
    listener = RedLineListener(node_name='turning_process_node')
    rospy.spin()
