#!/usr/bin/env python3

import cv2
import rospy
import numpy as np
import os
import yaml
import math

from cv_bridge import CvBridge
from duckietown.dtros import DTROS, NodeType
from std_msgs.msg import Float64, Float64MultiArray, Int32
from sensor_msgs.msg import CompressedImage
from duckietown_msgs.msg import Twist2DStamped


class BypassDuckieNode(DTROS):
    def __init__(self, node_name):
        super(BypassDuckieNode, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)
        self._vehicle_name = os.environ['VEHICLE_NAME']

        self.debug = True
        with open('packages/followlane/config/detect_duckie.yaml', 'r') as f:
            self.conf = yaml.safe_load(f)

        self._window_name = "Bypass Target View"
        self._bridge = CvBridge()
        self.image = None

        self.right_x = None
        self.left_x = None

        # Duckie Infos
        self.duckie_seen = False
        self.y2 = None
        self.last_y2 = None

        # Ausweich-Modus
        self.mode2_start_time = None
        self.bypass_mode = 0
        self.mode1_start_time = None
        self.right_duckie_last_seen = rospy.Time.now()
        self.current_offset = 0
        self.max_offset = 120

        # Subscriber
        rospy.Subscriber(f"/{self._vehicle_name}/detect/object/duckieNearestBB", Float64MultiArray, self.cb_duckieNearestBB, queue_size=1)
        rospy.Subscriber(f"/{self._vehicle_name}/detect/object/duckieNearestRightBB", Float64MultiArray, self.cb_nearestBBright, queue_size=1)
        rospy.Subscriber(f"/{self._vehicle_name}/detect/lane/right_x", Float64, self.cb_right_x, queue_size=1)
        rospy.Subscriber(f"/{self._vehicle_name}/detect/lane/left_x", Float64, self.cb_left_x, queue_size=1)
        rospy.Subscriber(f"/{self._vehicle_name}/camera_node/image/compressed", CompressedImage, self.cb_image, queue_size=1)

        # Publisher
        self.pub_target_override = rospy.Publisher(f"/{self._vehicle_name}/detect/duckie/bypass_target", Float64, queue_size=1)
        self.pub_duckie_info = rospy.Publisher(f"/{self._vehicle_name}/detect/duckie/info", Int32, queue_size=1)
        self.pub_bypassCmd_vel = rospy.Publisher(f"/{self._vehicle_name}/detect/duckie/bypass_cmd", Twist2DStamped, queue_size=1)

    def cb_duckieNearestBB(self, msg):
        if msg.data:
            x1, y1, x2, y2 = map(int, msg.data)
            self.y2 = y2
            self.last_y2 = y2  # nur aktualisieren, nicht löschen
            self.duckie_seen = True
            if self.debug:
                rospy.loginfo(f"[DUCKIE] BB: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
        else:
            self.duckie_seen = False
            self.y2 = None
            # self.last_y2 NICHT löschen
            if self.debug:
                rospy.loginfo("[DUCKIE] Keine Ente erkannt")

    def cb_nearestBBright(self, msg):
        if msg.data:
            self.right_duckie_last_seen = rospy.Time.now()
            if self.debug:
                rospy.loginfo("[BYPASS] Ente im rechten Sichtfeld erkannt")

    def cb_right_x(self, msg):
        self.right_x = msg.data

    def cb_left_x(self, msg):
        self.left_x = msg.data

    def cb_image(self, msg):
        self.image = self._bridge.compressed_imgmsg_to_cv2(msg)

    def update_gui(self, target_x=None):
        if self.image is None:
            return

        img = self.image.copy()
        y = img.shape[0] - 50

        if target_x is not None:
            cv2.circle(img, (int(target_x), y), 6, (255, 0, 255), -1)
            cv2.putText(img, f"Bypass Target X = {int(target_x)}", (int(target_x) - 40, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        if self.debug:
            cv2.imshow(self._window_name, img)
            cv2.waitKey(1)

    def run(self):
        rate = rospy.Rate(10)
        y_min4stop = self.conf['min_distance_nearestDuckie']

        while not rospy.is_shutdown():
            target_x = None

            # --- MODUS 0 ---
            if self.bypass_mode == 0:
                if self.duckie_seen and self.y2 is not None:
                    if self.y2 < y_min4stop:
                        self.pub_duckie_info.publish(Int32(1))
                        if self.debug:
                            rospy.loginfo("[BYPASS] Duckie erkannt (fern) – verlangsamen")
                    elif self.y2 >= y_min4stop:
                        self.pub_duckie_info.publish(Int32(2))
                        self.bypass_mode = 1
                        self.mode1_start_time = rospy.Time.now()
                        if self.debug:
                            rospy.loginfo("[BYPASS] Duckie nah – starte Ausweichmanöver (Modus 1)")
                else:
                    self.pub_duckie_info.publish(Int32(0))
                    if self.debug:
                        rospy.loginfo("[BYPASS] Duckie nicht sichtbar in Modus 0")

            # --- MODUS 1 ---
            if self.bypass_mode == 1:
                self.pub_duckie_info.publish(Int32(2))

                if self.right_x is not None:
                    time_in_mode1 = (rospy.Time.now() - self.mode1_start_time).to_sec()
                    duration = 1.5

                    # Zeit normieren
                    t_norm = min(max(time_in_mode1 / duration, 0.0), 1.0)

                    # Cos-Funktion: Offset von +150 → -150
                    offset_factor = math.cos(t_norm * math.pi)  # 1 → -1
                    offset = offset_factor * 150
                    target_x = self.right_x + offset
                    self.pub_target_override.publish(Float64(target_x))

                    if self.debug:
                        rospy.loginfo(f"[BYPASS DEBUG] Modus 1 – t={round(t_norm,2)}, offset={int(offset)}, target_x = {int(target_x)}")
                else:
                    if self.debug:
                        rospy.logwarn("[BYPASS DEBUG] Modus 1 – keine Linien erkannt")

                # Bedingung zum Wechsel in Modus 2
                time_in_mode1 = (rospy.Time.now() - self.mode1_start_time).to_sec()
                time_since_right = (rospy.Time.now() - self.right_duckie_last_seen).to_sec()
                if time_in_mode1 >= 4 and time_since_right >= 1.0:
                    self.bypass_mode = 2
                    self.mode2_start_time = rospy.Time.now()
                    if self.debug:
                        rospy.loginfo("[BYPASS] Starte Rückkehrbewegung (Modus 2)")

            # --- MODUS 2 ---
            if self.bypass_mode == 2:
                self.pub_duckie_info.publish(Int32(2))

                if self.right_x is not None:
                    time_in_mode2 = (rospy.Time.now() - self.mode2_start_time).to_sec()
                    duration = 1.5

                    t_norm = min(max(time_in_mode2 / duration, 0.0), 1.0)

                    # Rückfahrt: Offset von -150 → +150
                    offset_factor = -math.cos(t_norm * math.pi)  # -1 → +1
                    offset = offset_factor * 150
                    target_x = self.right_x + offset
                    self.pub_target_override.publish(Float64(target_x))

                    if self.debug:
                        rospy.loginfo(f"[BYPASS DEBUG] Modus 2 – t={round(t_norm,2)}, offset={int(offset)}, target_x = {int(target_x)}")
                else:
                    if self.debug:
                        rospy.logwarn("[BYPASS DEBUG] Modus 2 – keine Linien erkannt")

                # Rückkehr abgeschlossen → Modus 0
                if time_in_mode2 >= duration + 0.2:
                    self.bypass_mode = 0
                    self.mode1_start_time = None
                    self.mode2_start_time = None
                    self.current_offset = 0
                    self.last_y2 = None
                    self.pub_duckie_info.publish(Int32(0))
                    if self.debug:
                        rospy.loginfo("[BYPASS] Rückkehr abgeschlossen – zurück in Modus 0")

            self.update_gui(target_x)
            rate.sleep()



if __name__ == '__main__':
    node = BypassDuckieNode(node_name='bypass_duckie_node')
    node.run()
