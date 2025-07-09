#!/usr/bin/env python3

import cv2
import rospy
import numpy as np
import os
import yaml

from cv_bridge import CvBridge
from duckietown.dtros import DTROS, NodeType
from std_msgs.msg import Float64, Float64MultiArray, Int32
from sensor_msgs.msg import CompressedImage


class BypassDuckieNode(DTROS):
    def __init__(self, node_name):
        super(BypassDuckieNode, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)
        self._vehicle_name = os.environ['VEHICLE_NAME']
        self.debug = False

        # Konfiguration
        with open('packages/followlane/config/detect_duckie.yaml', 'r') as f:
            self.conf = yaml.safe_load(f)

        self._window_name = "Bypass Target View"
        self._bridge = CvBridge()
        self.image = None

        # Fahrbahnrand-Daten
        self.right_x = None  # gelbe Linie
        self.left_x = None   # weiße Linie

        # Zustand
        self.bypass_mode = 0
        self.mode1_start_time = None
        self.right_duckie_last_seen = rospy.Time.now()

        # Subscriber
        rospy.Subscriber(f"/{self._vehicle_name}/detect/duckie/nearestBB", Float64MultiArray, self.bypassDuckie, queue_size=1)
        rospy.Subscriber(f"/{self._vehicle_name}/detect/duckie/nearestBBright", Float64MultiArray, self.cb_nearestBBright, queue_size=1)
        rospy.Subscriber(f"/{self._vehicle_name}/detect/lane/right_x", Float64, self.cb_right_x, queue_size=1)
        rospy.Subscriber(f"/{self._vehicle_name}/detect/lane/left_x", Float64, self.cb_left_x, queue_size=1)
        rospy.Subscriber(f"/{self._vehicle_name}/camera_node/image/compressed", CompressedImage, self.cb_image, queue_size=1)

        # Publisher
        self.pub_target_override = rospy.Publisher(f"/{self._vehicle_name}/detect/duckie/bypass_target", Float64, queue_size=1)
        self.pub_duckie_info = rospy.Publisher(f"/{self._vehicle_name}/detect/duckie/info", Int32, queue_size=1)

        # Timer für periodisches Prüfen auf Rückkehr in Modus 0
        rospy.Timer(rospy.Duration(0.1), self.check_mode_reset)

    def cb_right_x(self, msg):
        self.right_x = msg.data

    def cb_left_x(self, msg):
        self.left_x = msg.data

    def cb_image(self, msg):
        self.image = self._bridge.compressed_imgmsg_to_cv2(msg)

    def cb_nearestBBright(self, msg):
        if msg.data:
            self.right_duckie_last_seen = rospy.Time.now()
            if self.debug:
                rospy.loginfo("[BYPASS] Ente im rechten Sichtfeld erkannt")

    def bypassDuckie(self, nearestDuckie):
        y_min4stop = self.conf['min_distance_nearestDuckie']
        duckie_seen = False
        y2 = None
        target_x = None

        if nearestDuckie.data:
            x1, y1, x2, y2 = map(int, nearestDuckie.data)
            duckie_seen = True
            if self.debug:
                rospy.loginfo(f"[DUCKIE] BB: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
        else:
            if self.debug:
                rospy.loginfo("[DUCKIE] Keine Ente erkannt")

        if self.debug:
            rospy.loginfo(f"[BYPASS] Aktueller Modus: {self.bypass_mode}")

        # --- MODUS 0: Ente erkennen und auf Modus 1 umschalten ---
        if self.bypass_mode == 0:
            if duckie_seen and y2 < y_min4stop:
                self.pub_duckie_info.publish(Int32(1))
                if self.debug:
                    rospy.loginfo("[BYPASS] Duckie erkannt (fern) – verlangsamen")
            elif duckie_seen and y2 >= y_min4stop:
                self.pub_duckie_info.publish(Int32(2))
                self.bypass_mode = 1
                self.mode1_start_time = rospy.Time.now()
                if self.debug:
                    rospy.loginfo("[BYPASS] Duckie nah – starte Ausweichmanöver (Modus 1)")

        # --- MODUS 1: Mindestens 2 Sekunden aktiv bleiben und Linien folgen ---
        if self.bypass_mode == 1:
            self.pub_duckie_info.publish(Int32(2))
            if self.right_x is not None:
                target_x = self.right_x - 100
                self.pub_target_override.publish(Float64(target_x))
                if self.debug:
                    rospy.loginfo(f"[BYPASS DEBUG] Modus 1 – nur gelb: target_x = {target_x}")
            else:
                if self.debug:
                    rospy.logwarn("[BYPASS DEBUG] Modus 1 – keine Linien erkannt")

        # Duckie nicht mehr sichtbar in Modus 0
        if not duckie_seen and self.bypass_mode == 0:
            self.pub_duckie_info.publish(Int32(0))
            if self.debug:
                rospy.loginfo("[BYPASS] Duckie nicht mehr sichtbar in Modus 0")

        # GUI immer aktualisieren
        self.update_gui(target_x)

    def check_mode_reset(self, event):
        """Timer-callback: Prüft, ob wir aus Modus 1 zurück auf Modus 0 wechseln dürfen."""
        if self.bypass_mode == 1:
            time_in_mode1 = (rospy.Time.now() - self.mode1_start_time).to_sec()
            time_since_right_duckie = (rospy.Time.now() - self.right_duckie_last_seen).to_sec()

            if time_in_mode1 >= 5.0 and time_since_right_duckie >= 1.0:
                self.bypass_mode = 0
                self.mode1_start_time = None
                if self.debug:
                    rospy.loginfo("[BYPASS] Wechsel zurück zu Modus 0 nach Umfahrung")

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


if __name__ == '__main__':
    node = BypassDuckieNode(node_name='bypass_duckie_node')
    rospy.spin()
