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
        self.debug = True

        # Konfiguration
        with open('packages/followlane/config/detect_duckie.yaml', 'r') as f:
            self.conf = yaml.safe_load(f)

        self._window_name = "Bypass Target View"
        self._bridge = CvBridge()
        self.image = None
        self.bypass_start_time = None


        # Fahrbahnrand-Datens
        self.right_x = None  # gelbe Linie
        self.left_x = None   # weiße Linie

        # Zustand
        self.bypass_mode = 0

        # Subscriber
        rospy.Subscriber(f"/{self._vehicle_name}/detect/duckie/nearestBB", Float64MultiArray, self.bypassDuckie, queue_size=1)
        rospy.Subscriber(f"/{self._vehicle_name}/detect/lane/right_x", Float64, self.cb_right_x, queue_size=1)
        rospy.Subscriber(f"/{self._vehicle_name}/detect/lane/left_x", Float64, self.cb_left_x, queue_size=1)
        rospy.Subscriber(f"/{self._vehicle_name}/camera_node/image/compressed", CompressedImage, self.cb_image, queue_size=1)

        # Publisher
        self.pub_target_override = rospy.Publisher(f"/{self._vehicle_name}/detect/duckie/bypass_target", Float64, queue_size=1)
        self.pub_duckie_info = rospy.Publisher(f"/{self._vehicle_name}/detect/duckie/info", Int32, queue_size=1)

    def cb_right_x(self, msg):
        self.right_x = msg.data

    def cb_left_x(self, msg):
        self.left_x = msg.data

    def cb_image(self, msg):
        self.image = self._bridge.compressed_imgmsg_to_cv2(msg)

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

        # --- MODUS 0: Duckie erkennen und auf Modus 1 umschalten ---
        if self.bypass_mode == 0:
            if duckie_seen and y2 < y_min4stop:
                self.pub_duckie_info.publish(Int32(1))
                if self.debug:
                    rospy.loginfo("[BYPASS] Duckie erkannt (fern) – verlangsamen")
            elif duckie_seen and y2 >= y_min4stop:
                self.pub_duckie_info.publish(Int32(2))
                self.bypass_mode = 1
                if self.debug:
                    rospy.loginfo("[BYPASS] Duckie nah – starte Ausweichmanöver (Modus 1)")

        # --- MODUS 1: Nach gelber Linie (right_x) steuern ---
        if self.bypass_mode == 1:
            if self.left_x is not None and self.left_x < 400:
                self.bypass_mode = 2
                rospy.loginfo("[BYPASS DEBUG] Zielpunkt > 450 → Übergang in Modus 2")
            elif self.right_x is not None:
                target_x = self.right_x - 200
                self.pub_target_override.publish(Float64(target_x))
                if self.debug:
                    rospy.loginfo(f"[BYPASS DEBUG] Modus 1 – Zielpunkt (gelbe Linie = right_x): {target_x}")
            else:
                if self.debug:
                    rospy.logwarn("[BYPASS DEBUG] Keine gelbe Linie (right_x) erkannt – kein Zielpunkt.")

        # --- MODUS 2: Platzhalter für Weiterfahrt ---
        elif self.bypass_mode == 2:
            if self.bypass_start_time is None:
                self.bypass_start_time = rospy.Time.now()
                rospy.loginfo("[BYPASS DEBUG] Start Modus 2 – 1 Sekunde Geradeausregelung")

            # Prüfe ob 1 Sekunde schon vergangen ist
            elapsed = (rospy.Time.now() - self.bypass_start_time).to_sec()
            if elapsed < 1.0:
                # Bestimme Zielpunkt wie in CameraReaderNode
                if self.left_x is not None and self.right_x is not None:
                    target_x = (self.left_x + self.right_x) / 2
                    self.pub_target_override.publish(Float64(target_x))
                    if self.debug:
                        rospy.loginfo(f"[BYPASS DEBUG] Modus 2 – beide Linien: target_x = {target_x}")
                elif self.right_x is not None:
                    target_x = self.right_x - 200
                    self.pub_target_override.publish(Float64(target_x))
                    if self.debug:
                        rospy.loginfo(f"[BYPASS DEBUG] Modus 2 – nur gelb: target_x = {target_x}")
                elif self.left_x is not None:
                    target_x = self.left_x + 200
                    self.pub_target_override.publish(Float64(target_x))
                    if self.debug:
                        rospy.loginfo(f"[BYPASS DEBUG] Modus 2 – nur weiß: target_x = {target_x}")
                else:
                    if self.debug:
                        rospy.logwarn("[BYPASS DEBUG] Modus 2 – keine Linien erkannt")
            else:
                rospy.loginfo("[BYPASS DEBUG] Modus 2 abgeschlossen – ggf. nächsten Modus starten")
                # Optional: Übergang zu Modus 3 o.ä.


        # Duckie verschwindet – kein Einfluss mehr auf Manöver, außer in Modus 0
        if not duckie_seen and self.bypass_mode == 0:
            self.pub_duckie_info.publish(Int32(0))
            if self.debug:
                rospy.loginfo("[BYPASS] Duckie nicht mehr sichtbar in Modus 0")

        # GUI immer aktualisieren
        self.update_gui(target_x)

    def update_gui(self, target_x=None):
        if self.image is None:
            return

        img = self.image.copy()
        y = img.shape[0] - 50

        if target_x is not None:
            cv2.circle(img, (int(target_x), y), 6, (255, 0, 255), -1)
            cv2.putText(img, f"Bypass Target X = {int(target_x)}", (int(target_x) - 40, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

        cv2.imshow(self._window_name, img)
        cv2.waitKey(1)


if __name__ == '__main__':
    node = BypassDuckieNode(node_name='bypass_duckie_node')
    rospy.spin()
