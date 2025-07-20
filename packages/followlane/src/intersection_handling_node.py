#!/usr/bin/env python3

import cv2
import yaml
import numpy as np
import os
import json
import rospy
import random
from duckietown.dtros import DTROS, NodeType
from std_msgs.msg import String, Bool, Int32, Float64
from sensor_msgs.msg import CompressedImage

class RedLineDetector(DTROS):
    def __init__(self, node_name):
        super(RedLineDetector, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)

        self._vehicle_name = os.environ['VEHICLE_NAME']
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        self._redLine_topic = f"/{self._vehicle_name}/stop_line_detected"
        self._config_path = 'packages/followlane/config/detect_lane.yaml'
        with open(self._config_path, 'r') as f:
            self.conf = yaml.safe_load(f)

        self.pub_red_line_info = rospy.Publisher(f"/{self._vehicle_name}/red_line_info", String, queue_size=1)
        self.pub_info = rospy.Publisher(f"/{self._vehicle_name}/abfrage_info", Int32, queue_size=1)
        self.pub_target_x = rospy.Publisher(f"/{self._vehicle_name}/target_x", Int32, queue_size=1)

        self.sub_image = rospy.Subscriber(self._camera_topic, CompressedImage, self.cb_image, queue_size=1)
        self.sub_redline = rospy.Subscriber(self._redLine_topic, Bool, self.process_stop_line, queue_size=1)
        self.sub_left_x = rospy.Subscriber(f"/{self._vehicle_name}/detect/lane/left_x", Float64, self.cb_left_x)

        self.current_image = None

        self.direction_already_published = False
        self.stop_line_detected = False
        self.waiting_at_line = False
        self.wait_start_time = None
        self.abbiegephase_gestartet = False
        self.abgeschlossen = False
        self.chosen_direction = None
        self.left_x_value = None
        self.abbiege_start_time = None
        self.last_turn_completed_time = None

        self.debug = False
        if self.debug:
            rospy.loginfo("[RedLineDetector] Debug-Modus aktiviert")

    def cb_image(self, msg):
        # Bild puffern
        np_arr = np.frombuffer(msg.data, np.uint8)
        self.current_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    def cb_left_x(self, msg):
        self.left_x_value = msg.data
        # if self.debug:
        #     rospy.loginfo(f"[Abbiegen] Empfange weißen Left-X: {self.left_x_value}")

    def process_stop_line(self, msg):
        current_time = rospy.get_time()
        in_cooldown = (
            self.last_turn_completed_time is not None and
            (current_time - self.last_turn_completed_time) < 4.0
        )

        if msg.data and not in_cooldown:
            self.abgeschlossen = False
            if self.debug:
                rospy.loginfo_throttle(5, "[STOP] Stoplinie erkannt")
            if not self.abbiegephase_gestartet and not self.waiting_at_line:
                self.stop_line_detected = True
                self.waiting_at_line = True
                self.wait_start_time = current_time
        elif msg.data and in_cooldown:
            if self.debug:
                rospy.loginfo_throttle(5, "[STOP] Stoplinie ignoriert – Cooldown aktiv")

    def run(self):
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            if self.current_image is None:
                rate.sleep()
                continue

            frame = self.current_image.copy()
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            rd = self.conf['red']
            lower_red1 = np.array([rd['hl'], rd['sl'], rd['vl']])
            upper_red1 = np.array([rd['hh'], rd['sh'], rd['vh']])
            lower_red2 = np.array([170, rd['sl'], rd['vl']])
            upper_red2 = np.array([180, rd['sh'], rd['vh']])

            mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask_red = cv2.bitwise_or(mask_red1, mask_red2)

            contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            min_area = 150
            filtered_contours_red = [cnt for cnt in contours_red if cv2.contourArea(cnt) > min_area]

            height, width = frame.shape[:2]
            half_width = width // 2
            fifth_width = width // 5
            quarter_height = height // 4
            y_horizontal = 3 * quarter_height

            options = []
            for contour in filtered_contours_red:
                x, y, w, h = cv2.boundingRect(contour)
                center_x = x + w // 2
                center_y = y + h // 2
                if center_y < y_horizontal:
                    if center_x <= fifth_width:
                        options.append("links")
                    elif center_x <= half_width:
                        options.append("geradeaus")
                    else:
                        options.append("rechts")

            if options and not self.direction_already_published:
                rospy.loginfo(f"[Abbiegen] Mögliche Richtungen erkannt: {options}")
                self.chosen_direction = random.choice(options)
                msg_out = String()
                msg_out.data = json.dumps({"richtung": self.chosen_direction})
                self.pub_red_line_info.publish(msg_out)
                self.direction_already_published = True
                rospy.loginfo(f"[Abbiegen] Richtung gewählt: {self.chosen_direction}")

            current_time = rospy.get_time()
            if self.waiting_at_line:
                if current_time - self.wait_start_time < 2.0:
                    self.pub_info.publish(Int32(3))
                    #rate.sleep()
                    continue
                else:
                    self.waiting_at_line = False
                    self.abbiegephase_gestartet = True
                    self.abbiege_start_time = current_time
                    rospy.loginfo("[Abbiegen] Abbiegevorgang gestartet")

            if self.abbiegephase_gestartet and not self.abgeschlossen:
                self.pub_info.publish(Int32(4))
                if self.chosen_direction == "links":
                    rospy.loginfo_throttle(1, "[Abbiegen] Linksabbiegen aktiv")
                    if filtered_contours_red:
                        leftmost = min(filtered_contours_red, key=lambda cnt: cv2.boundingRect(cnt)[0])
                        x, y, w, h = cv2.boundingRect(leftmost)
                        y_clamped = max(100, min(y, 400))
                        offset = int(np.interp(y_clamped, [100, 300], [250, 0]))
                        target_x = x + w + offset + 150
                        if self.debug:
                            cv2.circle(frame, (target_x, y), 6, (255, 0, 255), -1)
                    else:
                        target_x = 200
                        rospy.logwarn_throttle(2, "[Abbiegen] Keine rote Kontur gefunden")
                    self.pub_target_x.publish(Int32(target_x))

                elif self.chosen_direction == "geradeaus":
                    rospy.loginfo_throttle(1, "[Abbiegen] Geradeaus aktiv")
                    if filtered_contours_red:
                        highest = min(filtered_contours_red, key=lambda cnt: cv2.boundingRect(cnt)[1])
                        x, y, w, h = cv2.boundingRect(highest)
                        contour_area = cv2.contourArea(highest)
                        if contour_area > 500:
                            y_clamped = max(100, min(y, 400))
                            offset = int(np.interp(y_clamped, [100, 300], [50, 120]))
                            target_x = x + w + offset
                            if self.debug:
                                cv2.circle(frame, (target_x, y), 6, (0, 255, 0), -1)
                        else:
                            rospy.logwarn_throttle(2, "[Abbiegen] Kontur zu klein")
                            target_x = 320
                    else:
                        target_x = 320
                        rospy.logwarn_throttle(2, "[Abbiegen] Keine Kontur gefunden")
                    self.pub_target_x.publish(Int32(target_x))
                
                elif self.chosen_direction == "rechts":
                    rospy.loginfo_throttle(1, "[Abbiegen] Rechtsabbiegen aktiv")

                    if filtered_contours_red:
                        # Rechtseste Kontur finden (größter x-Wert)
                        rightmost = max(filtered_contours_red, key=lambda cnt: cv2.boundingRect(cnt)[0])
                        x, y, w, h = cv2.boundingRect(rightmost)

                        # Nutze rechte untere Ecke der Box
                        corner_x = x + w
                        corner_y = y + h

                        # Offset basierend auf y-Distanz zur Kamera (weiter weg = kleiner)
                        y_clamped = max(100, min(corner_y, 400))
                        offset = int(np.interp(y_clamped, [100, 300], [0, 200]))

                        target_x = corner_x + offset - 200  # Nach links korrigieren für Rechtskurve

                        if self.debug:
                            cv2.circle(frame, (target_x, corner_y), 6, (0, 0, 255), -1)
                            cv2.putText(frame, f"Target (rechts)", (target_x - 30, corner_y - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                            cv2.putText(frame, f"Offset: {offset}", (target_x - 30, corner_y + 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    else:
                        target_x = 440  # Fallback weit rechts
                        rospy.logwarn_throttle(2, "[Abbiegen] Keine rote Kontur für Rechtsabbiegen gefunden")

                    self.pub_target_x.publish(Int32(target_x))



                # Richtungsabhängiger Zeitwert
                required_time = 1.0 if self.chosen_direction == "rechts" else 3.0

                if (
                    self.abbiege_start_time is not None and
                    (current_time - self.abbiege_start_time) >= required_time and
                    self.left_x_value is not None
                ):
                    rospy.loginfo("[Abbiegen] Weiße Linie erkannt – Abbiegevorgang abgeschlossen")
                    self.abbiegephase_gestartet = False
                    self.abgeschlossen = True
                    self.pub_info.publish(Int32(0))
                    self.left_x_value = None
                    self.abbiege_start_time = None
                    self.last_turn_completed_time = current_time
                    self.direction_already_published = False

            # --- NEU: Nur tiefste rote Box überwachen ---
            if not self.abbiegephase_gestartet and not self.waiting_at_line:
                if filtered_contours_red:
                    bottommost = max(filtered_contours_red, key=lambda cnt: cv2.boundingRect(cnt)[1])
                    x, y, w, h = cv2.boundingRect(bottommost)
                    if y > 210:
                        self.pub_info.publish(Int32(1))
                        if self.debug:
                            rospy.loginfo(f"[Info] Unterste rote Box bei y={y} → sende Int32(1)")
                else:
                    self.pub_info.publish(Int32(0))
                    if self.debug:
                        rospy.loginfo("[Info] Keine rote Box erkannt → sende Int32(0)")

            if self.debug:
                for cnt in filtered_contours_red:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.imshow("Rote Linien", frame)
                cv2.waitKey(1)

            rate.sleep()


if __name__ == "__main__":
    node = RedLineDetector(node_name="red_line_detector")
    node.run()
