import cv2
import yaml
import numpy as np
import random
import os
import json
import rospy
from duckietown.dtros import DTROS, NodeType
from std_msgs.msg import String, Bool, Int32
from sensor_msgs.msg import CompressedImage
import time

class RedLineDetector(DTROS):
    def __init__(self, node_name):
        super(RedLineDetector, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)

        self._vehicle_name = os.environ['VEHICLE_NAME']
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        self._redLine_topic = f"/{self._vehicle_name}/stop_line_detected"
        self._config_path = 'packages/followlane/config/detect_lane.yaml'
        with open(self._config_path, 'r') as f:
            self.conf = yaml.safe_load(f)

        self.pub_red_line_info = rospy.Publisher(f"/{self._vehicle_name}/red_line_info", String, queue_size=10)
        self.pub_info = rospy.Publisher(f"/{self._vehicle_name}/abfrage_info", Int32, queue_size=10)
        self.pub_target_x = rospy.Publisher(f"/{self._vehicle_name}/target_x", Int32, queue_size=10)

        self.sub_image = rospy.Subscriber(self._camera_topic, CompressedImage, self.process_image, queue_size=1)
        self.sub_redline = rospy.Subscriber(self._redLine_topic, Bool, self.process_stop_line, queue_size=1)

        self.direction_already_published = False
        self.stop_line_detected = False
        self.waiting_at_line = False
        self.wait_start_time = None
        self.abbiegephase_gestartet = False
        self.abgeschlossen = False
        self.chosen_direction = None

        # Initialisiere Variablen für Abbiegevorgang
        self.no_red_frame_count = 0
        self.no_red_frame_threshold = 3


        self.debug = True
        if self.debug:
            rospy.loginfo("[RedLineDetector] Debug-Modus aktiviert")

    def process_stop_line(self, msg):
        if msg.data:
            if self.debug:
                rospy.loginfo_throttle(5, "Stoplinie erkannt")
            if not self.abbiegephase_gestartet and not self.waiting_at_line:
                self.stop_line_detected = True
                self.waiting_at_line = True
                self.wait_start_time = rospy.get_time()

    def process_image(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
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

        turn_left = turn_straight = turn_right = False
        options = []

        for contour in filtered_contours_red:
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w // 2
            center_y = y + h // 2

            if center_y < y_horizontal:
                if center_x <= fifth_width:
                    turn_left = True
                    options.append("links")
                elif center_x <= half_width:
                    turn_straight = True
                    options.append("geradeaus")
                else:
                    turn_right = True
                    options.append("rechts")

        if options and not self.direction_already_published:
            self.chosen_direction = "links"             
            #random.choice(options)
            data = {"richtung": self.chosen_direction}
            msg_out = String()
            msg_out.data = json.dumps(data)
            self.pub_red_line_info.publish(msg_out)
            self.direction_already_published = True
            rospy.loginfo("Richtung gewählt und gesendet: %s", self.chosen_direction)

        # === Kontrolllogik für Abbiegevorgang ===

        current_time = rospy.get_time()

        if self.waiting_at_line:
            # Während wir "stehen" (info=3)
            if current_time - self.wait_start_time < 2.0:
                self.pub_info.publish(Int32(3))  # publish info=3 kontinuierlich
                return
            else:
                self.waiting_at_line = False
                self.abbiegephase_gestartet = True
                rospy.loginfo("Abbiegevorgang wird gestartet")

        if self.abbiegephase_gestartet and not self.abgeschlossen:
            self.pub_info.publish(Int32(4))  # während Abbiegen

            if self.chosen_direction == "links":
                rospy.loginfo_throttle(1, "[Abbiegen] Linksabbiegen aktiv")

                if filtered_contours_red:
                    # Linkeste Box finden (kleinster x-Wert)
                    leftmost_contour = min(filtered_contours_red, key=lambda cnt: cv2.boundingRect(cnt)[0])
                    x, y, w, h = cv2.boundingRect(leftmost_contour)
                    target_x = x + w  # rechte obere Ecke (x + Breite)

                    if self.debug:
                        cv2.circle(frame, (target_x, y), 6, (255, 0, 255), -1)
                        cv2.putText(frame, "Target (links)", (target_x - 30, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

                    self.no_red_frame_count = 0  # Reset-Zähler
                else:
                    rospy.logwarn_throttle(2, "[Abbiegen] Keine rote Kontur für Linksabbiegen gefunden")
                    target_x = 200  # fallback
                    self.no_red_frame_count += 1

                self.pub_target_x.publish(Int32(target_x))

            elif self.chosen_direction == "geradeaus":
                rospy.loginfo_throttle(1, "[Abbiegen] Geradeaus aktiv")

                # TODO: Berechne target_x für Geradeausfahren
                target_x = 320  # Platzhalter

                if filtered_contours_red:
                    self.no_red_frame_count = 0
                else:
                    rospy.logwarn_throttle(2, "[Abbiegen] Keine rote Kontur für Geradeausfahren gefunden")
                    self.no_red_frame_count += 1

                self.pub_target_x.publish(Int32(target_x))

            elif self.chosen_direction == "rechts":
                rospy.loginfo_throttle(1, "[Abbiegen] Rechtsabbiegen aktiv")

                # TODO: Berechne target_x für Rechtsabbiegen
                target_x = 440  # Platzhalter

                if filtered_contours_red:
                    self.no_red_frame_count = 0
                else:
                    rospy.logwarn_throttle(2, "[Abbiegen] Keine rote Kontur für Rechtsabbiegen gefunden")
                    self.no_red_frame_count += 1

                self.pub_target_x.publish(Int32(target_x))

            else:
                rospy.logwarn_throttle(2, f"[Abbiegen] Unbekannte Richtung: {self.chosen_direction}")
                self.no_red_frame_count += 1

            # Abbruchbedingung – stabil durch 3 leere Frames
            if self.no_red_frame_count >= 3:
                rospy.loginfo("[Abbiegen] Keine roten Linien in 3 aufeinanderfolgenden Frames – Abbiegevorgang abgeschlossen")
                self.abbiegephase_gestartet = False
                self.abgeschlossen = True
                self.pub_info.publish(Int32(0))



        # === Debug Visualisierung ===
        if self.debug:
            for cnt in filtered_contours_red:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.imshow("Rote Linien", frame)
            cv2.waitKey(1)

if __name__ == "__main__":
    node = RedLineDetector(node_name="red_line_detector")
    rospy.spin()
