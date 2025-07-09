import cv2
import yaml
import numpy as np
import random
import os
import json
import rospy
from duckietown.dtros import DTROS, NodeType
from std_msgs.msg import String, Bool
from sensor_msgs.msg import CompressedImage  # Import für das Kamerabild

class RedLineDetector(DTROS):
    def __init__(self, node_name):
        super(RedLineDetector, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)

        self._vehicle_name = os.environ['VEHICLE_NAME']
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        self._redLine_topic = f"/{self._vehicle_name}/stop_line_detected"
        self._config_path = 'packages/followlane/config/detect_lane.yaml'
        with open(self._config_path, 'r') as f:
            self.conf = yaml.safe_load(f)

        # Publisher für die Ergebnisse
        self.pub_red_line_info = rospy.Publisher(f"/{self._vehicle_name}/red_line_info", String, queue_size=10)

        # Subscriber für das Kamerabild
        self.sub_image = rospy.Subscriber(self._camera_topic,
                                          CompressedImage, self.process_image, queue_size=1)

        # Subscriber für die Stop-Line-Erkennung
        self.sub_redline = rospy.Subscriber(self._redLine_topic,
                                            Bool, self.process_stop_line, queue_size=1)
        
        self.direction_already_published = False

        # Debug-Modus aktivieren über ROS-Parameter (z. B. in launch file oder param server)
        self.debug = True
        if self.debug:
            rospy.loginfo("[RedLineDetector] Debug-Modus aktiviert")

    def process_image(self, msg):
        # Bild aus der Message dekodieren
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        rd = self.conf['red']  # rot = rotHSV-Bereich

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
        turning_options = len(filtered_contours_red) - 1

        for cnt in filtered_contours_red:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

        x_red = None
        if contours_red:
            largest_contour = max(contours_red, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            x_red = x + w // 2
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

        height, width = frame.shape[:2]
        half_height, half_width = height // 2, width // 2
        quarter_height, fifth_width = height // 4, width // 5

        y_horizontal = 3 * quarter_height
        cv2.line(frame, (0, y_horizontal), (width, y_horizontal), (0, 255, 0), 2)

        left_intersection = (0, y_horizontal)
        right_intersection = (width, y_horizontal)

        if self.debug:
            rospy.loginfo_throttle(2, f"Schnittpunkte der horizontalen Linie: links={left_intersection}, rechts={right_intersection}")

        cv2.circle(frame, left_intersection, 5, (255, 0, 0), -1)
        cv2.circle(frame, right_intersection, 5, (255, 0, 0), -1)

        cv2.line(frame, (fifth_width, 0), (fifth_width, height), (0, 255, 0), 2)
        cv2.line(frame, (half_width, 0), (half_width, height), (0, 255, 0), 2)

        turn_left = turn_straight = turn_right = False
        options = []

        for contour in filtered_contours_red:
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w // 2
            center_y = y + h // 2
            cv2.circle(frame, (center_x, center_y), radius=1, color=(200, 255, 100), thickness=-1)

            if center_y < 3 * quarter_height:
                if center_x <= fifth_width:
                    turn_left = True
                    if "links" not in options:
                        options.append("links")
                if fifth_width < center_x <= half_width:
                    turn_straight = True
                    if "geradeaus" not in options:
                        options.append("geradeaus")
                if center_x > half_width:
                    turn_right = True
                    if "rechts" not in options:
                        options.append("rechts")

        if self.debug:
            rospy.loginfo_throttle(1, f"Links:{turn_left}, Mitte:{turn_straight}, Rechts:{turn_right}")

        if options:
            chosen_direction = random.choice(options)
            if self.debug:
                rospy.loginfo_throttle(2, f"Zufällig gewählte Richtung: {chosen_direction}")
            data = {"richtung": chosen_direction, "left": left_intersection, "right": right_intersection}
            msg = String()
            msg.data = json.dumps(data)

            if not self.direction_already_published:
                self.pub_red_line_info.publish(msg)
                self.direction_already_published = True
                if self.debug:
                    rospy.loginfo("Hier wird jetzt 1x die Richtung gepublished")

        if self.debug:
            num_red_lines = len(contours_red)
            num_filtered_red_lines = len(filtered_contours_red)
            red_pixels = cv2.countNonZero(mask_red)
            rospy.loginfo_throttle(2, f"Gefundene Linien: {num_red_lines}, Nach Filter: {num_filtered_red_lines}, Rote Pixel: {red_pixels}")

        if self.debug:
            cv2.imshow("Rote_Linien_erkennen", frame)
            cv2.waitKey(1)

    def process_stop_line(self, msg):
        if msg.data and self.debug:
            rospy.loginfo_throttle(5, "Stoplinie erkannt, Verarbeitung läuft...")

if __name__ == "__main__":
    node = RedLineDetector(node_name="red_line_detector")
    rospy.spin()
