import cv2
import numpy as np
import random
import os
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

        # Publisher für die Ergebnisse
        self.pub_red_line_info = rospy.Publisher(f"/{self._vehicle_name}/red_line_info", String, queue_size=10)

        # Subscriber für das Kamerabild
        self.sub_image = rospy.Subscriber(self._camera_topic,
                                          CompressedImage, self.process_image, queue_size=1)

        # Subscriber für die Stop-Line-Erkennung
        self.sub_redline = rospy.Subscriber(self._redLine_topic,
                                            Bool, self.process_stop_line, queue_size=1)

    def process_image(self, msg):
        # Bild aus der Message dekodieren
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # # Bildverarbeitung (Rote Linien erkennen)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Bereich 1 – helleres und dunkleres Rot zulassen
        lower_red1 = np.array([0, 120, 120])
        upper_red1 = np.array([12, 255, 255])

        # Bereich 2 – auch rötliches Violett/Orange noch erwischen
        lower_red2 = np.array([165, 70, 70])
        upper_red2 = np.array([180, 255, 255])


        # Maske kombinieren
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        x_red = None

        contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_area = 150  # Mindestfläche in Pixeln

        filtered_contours_red = [cnt for cnt in contours_red if cv2.contourArea(cnt) > min_area]
        turning_options = len(filtered_contours_red) -1
        rospy.loginfo(f"Wegoptionen: {turning_options}")
        
        for cnt in filtered_contours_red:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)


        if contours_red:
            rospy.loginfo("Rote Kontur gefunden")
            # Größte Kontur verwenden (optional)
            largest_contour = max(contours_red, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            x_red = x + w // 2

            # Visualisierung Bounding Box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
            rospy.loginfo(f"Rote Kontur gefunden – x:{x}, y:{y}, w:{w}, h:{h}")

        height, width = frame.shape[:2]
        half_height, half_width = height // 2, width // 2
        quarter_height, fifth_width = height //4, width // 5

        # Horizontale Linie (von ganz links nach ganz rechts in der Mitte)
        cv2.line(frame, (0,(3*quarter_height) ), (width, (3*quarter_height)), (0, 255, 0), 2)  # grün, 2 Pixel dick

        # Vertikale Linie (von ganz oben nach ganz unten in der Mitte)
        cv2.line(frame, (fifth_width, 0), (fifth_width, height), (0, 255, 0), 2)  # grün, 2 Pixel dick
        cv2.line(frame, (half_width, 0), (half_width, height), (0, 255, 0), 2)  # grün, 2 Pixel dick


        turn_left = turn_straight = turn_right = False  # Initialisierung

        for contour in filtered_contours_red:
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w // 2
            center_y = y + h // 2

            # Nur obere Bildhälfte (oberhalb der waagerechten Linie)
            if center_y < 3 * quarter_height:

                if center_x < fifth_width:
                    turn_left = True
                    rospy.loginfo("Links")
                if fifth_width < center_x < half_width:
                    turn_straight = True
                    rospy.loginfo("Geradeaus")
                if center_x > half_width:
                    turn_right = True
                    rospy.loginfo("Rechts")
                rospy.loginfo(f"Links:{turn_left}, Mitte:{turn_straight}, Rechts:{turn_right}")



        num_red_lines = len(contours_red)
        num_filtered_red_lines = len(filtered_contours_red)
        rospy.loginfo(f"Linien: {num_red_lines}")
        rospy.loginfo(f"Gefilterte Linien: {num_filtered_red_lines}")
        red_pixels = cv2.countNonZero(mask_red)
        rospy.loginfo(f"Rote Pixel: {red_pixels}")
        message = f"Linien: {num_red_lines}"
        self.pub_red_line_info.publish(message)

        # Optional: Bild speichern statt `cv2.imshow()`
        cv2.imshow("/data/processed_image.jpg", frame)
        cv2.waitKey(1)

    def process_stop_line(self, msg):
        if msg.data:
            rospy.loginfo("Stoplinie erkannt, Verarbeitung läuft...")

if __name__ == "__main__":
    #rospy.init_node("red_line_detector", anonymous=True)
    node = RedLineDetector(node_name="red_line_detector")
    rospy.spin()
