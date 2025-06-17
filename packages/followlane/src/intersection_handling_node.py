# import cv2
# import numpy as np
# import random
# import rospy
# from duckietown.dtros import DTROS, NodeType
# from std_msgs.msg import String  # Verwenden einer simplen Message für die Ausgabe
# from std_msgs.msg import Bool 
 
#  # Publisher für die Ergebnisse
# pub = rospy.Publisher(f"/{vehicle_name}/red_line_info", String, queue_size=10)
# sub_redline = rospy.Subscriber(f"/{vehicle_name}/stop_line_detected", Bool, process_image)


# def process_image(msg):
#     #if msg ==True:
#     #rospy.init_node("red_line_detector", anonymous=True)

       

# #TODO Andere Kamera-Zugriffe raussuchen
#     # Kamera-Stream öffnen
#     #cap = cv2.VideoCapture(0)  # Ändere die Quelle je nach Kamerainput
#     camera_topic = f"/{vehicle_name}/camera_node/image/compressed"
#     sub_image = rospy.Subscriber(camera_topic, CompressedImage, process_image, queue_size = 1)

#     #def process_image(msg):
#     # ROS-Bilddaten in ein NumPy-Array umwandeln
#     np_arr = np.frombuffer(msg.data, np.uint8)
    
#     # Bild mit OpenCV dekodieren
#     frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

#     if frame is None:
#         rospy.logwarn("Fehler beim Dekodieren des Bildes.")
#         return

#     # Bildgröße bestimmen
#     height, width, _ = frame.shape
#     half_height, half_width = height // 2, width // 2

#     # Bild in HSV umwandeln
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

#     # Rote Farbe im HSV-Bereich definieren
#     lower_red1 = np.array([0, 120, 70])
#     upper_red1 = np.array([10, 255, 255])
#     lower_red2 = np.array([170, 120, 70])
#     upper_red2 = np.array([180, 255, 255])

#     # Maske für rote Farbe erstellen
#     mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
#     mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
#     mask = mask1 + mask2

#     # Konturen finden
#     contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#     # Anzahl der erkannten roten Linien berechnen
#     num_red_lines = len(contours)
#     num_path_options = max(0, num_red_lines - 1)

#     # Variablen für Richtungsentscheidungen initialisieren
#     turn_straight = False
#     turn_right = False
#     turn_left = False

#     for contour in contours:
#         x, y, w, h = cv2.boundingRect(contour)

#         if x < half_width and y < half_height:
#             turn_straight = True
#         elif x >= half_width and y < half_height:
#             turn_right = True
#         elif x < half_width and y >= half_height:
#             turn_left = True

#     chosen_direction = None
#     if num_path_options == 3:
#         chosen_direction = random.choice(["left", "straight", "right"])
#         rospy.loginfo(f"Drei Optionen gefunden: Zufällig ausgewählt -> {chosen_direction}")
#     elif num_path_options == 2:
#         rospy.loginfo("Kreuzung mit zwei Optionen erkannt.")
#         if (turn_left and  turn_straight == True):
#             chosen_direction = random.choice(["left","straight"])
#         elif (turn_left and turn_right ==True):
#             chosen_direction = random.choice(["left","right"])
#         elif (turn_straight and turn_right==True):
#             chosen_direction = random.choice(["straight","right"])
#     elif num_path_options == 1:
#         rospy.loginfo("Nur eine Möglichkeit vorhanden.")
#         if turn_left ==True:
#             chosen_direction = "left"
#         elif turn_straight == True:
#             chosen_direction = "straight"
#         elif turn_right ==True:
#             chosen_direction = "right" 

#     #message = f"Linien: {num_red_lines}, Optionen: {num_path_options}, Turn Straight: {turn_straight}, Turn Right: {turn_right}, Turn Left: {turn_left}, Chosen: {chosen_direction}"
#     message = f"{chosen_direction}"
#     pub.publish(message)  # Nachricht senden

#     cv2.imshow("Original", frame)
#     cv2.imshow("Maske", mask)

#     #if cv2.waitKey(1) & 0xFF == ord('q'):
        

#     #cap.release()
#    # cv2.destroyAllWindows()
#     cv2.destroyAllWindows()

 
# if __name__ == "__main__":
#     try:
#         #process_image(msg)
#         rospy.init_node("red_line_detector", anonymous=True)
#         rospy.spin()
#     except rospy.ROSInterruptException:
#         pass

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

        # if frame is None:
        #     rospy.logwarn("Kein Bild erhalten.")
        #     return

        # # Bildverarbeitung (Rote Linien erkennen)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red = np.array([0, 120, 70])
        upper_red = np.array([10, 255, 255])
        mask = cv2.inRange(hsv, lower_red, upper_red)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
        num_red_lines = len(contours)
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
