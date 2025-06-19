import cv2
import numpy as np
import rospy
import os
from std_msgs.msg import String
from duckietown.dtros import DTROS, NodeType



class RedLineListener(DTROS):

    
    def __init__(self, node_name):
        super(RedLineListener, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)

        self._vehicle_name = os.environ['VEHICLE_NAME']
        self.direction_redline_topic = f"/{self._vehicle_name}/red_line_info"
        #self.vehicle_name = rospy.get_param("~vehicle_name", "default_name")  # oder direkt setzen
        #topic = f"/{self.vehicle_name}/red_line_info"
        self.direction_redline = rospy.Subscriber(self.direction_redline_topic, String, self.callback, queue_size=10)
        #rospy.loginfo(f"Subscribed to: {f"/{self.vehicle_name}/red_line_info"}")

        rospy.loginfo("Red Line Listener Node Initialized")


    def callback(self, msg):
        print(msg)
        rospy.loginfo(f"Empfangen: {msg.data}")

if __name__ == "__main__":
    #rospy.init_node("red_line_listener")
    listener = RedLineListener(node_name='turning_process_node')
    rospy.spin()








# def detect_intersection(frame, msg):

#     sub = rospy.Subscriber(f"/{self.vehicle_name}/red_line_info", String, detect_intersection)
#     #""" Erkennung von Linien und Kreuzungseckpunkten """
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     edges = cv2.Canny(gray, 50, 150)

#     lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)

#     intersection_points = []
#     if lines is not None:
#         for i in range(len(lines)):
#             for j in range(i+1, len(lines)):
#                 x1, y1, x2, y2 = lines[i][0]
#                 x3, y3, x4, y4 = lines[j][0]

#                 # Berechnung des Schnittpunkts der Linien
#                 A = np.array([[x2-x1, y2-y1], [x4-x3, y4-y3]])
#                 b = np.array([x3-x1, y3-y1])
#                 if np.linalg.det(A) != 0:  # Falls die Linien nicht parallel sind
#                     t, s = np.linalg.solve(A, b)
#                     if 0 <= t <= 1 and 0 <= s <= 1:
#                         intersection_x = int(x1 + t * (x2 - x1))
#                         intersection_y = int(y1 + t * (y2 - y1))
#                         intersection_points.append((intersection_x, intersection_y))

#     return intersection_points

# def visualize_path(frame, intersection_points, msg):
# #     """ Zeichnet die berechnete Linie im Kamerabild """
#     if len(intersection_points) >= 2:
#         mid_x = (intersection_points[0][0] + intersection_points[1][0]) // 2
#         mid_y = (intersection_points[0][1] + intersection_points[1][1]) // 2

#         for point in intersection_points:
#             cv2.circle(frame, point, 5, (0, 255, 0), -1)  # Eckpunkte markieren

#         cv2.circle(frame, (mid_x, mid_y), 7, (255, 0, 0), -1)  # Mittelpunkt markieren
    
#         # Berechnung des Fahrwegs (hier berechnen)
#         path_start = (mid_x, mid_y)  
#         path_end = (intersection_points[0]) if intersection_points[0][0] < mid_x else intersection_points[1]
#         cv2.line(frame, path_start, path_end, (255, 255, 0), 2)  # Linie zeichnen

#         if msg == "left":
#             print("Ich biege jetzt links ab")
#             #Funktion für Linksabbiegen
#         elif msg == "straight":
#             # Funktion für Geradeausfahren
#             print("Ich biege jetzt nicht ab")
#         elif msg == "right":
#             # Funktion für Rechtsabbiegen
#             print("Ich biege jetzt rechts ab")


#     return frame

# # Kamera-Stream öffnen
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     intersection_points = detect_intersection(frame)
#     frame = visualize_path(frame, intersection_points)

#     cv2.imshow("Duckiebot Navigation", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()