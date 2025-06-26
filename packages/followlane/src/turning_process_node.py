import cv2
import numpy as np
import rospy
import os
import json
from std_msgs.msg import String
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage  # Import für das Kamerabild



class RedLineListener(DTROS):

    def __init__(self, node_name):
        super(RedLineListener, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)

        self._vehicle_name = os.environ['VEHICLE_NAME']
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"

        self.direction_redline_topic = f"/{self._vehicle_name}/red_line_info"
        self.direction_redline = rospy.Subscriber(self.direction_redline_topic, String, self.callback, queue_size=10)
        self.sub_image = rospy.Subscriber(self._camera_topic, CompressedImage, self.process_image, queue_size=1)

        self.left_point = None
        self.right_point = None

        rospy.loginfo("Red Line Listener Node Initialized")

    def callback(self, msg):
        data = json.loads(msg.data)
        direction = data["richtung"]
        left = data["left"]
        right = data["right"]
        self.left_point = (int(left[0]), int(left[1]))
        self.right_point = (int(right[0]), int(right[1]))

        rospy.loginfo_throttle(1, f"Empfangen: {direction} linker Punkt {self.left_point} rechter Punkt {self.right_point}")

    def process_image(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if self.left_point is not None and self.right_point is not None:
            self.ellipseForTurning(frame, self.left_point, self.right_point)

        cv2.imshow("Abbiegelinien", frame)
        cv2.waitKey(1)  # nicht 0, sonst blockiert es den Node


    def ellipseForTurning(self, frame, left, right):
        # Halbachsen als Tupel (x-Achse, y-Achse)
        left_axes = (150, 300)
        right_axes = (350, 80)

        color = (255, 0, 0)
        thickness = 2

        # Ellipse links (z.B. 0° bis 90°)
        cv2.ellipse(frame, center=left, axes=left_axes, angle=90, startAngle=200, endAngle=270, color=color, thickness=thickness)

        # Ellipse rechts (z.B. 90° bis 180°)
        cv2.ellipse(frame, center=right, axes=right_axes, angle=0, startAngle=180, endAngle=250, color=color, thickness=thickness)

        height, width = frame.shape[:2]
        third_height, half_width = height // 3, width // 2
        
        # Vertikale Linie für geradeaus fahren
        cv2.line(frame, (half_width, 400), (half_width, third_height), (0, 255, 0), 2)  # grün, 2 Pixel dick

    def publish_turn_direction(direction):





if __name__ == "__main__":
    #rospy.init_node("red_line_listener")
    listener = RedLineListener(node_name='turning_process_node')
    rospy.spin()
