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


