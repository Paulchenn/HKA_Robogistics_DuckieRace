#!/usr/bin/env python3

import rospy
import os
from enum import Enum
from std_msgs.msg import Int32
from duckietown_msgs.msg import Twist2DStamped
from duckietown.dtros import DTROS, NodeType

class ControlType(Enum):
    Lane = 1
    Obstacle = 2

class SwitchControlNode(DTROS):
    def __init__(self, node_name):
        super(SwitchControlNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)

        self._vehicle_name = os.environ['VEHICLE_NAME']
        self._control_mode = ControlType.Lane
        self._last_duckie_twist_time = rospy.Time(0)
        rospy.on_shutdown(self.fnShutDown)

        self.duckie_twist = None
        self.lane_twist = None

        # Publisher: Steuerbefehl an Motor
        twist_topic_out = f"/{self._vehicle_name}/car_cmd_switch_node/cmd"
        self.pub_cmd = rospy.Publisher(twist_topic_out, Twist2DStamped, queue_size=1)

        # Publisher: Control-Modus
        self.pub_control = rospy.Publisher(f"/{self._vehicle_name}/switch/control", Int32, queue_size=1)

        # Subscriber: Twist von Bypass (Obstacle)
        self.sub_duckie_twist = rospy.Subscriber(
            f"/{self._vehicle_name}/detect/duckie/bypass_cmd",
            Twist2DStamped,
            self.cbDuckieTwist,
            queue_size=1
        )

        # Subscriber: Twist von Lane Following
        self.sub_lane_twist = rospy.Subscriber(
            f"/{self._vehicle_name}/car_cmd/lane",
            Twist2DStamped,
            self.cbLaneTwist,
            queue_size=1
        )

    def cbDuckieTwist(self, msg: Twist2DStamped):
        """ Twist von BypassDuckieNode empfangen. """
        self.duckie_twist = msg
        if msg.v != 0.0 or msg.omega != 0.0:
            self._last_duckie_twist_time = rospy.Time.now()
            rospy.loginfo("[SWITCH] Twist von Bypass empfangen → Duckie erkannt")

    def cbLaneTwist(self, msg: Twist2DStamped):
        """ Twist von ControlLaneNode empfangen. """
        self.lane_twist = msg

    
    def fnShutDown(self):
        rospy.loginfo("[SHUTDOWN] Node wird beendet – Stoppe Fahrzeug.")
        stop_msg = Twist2DStamped()
        stop_msg.v = 0.0
        stop_msg.omega = 0.0
        for _ in range(5):
            self.pub_cmd.publish(stop_msg)
            rospy.sleep(0.1)

    def run(self):
            rate = rospy.Rate(10)
            while not rospy.is_shutdown():
                now = rospy.Time.now()
                delta = (now - self._last_duckie_twist_time).to_sec()

                if self.duckie_twist and delta < 2.0:
                    # Priorität: Obstacle
                    self._control_mode = ControlType.Obstacle
                    self.pub_cmd.publish(self.duckie_twist)
                elif self.lane_twist:
                    # Fallback: Lane Following
                    self._control_mode = ControlType.Lane
                    self.pub_cmd.publish(self.lane_twist)
                else:
                    # Sicherheit: Stop
                    stop_msg = Twist2DStamped()
                    stop_msg.v = 0.0
                    stop_msg.omega = 0.0
                    self.pub_cmd.publish(stop_msg)

                self.pub_control.publish(Int32(self._control_mode.value))
                rate.sleep()
                

if __name__ == '__main__':
    node = SwitchControlNode(node_name='switch_control_node')
    node.run()
