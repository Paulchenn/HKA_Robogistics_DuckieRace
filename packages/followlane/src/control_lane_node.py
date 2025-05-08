#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float64, Int32
from duckietown_msgs.msg import Twist2DStamped
import os
from duckietown.dtros import DTROS, NodeType
from switch_control_node import ControlType

class ControlLaneNode(DTROS):
    def __init__(self,node_name):
        super(ControlLaneNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        
        self.enable = False
        self._vehicle_name = os.environ['VEHICLE_NAME']

        # Publisher an den car_cmd_switch_node
        twist_topic = f"/{self._vehicle_name}/car_cmd_switch_node/cmd"
        self.pub_cmd_vel = rospy.Publisher(twist_topic, Twist2DStamped, queue_size=1)

        # Subscriber für den Linienschwerpunkt und Control-Modus
        self.sub_lane = rospy.Subscriber(f'/{self._vehicle_name}/detect/lane', Float64, self.cbFollowLane, queue_size=1)
        self.sub_control = rospy.Subscriber(f"/{self._vehicle_name}/switch/control", Int32, self.cbControl , queue_size=1)

        # PID-Parameter
        self.kp = 2.0
        self.ki = 0.0
        self.kd = 0.5

        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = rospy.Time.now()
        
        rospy.on_shutdown(self.fnShutDown)

    def cbControl(self, msg):
        self.enable = (msg.data == ControlType.Lane.value)

    def cbFollowLane(self, desired_center):
        if not self.enable:
            return

        center = desired_center.data
        self.followLane(center)

    def followLane(self, center):
        image_center = 640 / 2  # Bildmitte
        error = (center - image_center) / image_center

        current_time = rospy.Time.now()
        dt = (current_time - self.last_time).to_sec()
        self.last_time = current_time

        # PID-Berechnung
        self.integral += error * dt
        derivative = (error - self.last_error) / dt if dt > 0 else 0
        self.last_error = error

        omega = self.kp * error + self.ki * self.integral + self.kd * derivative
        v = 0.2

        twist = Twist2DStamped(v=v, omega=omega)
        self.pub_cmd_vel.publish(twist)

        print(f"[PID] v={v:.2f}, omega={omega:.2f}, error={error:.2f}, dt={dt:.3f}")

    def fnShutDown(self):
        rospy.loginfo("Shutting down. cmd_vel will be 0")
        self.pub_cmd_vel.publish(Twist2DStamped(v=0.0, omega=0.0))

if __name__ == '__main__':
    node = ControlLaneNode(node_name='control_lane_node')
    rospy.spin()
