#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float64, Int32
from duckietown_msgs.msg import Twist2DStamped
import os
from duckietown.dtros import DTROS, NodeType
from switch_control_node import ControlType

class ControlLaneNode(DTROS):
    def __init__(self, node_name):
        super(ControlLaneNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        
        self.enable = False
        self._vehicle_name = os.environ['VEHICLE_NAME']

        # Publisher an den car_cmd_switch_node
        twist_topic = f"/{self._vehicle_name}/car_cmd_switch_node/cmd"
        self.pub_cmd_vel = rospy.Publisher(twist_topic, Twist2DStamped, queue_size=1)

        # Subscriber für Zielspurpunkt (target_x) und Modus
        self.sub_lane = rospy.Subscriber(f'/{self._vehicle_name}/detect/lane', Float64, self.cbFollowLane, queue_size=1)
        self.sub_control = rospy.Subscriber(f"/{self._vehicle_name}/switch/control", Int32, self.cbControl, queue_size=1)

        # PID-Parameter
        self.kp = 7.5
        self.ki = 0.01
        self.kd = 0.3
        self.alpha = 0.9  # Glättungsfaktor (0 = sehr glatt, 1 = keine Glättung)

        self.filtered_error = 0.0
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = rospy.Time.now()

        # Debugging
        self.print_counter = 0

        rospy.on_shutdown(self.fnShutDown)

    def cbControl(self, msg):
        self.enable = (msg.data == ControlType.Lane.value)

    def cbFollowLane(self, desired_center):
        if not self.enable:
            print("[INFO] Lane following not enabled")
            return

        if self.print_counter % 100 == 0:
            print(f"[INFO] Received target_x: {desired_center.data}")

        center = desired_center.data
        self.followLane(center)

    def followLane(self, center):
        image_center = 640 / 2
        raw_error = (image_center - center) / image_center

        current_time = rospy.Time.now()
        dt = (current_time - self.last_time).to_sec()
        self.last_time = current_time

        # Glättung
        self.filtered_error = self.alpha * raw_error + (1 - self.alpha) * self.filtered_error
        error = self.filtered_error

        # Integral-Anteil
        self.integral += error * dt

        # Differential-Anteil
        derivative = (error - self.last_error) / dt if dt > 0 else 0.0
        self.last_error = error

        # PID-Regler
        omega = self.kp * error + self.ki * self.integral + self.kd * derivative
        omega = max(min(omega, 6.0), -6.0)

        print(omega)

        v = 0.25

        self.print_counter += 1
        if self.print_counter % 100 == 0:
            print(f"[PID TEST] P={self.kp*error:.3f}, I={self.ki*self.integral:.3f}, D={self.kd*derivative:.3f}, omega={omega:.3f}, dt={dt:.3f}")

        #if you dont want to drive yet leave as text
        #twist = Twist2DStamped(v=v, omega=omega)
        #self.pub_cmd_vel.publish(twist)

    def fnShutDown(self):
        rospy.loginfo("Shutting down. Sending stop command...")
        stop_msg = Twist2DStamped(v=0.0, omega=0.0)

        for _ in range(5):
            self.pub_cmd_vel.publish(stop_msg)
            rospy.sleep(0.1)

if __name__ == '__main__':
    node = ControlLaneNode(node_name='control_lane_node')
    rospy.spin()
