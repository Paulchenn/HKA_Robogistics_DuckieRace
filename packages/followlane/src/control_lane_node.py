#!/usr/bin/env python3

import rospy
import os
import yaml
from std_msgs.msg import Float64, Int32
from duckietown_msgs.msg import Twist2DStamped
from duckietown.dtros import DTROS, NodeType
from switch_control_node import ControlType

class ControlLaneNode(DTROS):
    def __init__(self, node_name):
        super(ControlLaneNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)

        self._vehicle_name = os.environ['VEHICLE_NAME']
        self.enable = False
        self.debug = False  # Debug-Modus für Konsolenausgaben

        # Konfiguration laden
        config_path = 'packages/followlane/config/detect_lane.yaml'
        with open(config_path, 'r') as f:
            self.conf = yaml.safe_load(f)

        self.v_min = self.conf.get("v_min", 0.1)
        self.v_max = self.conf.get("v_max", 0.3)

        # Publisher – an SwitchControlNode weitergeleitetes Regler-Kommando
        lane_cmd_topic = f"/{self._vehicle_name}/car_cmd_switch_node/cmd"
        self.pub_lane_twist = rospy.Publisher(lane_cmd_topic, Twist2DStamped, queue_size=1)

        # Subscriber für Ziel-X vom SwitchControlNode
        self.sub_target_x = rospy.Subscriber(
            f"/{self._vehicle_name}/control/selected_x", Float64, self.cbFollowLane, queue_size=1
        )

        # Subscriber für aktuellen Steuerungsmodus (Lane oder Obstacle)
        self.sub_control = rospy.Subscriber(
            f"/{self._vehicle_name}/switch/control", Int32, self.cbControl, queue_size=1
        )

        # PID Parameter
        self.kp = 10
        self.ki = 0.1
        self.kd = 0.4

        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = rospy.Time.now()

        rospy.on_shutdown(self.fnShutDown)

    def cbControl(self, msg):
        self.enable = (msg.data == ControlType.Lane.value or msg.data == ControlType.Obstacle.value)
        if self.debug:
            rospy.loginfo(f"[CONTROL] Fahrmodus {'aktiviert' if self.enable else 'deaktiviert'} (Modus: {msg.data})")

    def cbFollowLane(self, desired_center):
        if not self.enable:
            if self.debug:
                rospy.loginfo("[CONTROL] Regelung deaktiviert – kein Kommando")
            return

        if self.debug:
            rospy.loginfo(f"[CONTROL] Empfange Ziel-X: {desired_center.data}")
        self.followLane(desired_center.data)

    def followLane(self, center):
        image_center = 640 / 2
        error = (image_center - center) / image_center  # Normalisierter Fehler [-1, 1]

        current_time = rospy.Time.now()
        dt = (current_time - self.last_time).to_sec()
        self.last_time = current_time

        # PID-Regelung
        self.integral += error * dt
        derivative = (error - self.last_error) / dt if dt > 0 else 0.0
        self.last_error = error

        omega = self.kp * error + self.ki * self.integral + self.kd * derivative
        omega = max(min(omega, 5.0), -5.0)

        # Geschwindigkeitsanpassung
        error_abs = min(abs(error), 1.0)
        v = self.v_max - (self.v_max - self.v_min) * error_abs

        # Befehl senden
        twist = Twist2DStamped()
        twist.header.stamp = rospy.Time.now()
        twist.v = v
        twist.omega = omega
        self.pub_lane_twist.publish(twist)

        if self.debug:
            rospy.loginfo(
                f"[PID] e={error:.3f}, P={self.kp * error:.3f}, I={self.ki * self.integral:.3f}, "
                f"D={self.kd * derivative:.3f}, ω={omega:.3f}, v={v:.3f}"
            )

    def fnShutDown(self):
        if self.debug:
            rospy.loginfo("[SHUTDOWN] Stoppe Fahrzeug")
        stop_msg = Twist2DStamped(v=0.0, omega=0.0)
        for _ in range(5):
            self.pub_lane_twist.publish(stop_msg)
            rospy.sleep(0.1)

if __name__ == '__main__':
    node = ControlLaneNode(node_name='control_lane_node')
    rospy.spin()
