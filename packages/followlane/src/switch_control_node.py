#!/usr/bin/env python3

import rospy
import os
from enum import Enum
from std_msgs.msg import Float64, Int32, Bool
from duckietown.dtros import DTROS, NodeType

class ControlType(Enum):
    Lane = 1
    Obstacle = 2
    Turn = 3

class SwitchControlNode(DTROS):
    def __init__(self, node_name):
        super(SwitchControlNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)

        self._vehicle_name = os.environ['VEHICLE_NAME']
        self._control_mode = ControlType.Lane
        self.duckie_info = 0  # 0 = nichts, 1 = Duckie fern, 2 = Duckie nah
        self.redline_info = False #True = rote Linie erkannt, False = keine rote Linie

        self.lane_x = None
        self.bypass_x = None
        self.turn_x = None

        self.debug = False
        self.debug_run = False

        rospy.on_shutdown(self.fnShutDown)

        # Publisher: Aktuell ausgewählter X-Wert (neu angepasstes Topic)
        self.pub_selected_x = rospy.Publisher(
            f"/{self._vehicle_name}/control/selected_x", Float64, queue_size=1
        )

        # Publisher: Control-Modus
        self.pub_control = rospy.Publisher(
            f"/{self._vehicle_name}/switch/control", Int32, queue_size=1
        )

        # Subscriber: X-Wert von Lane-Following
        self.sub_lane_x = rospy.Subscriber(
            f"/{self._vehicle_name}/detect/lane", Float64, self.cbLaneX, queue_size=1
        )

        # Subscriber: X-Wert vom Bypass-Manöver
        self.sub_bypass_x = rospy.Subscriber(
            f"/{self._vehicle_name}/detect/duckie/bypass_target", Float64, self.cbBypassX, queue_size=1
        )

        # Subscriber: Duckie-Erkennungsstatus
        self.sub_duckie_info = rospy.Subscriber(
            f"/{self._vehicle_name}/detect/duckie/info", Int32, self.cbDuckieInfo, queue_size=1
        )

        # Subscriber für die Stop-Line-Erkennung
        self.sub_redline = rospy.Subscriber(f"/{self._vehicle_name}/stop_line_detected",
                                            Bool, self.cbRedline, queue_size=1)
        
        #Subscriber: X-Wert vom Abbiegen
        self.sub_turn_direction = rospy.Subscriber(f"/{self._vehicle_name}/zielpunkt", Int32, self.cbTurnInfo,queue_size=1)

        

    def cbLaneX(self, msg: Float64):
        self.lane_x = msg.data
        if self.debug:
            rospy.loginfo(f"[SWITCH] Lane-X empfangen: {self.lane_x}")

    def cbBypassX(self, msg: Float64):
        self.bypass_x = msg.data
        if self.debug:
            rospy.loginfo(f"[SWITCH] Bypass-X empfangen: {self.bypass_x}")

    def cbDuckieInfo(self, msg: Int32):
        self.duckie_info = msg.data
        if self.debug:
            rospy.loginfo(f"[SWITCH] Duckie-Status: {self.duckie_info}")

    def cbRedline(self, msg: Bool):
        self.redline_info = msg.data
        if self.debug:
            rospy.loginfo(f"[SWITCH] Rote Linien Status {self.redline_info}")

    def cbTurnInfo(self, msg: Int32):
        self.turn_x = msg.data
        if self.debug:
            rospy.loginfo(f"[SWITCH] Abbiegen Status {self.turn_x}")
    
    def fnShutDown(self):
        if self.debug:
            rospy.loginfo("[SHUTDOWN] Node wird beendet.")

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            selected_x = None

            if self.debug_run:
                rospy.loginfo(f"[RUN] Duckie-Info: {self.duckie_info}")

            if self.duckie_info == 2:
                self._control_mode = ControlType.Obstacle
                if self.bypass_x is not None:
                    selected_x = self.bypass_x
                    if self.debug_run:
                        rospy.loginfo("[RUN] Obstacle-Modus – verwende Bypass-X")

            elif self.duckie_info == 1:
                self._control_mode = ControlType.Lane
                if self.lane_x is not None:
                    selected_x = self.lane_x
                    if self.debug_run:
                        rospy.loginfo("[RUN] Duckie fern – verwende Lane-X (langsam)")

            elif self.redline_info == True:
                self._control_mode = ControlType.Turn #Turning
                if self.turn_x is not None:
                    selected_x = self.turn_x
                    if self.debug_run:
                        rospy.loginfo(f"[RUN] Abbiegen – verwende Turn-X {self.turn_x}")
            
            elif self.duckie_info == 0:
                self._control_mode = ControlType.Lane
                if self.lane_x is not None:
                    selected_x = self.lane_x
                    if self.debug_run:
                        rospy.loginfo("[RUN] Normal – verwende Lane-X")

            if selected_x is not None:
                self.pub_selected_x.publish(Float64(selected_x))
            else:
                if self.debug_run:
                    rospy.logwarn("[RUN] Kein X-Wert verfügbar → nichts publiziert")

            self.pub_control.publish(Int32(self._control_mode.value))
            rate.sleep()

if __name__ == '__main__':
    node = SwitchControlNode(node_name='switch_control_node')
    node.run()
