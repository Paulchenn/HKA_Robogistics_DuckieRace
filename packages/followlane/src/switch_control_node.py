#!/usr/bin/env python3

import rospy
import os
from enum import Enum
from std_msgs.msg import Float64, Int32
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import Image

class ControlState(Enum):
    STOP            = 0
    INTERSECTION    = 1
    OBSTACLE        = 2
    PARKING         = 3
    LANE_SLOW       = 4
    LANE_NORMAL     = 5

class SwitchControlNode(DTROS):
    def __init__(self, node_name):
        super(SwitchControlNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)

        self._vehicle_name = os.environ['VEHICLE_NAME']
        self._state = ControlState.LANE_NORMAL

        # Debug
        self.debug = False  # Debug-Modus für Konsolenausgaben
        self.debug_run = True

        # Werte aus den anderen Nodes
        self.lane_x = None
        self.bypass_x = None
        self.intersection_x = None

        self.duckie_info = 0
        self.intersection_info = 0
        self.parking_info = 0

        self.tof_info = 0

        #Zeitstempel für Publisher
        self.duckie_info_time = None
        self.intersection_info_time = None
        self.parking_info_time = None
        self.tof_info_time = None



        # Publisher
        self.pub_selected_x = rospy.Publisher(
            f"/{self._vehicle_name}/control/selected_x", Float64, queue_size=1
        )
        self.pub_control = rospy.Publisher(
            f"/{self._vehicle_name}/switch/control", Int32, queue_size=1
        )

        # Subscriber: ToF Info (z. B. 3 = STOP, 0 = normal)
        rospy.Subscriber(
            f"/{self._vehicle_name}/tof/avoidance/info", Int32, self.cbToFInfo, queue_size=1
)
        # Subscriber: Lane-Following X
        rospy.Subscriber(
            f"/{self._vehicle_name}/detect/lane", Float64, self.cbLaneX, queue_size=1
        )

        # Subscriber: Duckie Bypass X
        rospy.Subscriber(
            f"/{self._vehicle_name}/detect/duckie/bypass_target", Float64, self.cbBypassX, queue_size=1
        )

        # Subscriber: Duckie Info (0 = nichts, 1 = langsam, 2 = ausweichen)
        rospy.Subscriber(
            f"/{self._vehicle_name}/detect/duckie/info", Int32, self.cbDuckieInfo, queue_size=1
        )

        # Subscriber: Intersection Info (0 = nichts, 1 = langsam, 3 = STOP, 4 = ABBIEGEN)
        rospy.Subscriber(
            f"/{self._vehicle_name}/abfrage_info", Int32, self.cbIntersectionInfo, queue_size=1
        )

        # Subscriber: Parking Info (0 = nichts, 1 = langsam, 5 = parken (in parking_node))
        rospy.Subscriber(
            f"/{self._vehicle_name}/detect/object/slow4park", Int32, self.cbParkingInfo, queue_size=1
        )

        # Subscriber: Intersection Target X
        rospy.Subscriber(
            f"/{self._vehicle_name}/target_x", Int32, self.cbIntersectionX, queue_size=1
        )

        rospy.on_shutdown(self.fnShutDown)

        self.checkYoloRunning = False

    def cbToFInfo(self, msg: Int32):
        self.tof_info = msg.data
        self.tof_info_time = rospy.Time.now()
        if self.debug:
            rospy.loginfo(f"[SWITCH] ToF Info: {self.tof_info}")


    def cbLaneX(self, msg: Float64):
        self.lane_x = msg.data
        if self.debug:
            rospy.loginfo(f"[SWITCH] Lane-X: {self.lane_x}")

    def cbBypassX(self, msg: Float64):
        self.bypass_x = msg.data
        if self.debug:
            rospy.loginfo(f"[SWITCH] Bypass-X: {self.bypass_x}")

    def cbDuckieInfo(self, msg: Int32):
        self.duckie_info = msg.data
        self.duckie_info_time = rospy.Time.now()
        if self.debug:
            rospy.loginfo(f"[SWITCH] Duckie Info: {self.duckie_info}")

    def cbIntersectionInfo(self, msg: Int32):
        self.intersection_info = msg.data
        self.intersection_info_time = rospy.Time.now()
        if self.debug:
            rospy.loginfo(f"[SWITCH] Intersection Info: {self.intersection_info}")

    def cbIntersectionX(self, msg: Int32):
        self.intersection_x = msg.data
        if self.debug:
            rospy.loginfo(f"[SWITCH] Intersection Target X: {self.intersection_x}")

    def cbParkingInfo(self, msg: Int32):
        self.parking_info = msg.data
        self.parking_info_time = rospy.Time.now()
        if self.debug:
            rospy.loginfo(f"[SWITCH] Parking Info: {self.parking_info}")

    def fnShutDown(self):
        rospy.loginfo("[SWITCH] Node wird heruntergefahren.")

    def is_recent(self, msg_time, max_age_sec=0.3):
        if msg_time is None:
            return False
        return (rospy.Time.now() - msg_time).to_sec() < max_age_sec


    def run(self):
        rate = rospy.Rate(20)

        if not self.checkYoloRunning:
            topic_name_1 = f"/{self._vehicle_name}/detect/object/image"
            rospy.wait_for_message(topic_name_1, Image, timeout=20.0)
            self.checkYoloRunning = True

        while not rospy.is_shutdown():
            selected_x = None

            # Duckie Info prüfen
            if not self.is_recent(self.duckie_info_time):
                self.duckie_info = 0
                if self.debug_run:
                    rospy.logwarn_once("[SWITCH] Duckie Info veraltet → zurück auf 0")

            # Intersection Info prüfen
            if not self.is_recent(self.intersection_info_time):
                self.intersection_info = 0
                if self.debug_run:
                    rospy.logwarn_once("[SWITCH] Intersection Info veraltet → zurück auf 0")

            # Parking Info prüfen
            if not self.is_recent(self.parking_info_time):
                self.parking_info = 0
                if self.debug_run:
                    rospy.logwarn_once("[SWITCH] Parking Info veraltet → zurück auf 0")
            
            # ToF Info prüfen
            if not self.is_recent(self.tof_info_time):
                self.tof_info = 0
                if self.debug_run:
                    rospy.logwarn_once("[SWITCH] ToF Info veraltet → zurück auf 0")



            # === Prioritätsbasierte Steuerung ===
            if self.intersection_info == 3 or self.tof_info == 3:
                self._state = ControlState.STOP
                selected_x = None
                if self.debug_run:
                    rospy.logwarn("[RUN] STOP – setze Geschwindigkeit auf 0")

            elif self.intersection_info == 4:
                self._state = ControlState.INTERSECTION
                if self.intersection_x is not None:
                    selected_x = float(self.intersection_x)
                    if self.debug_run:
                        rospy.loginfo("[RUN] INTERSECTION – verwende Intersection-X")

            elif self.duckie_info == 2:
                self._state = ControlState.OBSTACLE
                if self.bypass_x is not None:
                    selected_x = self.bypass_x
                    if self.debug_run:
                        rospy.loginfo("[RUN] OBSTACLE – verwende Bypass-X")

            elif self.parking_info == 5:
                self._state = ControlState.PARKING
                selected_x = None
                if self.debug_run:
                    rospy.logwarn("[RUN] PARKING – uebergebe Kontrolle an parking-node")

            elif self.duckie_info == 1 or self.parking_info == 1 or self.intersection_info == 1 or self.tof_info == 1:
                self._state = ControlState.LANE_SLOW
                if self.lane_x is not None:
                    selected_x = self.lane_x
                    if self.debug_run:
                        rospy.loginfo("[RUN] LANE_SLOW – verwende Lane-X")

            else:
                self._state = ControlState.LANE_NORMAL
                if self.lane_x is not None:
                    selected_x = self.lane_x
                    if self.debug_run:
                        rospy.loginfo("[RUN] LANE_NORMAL – verwende Lane-X")

            # === Publishing ===
            if selected_x is not None:
                self.pub_selected_x.publish(Float64(selected_x))
            else:
                if self.debug_run:
                    rospy.logwarn("[RUN] Kein gültiger selected_x verfügbar")

            # Aktuellen Steuerungsmodus publizieren
            self.pub_control.publish(Int32(self._state.value))

            rate.sleep()

if __name__ == '__main__':
    node = SwitchControlNode(node_name='switch_control_node')
    node.run()
