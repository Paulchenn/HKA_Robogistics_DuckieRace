#!/usr/bin/env python3
import rospy
import os
from sensor_msgs.msg import Range
from std_msgs.msg import Int32

class ToFCollisionAvoidanceNode:
    def __init__(self):
        rospy.init_node('tof_collision_avoidance_node')

        # === Parameter ===
        self.threshold = 0.2          # Meter: STOP ab hier
        self.slow_threshold = 0.4     # Meter: LANGSAM zwischen 0.2 und 0.4

        self.debug = rospy.get_param("~debug", True)
        self.vehicle_name = os.environ.get("VEHICLE_NAME", "default_bot")

        # === Variable ===
        self.distance = None

        # === Publisher ===
        self.ToFInfo_pub = rospy.Publisher(
            f'/{self.vehicle_name}/tof/avoidance/info', Int32, queue_size=1
        )

        # === Subscriber ===
        tof_topic = f'/{self.vehicle_name}/front_center_tof_driver_node/range'
        rospy.Subscriber(tof_topic, Range, self.range_callback, queue_size=1)

        if self.debug:
            rospy.loginfo(f"[ToF] Node gestartet für Fahrzeug '{self.vehicle_name}' mit Topic: {tof_topic}")

    def range_callback(self, msg):
        self.distance = msg.range
        if self.debug:
            rospy.loginfo(f"[ToF] Neue Messung: {self.distance:.2f} m")

    def run(self):
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            if self.distance is not None:
                if self.distance < self.threshold:
                    if self.debug:
                        rospy.logwarn(f"[ToF] STOP – Abstand unter Threshold: {self.distance:.2f} m")
                    self.ToFInfo_pub.publish(Int32(3))

                elif self.distance < self.slow_threshold:
                    if self.debug:
                        rospy.loginfo(f"[ToF] LANGSAM – Abstand kritisch: {self.distance:.2f} m")
                    self.ToFInfo_pub.publish(Int32(1))

                else:
                    # Kein Publish → switch_control_node übernimmt Rückfall auf 0
                    if self.debug:
                        rospy.loginfo_throttle(5, f"[ToF] OK – Abstand: {self.distance:.2f} m")

            rate.sleep()

if __name__ == '__main__':
    node = ToFCollisionAvoidanceNode()
    node.run()
