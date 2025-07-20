#!/usr/bin/env python3
import rospy
import os
from sensor_msgs.msg import Range
from std_msgs.msg import Int32

class ToFCollisionAvoidanceNode:
    def __init__(self):
        rospy.init_node('tof_collision_avoidance_node')

        # === Parameter ===
        self.threshold = 0.2  # Meter
        self.debug = rospy.get_param("~debug", True)  # Debug-Modus
        self.vehicle_name = os.environ.get("VEHICLE_NAME", "default_bot")  # Bot-Name

        # === Variablen ===
        self.distance = None
        self.ok_count = 0
        self.ok_required = 5

        # === Publisher ===
        self.ToFInfo_pub = rospy.Publisher(f'/{self.vehicle_name}/tof/avoidance/info', Int32, queue_size=1)

        # === Dynamischer Subscriber ===
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
                    self.ok_count = 0
                    if self.debug:
                        rospy.logwarn(f"[ToF] STOP – Abstand unter Threshold: {self.distance:.2f} m")
                    self.ToFInfo_pub.publish(Int32(3))
                else:
                    self.ok_count += 1
                    if self.ok_count >= self.ok_required:
                        if self.debug:
                            rospy.loginfo_throttle(5, f"[ToF] OK – Abstand: {self.distance:.2f} m (ok_count = {self.ok_count})")
                        self.ToFInfo_pub.publish(Int32(0))
            rate.sleep()



if __name__ == '__main__':
    node = ToFCollisionAvoidanceNode()
    node.run()
