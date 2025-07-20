#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Range
from std_msgs.msg import Int32  # oder String, je nachdem was du willst

class ToFCollisionAvoidanceNode:
    def __init__(self):
        # Initialisiere Node
        rospy.init_node('tof_collision_avoidance_node')

        # Parameter
        self.threshold = 0.1  # Meter
        self.distance = None  # letzte empfangene Distanz

        # Publisher für Stop-Information (hier Int32, z.B. 3 = STOP)
        self.ToFInfo_pub = rospy.Publisher('/tof/avoidance/info', Int32, queue_size=1)

        # Subscriber für den ToF Sensor
        rospy.Subscriber('/vl53l1x_node/range', Range, self.range_callback)

    def range_callback(self, msg):
        self.distance = msg.range  # Speichere aktuelle Distanz

    def run(self):
        rate = rospy.Rate(10)  # 10 Hz Loop
        while not rospy.is_shutdown():
            if self.distance is not None:
                if self.distance < self.threshold:
                    rospy.logwarn(f"[ToF] STOP – Abstand unter Threshold: {self.distance:.2f} m")
                    self.ToFInfo_pub.publish(Int32(3))  # 3 = STOP
                else:
                    # Kein Stop nötig → Lane-Following kann weiterlaufen
                    rospy.loginfo_throttle(5, f"[ToF] OK – Abstand: {self.distance:.2f} m")
            rate.sleep()

if __name__ == '__main__':
    node = ToFCollisionAvoidanceNode()
    node.run()

