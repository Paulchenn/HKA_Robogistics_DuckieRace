#!/usr/bin/env python3
import rospy
import os
from sensor_msgs.msg import Range
from std_msgs.msg import Int32  # oder String, je nachdem was du willst

class ToFCollisionAvoidanceNode:
    def __init__(self):
        rospy.init_node('tof_collision_avoidance_node')
        self.threshold = 0.1  # Meter
        self.distance = None
        self.ToFInfo_pub = rospy.Publisher('/tof/avoidance/info', Int32, queue_size=1)
        rospy.Subscriber('/vl53l1x_node/range', Range, self.range_callback)

        self.ok_count = 0                # wie oft hintereinander Abstand > threshold
        self.ok_required = 5             # wie viele Messungen > threshold für "OK"

    def range_callback(self, msg):
        self.distance = msg.range

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.distance is not None:
                if self.distance < self.threshold:
                    self.ok_count = 0  # reset bei STOP
                    rospy.logwarn(f"[ToF] STOP – Abstand unter Threshold: {self.distance:.2f} m")
                    self.ToFInfo_pub.publish(Int32(3))
                else:
                    self.ok_count += 1
                    if self.ok_count >= self.ok_required:
                        rospy.loginfo_throttle(5, f"[ToF] OK – Abstand: {self.distance:.2f} m")
                        self.ToFInfo_pub.publish(Int32(0))
            rate.sleep()


if __name__ == '__main__':
    node = ToFCollisionAvoidanceNode()
    node.run()


