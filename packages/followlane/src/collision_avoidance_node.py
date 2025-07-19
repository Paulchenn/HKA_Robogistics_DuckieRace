#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Range
from std_msgs.msg import String

class ToFCollisionAvoidanceNode:
    def __init__(self, node_name):
        # Initialisiere Node
        super(ToFCollisionAvoidanceNode, self).__init__(node_name=node_name)

        # Parameter
        self.threshold = 0.1  # Meter

        # TODO Publisher für das Stoppen hier fertig erstellen
        self.ToFInfo_pub = rospy.Publisher(queue_size=1)

        # Subscriber für den ToF Sensor
        self.sub_ToFSensor = rospy.Subscriber('/vl53l1x_node/range', Range, self.range_callback)

    def range_callback(self, msg):
        distance = msg.range
        if distance < self.threshold:
            rospy.logwarn("Abstand unter Threshold: %.2f m", distance)
            # TODO hier stop publishen
            self.ToFInfo_pub.publish(3)  # Beispiel: 3 für STOP
        else:
            #hier normal lane following, man muss dafür nichts publishen, oder?
            print()

if __name__ == '__main__':
    node = ToFCollisionAvoidanceNode()
    rospy.spin()
