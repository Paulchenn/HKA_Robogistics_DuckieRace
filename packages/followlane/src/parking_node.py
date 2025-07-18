import os
import rospy
import threading
import yaml

from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import Twist2DStamped
from std_msgs.msg import Int32, Float64MultiArray


class ParkingNode(DTROS):
    def __init__(self, node_name):
        super(ParkingNode, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)
        self._vehicle_name = os.environ['VEHICLE_NAME']

        # State machine
        self.state = "IDLE"
        self.last_slot = None
        self.transition_lock = threading.Lock()

        # Subscribers
        self.sub_slot = rospy.Subscriber(f"/{self._vehicle_name}/detect/object/parkingBB", Float64MultiArray, self.cbSlot, queue_size=1)

        # Publisher for slow-down state
        self.pub_state = rospy.Publisher(f"/{self._vehicle_name}/detect/object/slow4park", Int32, queue_size=1)

        # Start in IDLE
        self.publish_state(0)

        # Read config file
        with open('packages/followlane/config/detect_duckieBotSlot.yaml', 'r') as f:
            self.conf = yaml.safe_load(f)


    def publish_state(self, val):
        self.pub_state.publish(Int32(val))


    def cbSlot(self, msg):
        if not msg.data or len(msg.data) != 4:
            return

        x1, y1, x2, y2 = msg.data

        with self.transition_lock:
            if self.state == "IDLE":
                # Slot is far right → start slow driving
                if x1 > 360 and y2 > 220:
                    self.state = "SLOW"
                    self.publish_state(1)

            elif self.state == "SLOW":
                if x1 > 600:  # Slot fast ganz rechts → vorbei
                    self.state = "PARKING"
                    self.publish_state(5)
                    if self.conf['debugPrints']:
                        rospy.loginfo("Start parking")
                    self._start_parking()

            elif self.state in ["WAIT", "EXIT"]:
                pass  # handled by timers

    def _start_parking(self):
        # Rückwärts mit Lenkeinschlag nach rechts (omega > 0)
        reverse_turn = Twist2DStamped(v=-0.2, omega=2.0)
        self.pub_lane_twist.publish(reverse_turn)

        # Nach 2 Sekunden gerade zurück
        rospy.Timer(rospy.Duration(2), self._straight_back, oneshot=True)

    def _straight_back(self, event):
        # Gerade zurückfahren
        reverse_straight = Twist2DStamped(v=-0.2, omega=0.0)
        self.pub_lane_twist.publish(reverse_straight)

        # Nach 1 Sekunde stoppen
        rospy.Timer(rospy.Duration(1), self._end_parking, oneshot=True)

    def _end_parking(self, event):
        if self.conf['debugPrints']:
            rospy.loginfo("Done parking, switching to WAIT")

        # Stop
        stop_msg = Twist2DStamped(v=0.0, omega=0.0)
        self.pub_lane_twist.publish(stop_msg)

        with self.transition_lock:
            self.state = "WAIT"
            rospy.Timer(rospy.Duration(3), self._do_exit, oneshot=True)

    def _do_exit(self, event):
        if self.conf['debugPrints']:
            rospy.loginfo("Exiting parking")

        # Optional: leicht vorwärts zum Ausparken
        forward_msg = Twist2DStamped(v=0.2, omega=-1.0)
        self.pub_lane_twist.publish(forward_msg)
        rospy.sleep(1.0)

        # Stop again
        stop_msg = Twist2DStamped(v=0.0, omega=0.0)
        self.pub_lane_twist.publish(stop_msg)

        with self.transition_lock:
            self.state = "IDLE"
            self.publish_state(0)
