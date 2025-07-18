import cv2
import os
import rospy
import threading
import yaml
import time

from cv_bridge import CvBridge
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import Twist2DStamped
from sensor_msgs.msg import Image
from std_msgs.msg import Int32, Float64MultiArray


class ParkingNode(DTROS):
    def __init__(self, node_name):
        super(ParkingNode, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)
        self._vehicle_name = os.environ['VEHICLE_NAME']

        # State machine
        self.state = "IDLE"
        self.last_slot = None
        self.transition_lock = threading.Lock()

        # === Subscribers ===
        # for BBox of nearest free slot
        self.sub_slot = rospy.Subscriber(f"/{self._vehicle_name}/detect/object/parkingBB", Float64MultiArray, self.cbSlot, queue_size=1)
        # for annotated Image
        self.sub_image = rospy.Subscriber(f"/{self._vehicle_name}/detect/object/image", Image, self.cb_image, queue_size=1)

        # === Publisher ===
        # for state (to switch_control_node)
        self.pub_state = rospy.Publisher(f"/{self._vehicle_name}/detect/object/slow4park", Int32, queue_size=1)
        # for driving command
        self.pub_lane_twist = rospy.Publisher(f"/{self._vehicle_name}/car_cmd_switch_node/cmd", Twist2DStamped, queue_size=1)

        # Start in IDLE
        self.publish_state(0)

        # Read config file
        with open('packages/followlane/config/detect_duckieBotSlot.yaml', 'r') as f:
            self.conf = yaml.safe_load(f)

        self.bridge = CvBridge()
        self.status_text = "Status: IDLE"

        self.curr_bbox = None

        self.time_lastBB = time.time()
        self.time_startPark = None
        self.time_inPark = None


    ##### ===== CALLBACK FUNCTIONS OF SUBSRIBERS ===== #####
    def cb_image(self, msg):
        '''
        Callback function for subscrier of annotated image (after YOLO).

        Args:
            self
            msg: annotaded image (with bounding boxes of detected stuff.

        Returns:
            none
        '''
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logerr(f"Could not convert image: {e}")
            return

        # Draw the status text on the image
        if self.status_text == "Status: SLOW --> approaching slot":
            myText = f"{self.status_text}, {self.delay_startPark:.2f}s"
        elif self.status_text == "Status: WAIT --> parked":
            if self.time_inPark is not None:
                self.delay_inPark = time.time()-self.time_inPark
            myText = f"{self.status_text}, {self.delay_inPark:.2f}s"
        else:
            myText = self.status_text

        cv2.putText(cv_image, myText, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Optional: show image (for debugging only, not on robot)
        if self.conf['show_image']:
            cv2.imshow("Parking Status", cv_image)
            cv2.waitKey(1)

        # Optionally: republish image with overlay if needed
        # img_out_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
        # self.pub_overlay.publish(img_out_msg)


    def cbSlot(self, msg):
        '''
        Callback function for subscrier of bounding boxes for free parking Slots.

        Args:
            self
            msg: contains x1, y1 for upper left corner and x2,y2 for lower right corner of last detected free parking Slot.

        Returns:
            none
        '''
        if not msg.data or len(msg.data) != 4 or not self.conf["go_parking"]:
            if self.conf["debugPrints_parking"]:
                rospy.loginfo(f"[PARKING] Wrong data or no parking activated")
            self.curr_bbox = None
            return
        else:
            self.curr_bbox = msg
            self.time_lastBB = time.time()
            if self.conf["debugPrints_parking"]:
                rospy.loginfo(f"[PARKING] Correct data and parking activated")



    ##### ===== OTHER FUNCTIONS ===== #####
    def publish_state(self, val):
        '''
        Publishing thecurrent state.
         
        Args:
            self
            val: state to publish.

        Returns:
            none
        '''
        self.pub_state.publish(Int32(val))


    def _start_parking(self):
        '''
        Start the parking process.
         
        Args:
            self

        Returns:
            none
        '''
        self.status_text = "Status: PARKING --> turning"

        # Reverse with steering to the right (omega > 0)
        reverse_turn = Twist2DStamped(v=-0.3, omega=4.5)
        self.pub_lane_twist.publish(reverse_turn)

        # After 2 seconds, continue straight backward
        rospy.Timer(rospy.Duration(1), self._straight_back, oneshot=True)


    def _straight_back(self, event):
        '''
        Driving straight back.

        Args:
            self

        Returns:
            none
        '''
        # Drive straight backward
        reverse_straight = Twist2DStamped(v=-0.1, omega=0.0)
        self.pub_lane_twist.publish(reverse_straight)

        # After 1 second, stop
        rospy.Timer(rospy.Duration(0.5), self._end_parking, oneshot=True)


    def _end_parking(self, event):
        '''
        End parking.
         
        Args:
            self

        Returns:
            none
        '''
        self.status_text = "Status: WAIT --> parked"
        self.time_inPark = time.time()
        if self.conf['debugPrints_parking']:
            rospy.loginfo("[PARKING] Done parking, switching to WAIT")

        # Stop the bot
        stop_msg = Twist2DStamped(v=0.0, omega=0.0)
        self.pub_lane_twist.publish(stop_msg)

        with self.transition_lock:
            self.state = "WAIT"
            # Wait for 3 seconds before exiting
            rospy.Timer(rospy.Duration(5), self._do_exit, oneshot=True)


    def _do_exit(self, event):
        '''
        Exit slot.
         
        Args:
            self

        Returns:
            none
        '''
        self.status_text = "Status: EXIT --> driving out"
        if self.conf['debugPrints_parking']:
            rospy.loginfo("[PARKING] Exiting parking")

        # Move slightly forward to get out of the parking slot
        forward_msg = Twist2DStamped(v=0.3, omega=-4.5)
        self.pub_lane_twist.publish(forward_msg)
        rospy.sleep(1.0)

        # Stop again after moving forward
        stop_msg = Twist2DStamped(v=0.0, omega=0.0)
        self.pub_lane_twist.publish(stop_msg)

        with self.transition_lock:
            self.state = "IDLE"
            self.status_text = "Status: IDLE"
            self.publish_state(0)


    
    ##### ========== MAIN RUN FUNCTION ========== #####
    def run(self):
        '''
        Main run function. Running the whole time.
         
        Args:
            self

        Returns:
            none
        '''
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            timeDelta_toLastBB = time.time() - self.time_lastBB
            if self.curr_bbox == None:
                rate.sleep()
                continue
            
            if timeDelta_toLastBB > 2:
                x1 = x2 = y1 = y2 = 0
            else:
                x1, y1, x2, y2 = self.curr_bbox.data

            if self.conf["debugPrints_parking"]:
                rospy.loginfo(f"[PARKING] {self.state}")

            with self.transition_lock:
                if self.time_startPark is not None:
                    self.delay_startPark = time.time()-self.time_startPark

                if self.state == "IDLE":
                    # Slot is far right --> start slow driving
                    # if self.conf["debugPrints_parking"]:
                    #     rospy.loginfo(f"[PARKING] x1: {x1}; y2: {y2}")
                    if x1 > self.conf["xForSlowDrive"] and y2 > self.conf["yForSlowDrive"]:
                        self.state = "SLOW"
                        self.status_text = "Status: SLOW --> approaching slot"
                        self.time_startPark = time.time()
                        self.publish_state(1)

                elif self.state == "SLOW" and self.delay_startPark>10:
                    # Almost passed the slot --> start parking
                    self.state = "PARKING"
                    self.publish_state(5)
                    if self.conf['debugPrints_parking']:
                        rospy.loginfo("[PARKING] Start parking")
                    self._start_parking()

                elif self.state in ["WAIT", "EXIT"]:
                    pass  # handled by timers

                # self.status_text = "Status: IDLE"
        


if __name__ == '__main__':
    node = ParkingNode(node_name='parking_node')
    node.run()