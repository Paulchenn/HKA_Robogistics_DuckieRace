import cv2
import os
import rospy
import threading
import yaml
import time
import numpy as np
import functools
import math

from cv_bridge import CvBridge
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import Twist2DStamped
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Int32, Float64MultiArray


class ParkingNode(DTROS):
    def __init__(self, node_name):
        super(ParkingNode, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)
        self._vehicle_name = os.environ['VEHICLE_NAME']

        self.camImage = None
        self.cv_image = None
        self.curr_bbox = None

        # State machine
        self.state = "IDLE"
        self.last_slot = None
        self.transition_lock = threading.Lock()

        # Start in IDLE
        self.status_text = "Status: IDLE"

        # start cv bridge
        self._bridge_yoloImage = CvBridge()
        self._bridge_camImage = CvBridge()

        # timers
        self.time_lastBB = time.time()
        self.time_startPark = None
        self.time_inPark = None

        # Read config file
        with open('packages/followlane/config/detect_duckieBotSlot.yaml', 'r') as f:
            self.conf = yaml.safe_load(f)
        with open('packages/followlane/config/detect_lane.yaml', 'r') as f:
            self.conf_lane = yaml.safe_load(f)

        # === Subscribers ===
        # for BBox of nearest free slot
        self.sub_slot = rospy.Subscriber(f"/{self._vehicle_name}/detect/object/parkingBB", Float64MultiArray, self.cbSlot, queue_size=1)
        # for annotated Image
        self.sub_YoloImage = rospy.Subscriber(f"/{self._vehicle_name}/detect/object/image", Image, self.cbYoloImage, queue_size=1)
        # for camera images
        self.sub_camImage = rospy.Subscriber(f"/{self._vehicle_name}/camera_node/image/compressed", CompressedImage, self.cbCamImage, queue_size=1)

        # === Publisher ===
        # for state (to switch_control_node)
        self.pub_state = rospy.Publisher(f"/{self._vehicle_name}/detect/object/slow4park", Int32, queue_size=1)# Start in IDLE
        self.publish_state(0)
        # for driving command
        self.pub_lane_twist = rospy.Publisher(f"/{self._vehicle_name}/car_cmd_switch_node/cmd", Twist2DStamped, queue_size=1)


    ##### ===== CALLBACK FUNCTIONS OF SUBSRIBERS ===== #####
    def cbYoloImage(self, msg):
        '''
        Callback function for subscrier of annotated image (after YOLO).

        Args:
            self
            msg: annotaded image (with bounding boxes of detected stuff.

        Returns:
            none
        '''
        try:
            self.cv_image = self._bridge_yoloImage.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logerr(f"Could not convert image: {e}")
            return

        # Draw the status text on the image
        if self.status_text == "Status: SLOW --> approaching slot":
            if self.time_startPark is not None:
                self.delay_startPark = time.time()-self.time_startPark
                myText = f"{self.status_text}, {self.delay_startPark:.2f}s"
        elif self.status_text == "Status: PARKING --> is parking (reverse)":
            if self.time_doPark is not None:
                self.delay_doPark = time.time()-self.time_doPark
                myText = f"{self.status_text}, {self.delay_doPark:.2f}s"
        elif self.status_text == "Status: WAIT --> parked":
            if self.time_inPark is not None:
                self.delay_inPark = time.time()-self.time_inPark
                myText = f"{self.status_text}, {self.delay_inPark:.2f}s"
        else:
            myText = self.status_text

        cv2.putText(self.cv_image, myText, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Optional: show image (for debugging only, not on robot)
        if self.conf['show_image'] and self.cv_image is not None:
            cv2.imshow("Parking Status", self.cv_image)
            cv2.waitKey(1)

        # Optionally: republish image with overlay if needed
        # img_out_msg = self._bridge_yoloImage.cv2_to_imgmsg(self.cv_image, encoding="bgr8")
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


    def cbCamImage(self, msg):
        '''
        Callback function for subscrier of camera Image.

        Args:
            self
            msg: Compressed camera image.

        Returns:
            none
        '''
        self.camImage = self._bridge_camImage.compressed_imgmsg_to_cv2(msg)



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


    def create_polygon(self):
        return np.array([[
            [self.conf_lane['parking_image']['top_left_x'], self.conf_lane['parking_image']['top_left_y']],
            [self.conf_lane['parking_image']['top_right_x'], self.conf_lane['parking_image']['top_right_y']],
            [self.conf_lane['parking_image']['bottom_right_x'], self.conf_lane['parking_image']['bottom_right_y']],
            [self.conf_lane['parking_image']['bottom_left_x'], self.conf_lane['parking_image']['bottom_left_y']],
        ]], dtype=np.int32)
    

    def detect_yellow(self, image):
        min_area = 1

        # convert Image to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # get config value for white and yellow
        wh = self.conf_lane['white']
        gh = self.conf_lane['gelb']

        # create kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

        # create mask for white detection
        mask_white = cv2.inRange(hsv, (wh['hl'], wh['sl'], wh['vl']), (wh['hh'], wh['sh'], wh['vh']))
        mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, kernel)
        mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, kernel)

        # create mask for yellow detection
        mask_yellow = cv2.inRange(hsv, (gh['hl'], gh['sl'], gh['vl']), (gh['hh'], gh['sh'], gh['vh']))
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel)
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, kernel)

        # mask with polygon
        polygon = self.create_polygon()
        mask_poly = np.zeros_like(mask_white)
        cv2.fillPoly(mask_poly, polygon, 255)
        mw = cv2.bitwise_and(mask_white, mask_poly)
        my = cv2.bitwise_and(mask_yellow, mask_poly)

        edges_white = cv2.Canny(cv2.GaussianBlur(mw, (5, 5), 0), 50, 150)
        edges_yellow = cv2.Canny(cv2.GaussianBlur(my, (5, 5), 0), 50, 150)

        contours_white, _ = cv2.findContours(edges_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_yellow, _ = cv2.findContours(edges_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        centroids_white = []
        centroids_yellow = []

        for cnt in contours_white:
            area = cv2.contourArea(cnt)
            if area <= min_area:
                continue
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cx = M['m10'] / M['m00']
                cy = M['m01'] / M['m00']
                centroids_white.append((cx, cy))
                cv2.drawContours(image, [cnt], -1, (0, 255, 0), 2)

        for cnt in contours_yellow:
            area = cv2.contourArea(cnt)
            if area <= min_area:
                continue
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cx = M['m10'] / M['m00']
                cy = M['m01'] / M['m00']
                centroids_yellow.append((cx, cy))
                cv2.drawContours(image, [cnt], -1, (0, 255, 255), 2)

        # interpolate straight through white centroids (if enough points)
        if len(centroids_white) >= 2:
            xs = np.array([pt[0] for pt in centroids_white])
            ys = np.array([pt[1] for pt in centroids_white])

            # fit y = m * x + b
            m_white, b_white = np.polyfit(xs, ys, 1)

            # create two points on straight (in image)
            x1, x2 = 0, image.shape[1]
            y1 = int(m_white * x1 + b_white)
            y2 = int(m_white * x2 + b_white)

            # draw straghit into picture
            # cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # interpolate straight through yellow centroids (if enough points)
        if len(centroids_yellow) >= 2:

            xs = np.array([pt[0] for pt in centroids_yellow])
            ys = np.array([pt[1] for pt in centroids_yellow])

            # fit y = m * x + b
            self.m_yellow, self.b_yellow = np.polyfit(xs, ys, 1)

            # create two points on straight (in image)
            x1, x2 = 0, image.shape[1]
            y1 = int(self.m_yellow * x1 + self.b_yellow)
            y2 = int(self.m_yellow * x2 + self.b_yellow)

            # draw straghit into picture
            cv2.line(image, (x1, y1), (x2, y2), (100, 255, 255), 2)

            if self.conf["debugPrints_parking"]:
                rospy.loginfo(f"[PARKING] m_yellow: {self.m_yellow}; b_yellow: {self.b_yellow}")

        # Create Target line for yellow
        self.m_target = 0
        self.b_target = 290
        x1, x2 = 0, image.shape[1]
        y1 = int(self.m_target * x1 + self.b_target)
        y2 = int(self.m_target * x2 + self.b_target)
        cv2.line(image, (x1, y1), (x2, y2), (180, 105, 255), 2)


        if self.conf['show_lineDetect_image']:
            #cv2.imshow("edges-white", edges_white)
            #cv2.imshow("edges-yellow", edges_yellow)
            cv2.imshow("Parking", image)
            cv2.waitKey(1)


    def calculate_control(self):
        if self.m_yellow is None or self.b_yellow is None:
            return None, None  # Linien nicht gefunden

        # Bildmitte entlang x
        x_middle = self.camImage.shape[1] // 2

        # y-Werte an der Bildmitte für gelbe Linie und Ziel-Linie
        y_yellow = self.m_yellow * x_middle + self.b_yellow
        y_target = self.m_target * x_middle + self.b_target

        # laterale Differenz (Bildpixel, kann als Proxy für Abstand genutzt werden)
        lateral_error = y_target - y_yellow

        # Winkelabweichung (in Grad oder direkt Steigung)
        angle_error = math.atan(self.m_yellow - self.m_target)  # rad

        # einfache P-Regler
        Kp_steering = 0.005
        Kp_angle = 1.0

        # Steuerung
        omega = Kp_steering * lateral_error + Kp_angle * angle_error
        v = -0.15  # konstante Rückwärtsfahrt

        if self.conf["debugPrints_parking"]:
            rospy.loginfo(f"[PARKING] lateral_error: {lateral_error:.2f}, angle_error: {math.degrees(angle_error):.2f}, omega: {omega:.2f}")
        
        if abs(lateral_error) < 5 and abs(angle_error) < math.radians(2):
            v = omega = 0
            self.stateDoParking = False
        else:
            self.stateDoParking = True

        return v, omega

    

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
            if self.cv_image is None or self.camImage is None: #or self.curr_bbox is None
                rate.sleep()
                continue
            
            timeDelta_toLastBB = time.time() - self.time_lastBB
            if True: #timeDelta_toLastBB > 2:
                x1 = x2 = y1 = y2 = 0
            else:
                x1, y1, x2, y2 = self.curr_bbox.data

            if self.time_startPark is not None:
                self.delay_startPark = time.time()-self.time_startPark

            with self.transition_lock:
                if self.state == "IDLE":
                    # Slot is far right --> start slow driving
                    # if self.conf["debugPrints_parking"]:
                    #     rospy.loginfo(f"[PARKING] x1: {x1}; y2: {y2}")
                    if x1 > self.conf["xForSlowDrive"] and y2 > self.conf["yForSlowDrive"]:
                        self.state = "SLOW"
                        self.status_text = "Status: SLOW --> approaching slot"
                        self.time_startPark = time.time()
                        self.publish_state(1)
                    elif True:
                        self.state = "SLOW"
                        self.status_text = "Status: SLOW --> approaching slot"
                        self.time_startPark = time.time()
                        self.publish_state(1)

                elif self.state == "SLOW" and self.delay_startPark>3:
                    # Almost passed the slot --> start parking
                    self.state = "PARKING"
                    self.status_text == "Status: PARKING --> is parking (reverse)"
                    self.time_doPark = time.time()
                    self.stateDoParking = True
                    self.publish_state(5)

                elif self.state == "PARKING" and self.stateDoParking == True:
                    if self.camImage is not None:
                        self.detect_yellow(self.camImage)
                        v, omega = self.calculate_control()
                        if v is not None and omega is not None:
                            reverse_turn = Twist2DStamped(v=v, omega=omega)
                            self.pub_lane_twist.publish(reverse_turn)

                elif self.state == "PARKING" and self.stateDoParking == False:
                    self.time_inPark = time.time()
                    # calculate Control based on yellow dottet line
                    self.status_text = "Status: WAIT --> parked"

                elif self.state in ["WAIT", "EXIT"]:
                    pass  # handled by timers

                # self.status_text = "Status: IDLE"
        


if __name__ == '__main__':
    node = ParkingNode(node_name='parking_node')
    node.run()





















##### ===== RESERVE ===== #####
# def _start_parking(self, duration):
#         '''
#         Start the parking process.
         
#         Args:
#             self
#             duration: Time to do drive command

#         Returns:
#             none
#         '''
#         self.status_text = "Status: PARKING --> turning"

#         # Reverse with steering to the right (omega > 0)
#         reverse_turn = Twist2DStamped(v=-0.4, omega=2)
#         self.pub_lane_twist.publish(reverse_turn)

#         # After 2 seconds, continue straight backward
#         rospy.Timer(rospy.Duration(duration), functools.partial(self._straight_back, 1), oneshot=True)


#     def _straight_back(self, duration):
#         '''
#         Driving straight back.

#         Args:
#             self
#             duartion: Time to do drive command

#         Returns:
#             none
#         '''
#         self.status_text = "Status: PARKING --> straight"
#         # Drive straight backward
#         reverse_straight = Twist2DStamped(v=-0.4, omega=0.0)
#         self.pub_lane_twist.publish(reverse_straight)

#         # After 1 second, stop
#         rospy.Timer(rospy.Duration(duration), functools.partial(self._end_parking, 5), oneshot=True)


#     def _end_parking(self, duration):
#         '''
#         End parking.
         
#         Args:
#             self
#             duration: Time to wait

#         Returns:
#             none
#         '''
#         self.status_text = "Status: WAIT --> parked"
#         self.time_inPark = time.time()
#         if self.conf['debugPrints_parking']:
#             rospy.loginfo("[PARKING] Done parking, switching to WAIT")

#         # Stop the bot
#         stop_msg = Twist2DStamped(v=0.0, omega=0.0)
#         self.pub_lane_twist.publish(stop_msg)

#         with self.transition_lock:
#             self.state = "WAIT"
#             # Wait for 3 seconds before exiting
#             rospy.Timer(rospy.Duration(duration), functools.partial(self._do_exit), oneshot=True)


#     def _do_exit(self):
#         '''
#         Exit slot.
         
#         Args:
#             self

#         Returns:
#             none
#         '''
#         self.status_text = "Status: EXIT --> driving out"
#         if self.conf['debugPrints_parking']:
#             rospy.loginfo("[PARKING] Exiting parking")

#         # Move slightly forward to get out of the parking slot
#         forward_msg = Twist2DStamped(v=0.2, omega=-4.5)
#         self.pub_lane_twist.publish(forward_msg)
#         rospy.Timer(rospy.Duration(1.0), functools.partial(self._stop_after_exit), oneshot=True)
#         rospy.sleep(1.0)


#     def _stop_after_exit(self):
#         stop_msg = Twist2DStamped(v=0.0, omega=0.0)
#         self.pub_lane_twist.publish(stop_msg)
#         with self.transition_lock:
#             self.state = "IDLE"
#             self.status_text = "Status: IDLE"
#             self.publish_state(0)