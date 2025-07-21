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
        self.linedImage = None
        self.parkstatusImage = None

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
        self.time_lastFree = time.time()
        self.time_lastOccupied = time.time()
        self.time_startPark = None
        self.time_inPark = None

        # Initializing for PID-Control
        self.prev_lateral_error = 0.0
        self.integral_lateral_error = 0.0
        self.prev_angle_error = 0.0
        self.integral_angle_error = 0.0
        self.last_time = time.time()

        # set slot initialy to occupied
        self.slot = 'occupied'

        # Read config file
        with open('packages/followlane/config/detect_duckieBotSlot.yaml', 'r') as f:
            self.conf = yaml.safe_load(f)
        with open('packages/followlane/config/detect_lane.yaml', 'r') as f:
            self.conf_lane = yaml.safe_load(f)

        # === Subscribers ===
        # for BBox of nearest free slot
        self.sub_freeSlot = rospy.Subscriber(f"/{self._vehicle_name}/detect/object/parkingBB", Float64MultiArray, self.cbFreeSlot, queue_size=1)
        self.sub_occupiedSlot = rospy.Subscriber(f"/{self._vehicle_name}/detect/object/parkingOccupiedBB", Float64MultiArray, self.cbOccupiedSlot, queue_size=1)
        # for annotated Image
        self.sub_YoloImage = rospy.Subscriber(f"/{self._vehicle_name}/detect/object/image", Image, self.cbYoloImage, queue_size=1)
        # for camera images
        self.sub_camImage = rospy.Subscriber(f"/{self._vehicle_name}/camera_node/image/compressed", CompressedImage, self.cbCamImage, queue_size=1)

        # === Publisher ===
        # for state (to switch_control_node)
        self.pub_state = rospy.Publisher(f"/{self._vehicle_name}/detect/object/slow4park", Int32, queue_size=1)# Start in IDLE
        # for driving command
        self.pub_lane_twist = rospy.Publisher(f"/{self._vehicle_name}/car_cmd_switch_node/cmd", Twist2DStamped, queue_size=1)

        # publish state 0 (idle)



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
        
        self.parkstatusImage = self.cv_image

        # Optionally: republish image with overlay if needed
        # img_out_msg = self._bridge_yoloImage.cv2_to_imgmsg(self.cv_image, encoding="bgr8")
        # self.pub_overlay.publish(img_out_msg)


    def cbFreeSlot(self, msg):
        '''
        Callback function for subscrier of bounding boxes for free parking Slots.

        Args:
            self
            msg: contains x1, y1 for upper left corner and x2,y2 for lower right corner of last detected free parking Slot.

        Returns:
            none
        '''
        if not msg.data or len(msg.data) != 4:
            # if self.conf["debugPrints_parking"] and self.conf["go_parking"]:
            #     rospy.loginfo(f"[PARKING] Wrong data free")
            self.curr_bbox = None
            return
        else:
            self.curr_bbox = msg
            self.slot = 'free'
            self.time_lastFree = time.time()
            if self.conf["debugPrints_parking"]:
                rospy.loginfo(f"[PARKING] free Slot detected")
            
            
    def cbOccupiedSlot(self, msg):
        '''
        Callback function for subscrier of bounding boxes for free parking Slots.

        Args:
            self
            msg: contains x1, y1 for upper left corner and x2,y2 for lower right corner of last detected free parking Slot.

        Returns:
            none
        '''
        if not msg.data or len(msg.data) != 4:
            # if self.conf["debugPrints_parking"] and self.conf["go_parking"]:
            #     rospy.loginfo(f"[PARKING] Wrong data occupied")
            self.curr_bbox = None
            return
        else:
            self.curr_bbox = None
            self.slot = 'occupied'
            self.time_lastOccupied = time.time()
            if self.conf["debugPrints_parking"]:
                rospy.loginfo(f"[PARKING] occupied Slot detected")


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
        m_yellow = b_yellow = None

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
            m_yellow, b_yellow = np.polyfit(xs, ys, 1)

            # create two points on straight (in image)
            x1, x2 = 0, image.shape[1]
            y1 = int(m_yellow * x1 + b_yellow)
            y2 = int(m_yellow * x2 + b_yellow)

            # draw straghit into picture
            cv2.line(image, (x1, y1), (x2, y2), (100, 255, 255), 2)

        # Create Target line for yellow
        self.m_target = 0
        self.b_target = 270
        x1, x2 = 0, image.shape[1]
        y1 = int(self.m_target * x1 + self.b_target)
        y2 = int(self.m_target * x2 + self.b_target)
        cv2.line(image, (x1, y1), (x2, y2), (180, 105, 255), 2)

        self.linedImage = image

        return m_yellow, b_yellow


    def calculate_control(self, m_yellow, b_yellow):
        '''
        Calculate speed (v) and angular velocity (omega) to control the parking maneuver.

        Args:
            self
            m_yellow: gradient of yellow line
            b_yellow: y-intercept of yellow line
        
        Returns:
            v: speed (negative for reverse)
            omega: angular velocity
        '''
        if m_yellow is None or b_yellow is None:
            if self.conf["debugPrints_parking"]:
                rospy.loginfo(f"[PARKING] m_yellow={m_yellow}; b_yellow={b_yellow}")
            return None, None  # Yellow line not detected

        # Get image width and define two x-positions: left and right
        width = self.camImage.shape[1]
        x_left = int(width * 0)
        x_right = int(width * 1)

        # Calculate corresponding y-values for the yellow and target line at those x positions
        y_yellow_left = m_yellow * x_left + b_yellow
        y_target_left = self.m_target * x_left + self.b_target

        y_yellow_right = m_yellow * x_right + b_yellow
        y_target_right = self.m_target * x_right + self.b_target

        # Compute lateral errors at left and right
        lateral_error_left = y_target_left - y_yellow_left
        lateral_error_right = y_target_right - y_yellow_right

        # Average lateral error as main input for steering correction
        if lateral_error_left > 0 and lateral_error_right > 0:
            lateral_error = 0
            v = omega = 0
            self.state = "WAIT"
            self.status_text = "Status: WAIT --> parked"
            self.time_inPark = time.time()
        else:
            lateral_error = (lateral_error_left + lateral_error_right) / 2

        # Estimate angular deviation from difference between left and right error
        # (larger difference implies line is rotated relative to target line)
        angle_error = lateral_error_right - lateral_error_left

        # Time step
        current_time = time.time()
        dt = current_time - self.last_time if self.last_time else 0.1
        self.last_time = current_time

        # PID gains for lateral control
        Kp_lat = 0.007
        Ki_lat = 0.0005
        Kd_lat = 0.002

        # PID gains for angle control
        Kp_ang = 0.04
        Ki_ang = 0.0001
        Kd_ang = 0.01

        # PID for lateral error
        self.integral_lateral_error += lateral_error * dt
        self.integral_lateral_error = max(min(self.integral_lateral_error, 100), -100)
        derivative_lateral_error = (lateral_error - self.prev_lateral_error) / dt if dt > 0 else 0
        self.prev_lateral_error = lateral_error

        omega_lat = (Kp_lat * lateral_error +
                     Ki_lat * self.integral_lateral_error +
                     Kd_lat * derivative_lateral_error)

        # PID for angle error
        self.integral_angle_error += angle_error * dt
        self.integral_angle_error = max(min(self.integral_angle_error, 100), -100)
        derivative_angle_error = (angle_error - self.prev_angle_error) / dt if dt > 0 else 0
        self.prev_angle_error = angle_error

        omega_ang = (Kp_ang * angle_error +
                     Ki_ang * self.integral_angle_error +
                     Kd_ang * derivative_angle_error)

        omega = omega_lat + omega_ang

        if abs(omega) < 0.1:
            omega = 0.0

        # Dynamically adjust speed (v) based on total error
        total_error = abs(lateral_error) #+ abs(angle_error)
        min_v = 0.15
        max_v = 0.2
        v = -min(max_v, 0.02 * total_error)

        if abs(omega) > 0.1 and abs(v) < abs(min_v):
            v = -max(abs(v), abs(min_v))

        # # Calculate angular velocity (omega) and set constant backward velocity (v)
        # omega = Kp_lateral * lateral_error + Kp_angle * angle_error
        # v = -0.25  # Constant backward speed

        # Debug output
        if self.conf["debugPrints_parking"]:
            rospy.loginfo(f"[PARKING] lat_err L/R/ges: {lateral_error_left:.2f}/{lateral_error_right:.2f}/{lateral_error:.2f}, "
                        f"angle_err: {angle_error:.2f}, omega: {omega:.2f}, v: {v:.2f}")

        # Stop condition if errors are small enough
        if abs(lateral_error) < 5 and abs(angle_error) < 5:
            v = omega = 0
            self.state = "WAIT"
            self.status_text = "Status: WAIT --> parked"
            self.time_inPark = time.time()

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
            if self.cv_image is None or self.camImage is None or not self.conf["go_parking"]:
                if self.conf["debugPrints_parking"] and not self.conf["go_parking"]:
                    rospy.loginfo(f"[PARKING] deactivated")
                rate.sleep()
                continue
            elif self.curr_bbox is None and self.state == "IDLE":
                rate.sleep()
                continue
            elif self.slot=='occupied' or time.time()-self.time_lastOccupied<5:
                if self.conf["debugPrints_parking"]:
                    rospy.loginfo(f"[PARKING] time since last occupied {time.time()-self.time_lastOccupied:.2f}s")
                rate.sleep()
                continue
            elif self.curr_bbox is not None:
                timeDelta_toLastFree = time.time() - self.time_lastFree
                if timeDelta_toLastFree > 2:
                    x1 = x2 = y1 = y2 = 0
                else:
                    x1, y1, x2, y2 = self.curr_bbox.data

            if self.time_startPark is not None:
                self.delay_startPark = time.time()-self.time_startPark
            if self.time_inPark is not None:
                self.delay_inPark = time.time()-self.time_inPark

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
                    elif False:
                        self.state = "SLOW"
                        self.status_text = "Status: SLOW --> approaching slot"
                        self.time_startPark = time.time()
                        self.publish_state(1)

                elif self.state == "SLOW" and self.delay_startPark>2:
                    # Almost passed the slot --> start parking
                    self.state = "PARKING"
                    self.status_text == "Status: PARKING --> is parking (reverse)"
                    self.time_doPark = time.time()
                    self.publish_state(5)

                elif self.state == "PARKING":
                    if self.camImage is not None:
                        m_yellow, b_yellow = self.detect_yellow(self.camImage)
                        if self.conf["debugPrints_parking"]:
                            rospy.loginfo(f"[PARKING] m_yellow: {m_yellow}; b_yellow: {b_yellow}")
                        v, omega = self.calculate_control(m_yellow, b_yellow)
                        #v=omega=0
                        if v is None and omega is None:
                            v = omega = 0
                        reverse_turn = Twist2DStamped(v=v, omega=omega)
                        self.pub_lane_twist.publish(reverse_turn)

                elif self.state == "WAIT" and self.delay_inPark<=5:
                    stopBot = Twist2DStamped(v=0, omega=0)
                    self.pub_lane_twist.publish(stopBot)

                elif self.state == "WAIT":
                    self.state = "EXIT"
                    self.status_text = "Status: EXIT"
                    exit_turn = Twist2DStamped(v=0.2, omega=-3.5)
                    self.pub_lane_twist.publish(exit_turn)
                    rospy.sleep(1.0)
                    stopBot = Twist2DStamped(v=0, omega=0)
                    self.pub_lane_twist.publish(stopBot)
                    self.state = "IDLE"
                    self.status_text = "Status: IDLE"

            if self.conf['show_lineDetectImage']:
                #cv2.imshow("edges-white", edges_white)
                #cv2.imshow("edges-yellow", edges_yellow)
                if self.linedImage is not None:
                    cv2.imshow("Parking", self.linedImage)
                if self.parkstatusImage is not None:
                    cv2.imshow("Parking Status", self.parkstatusImage)

                cv2.waitKey(1)
        


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