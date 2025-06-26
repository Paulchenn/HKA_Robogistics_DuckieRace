# import cv2
# import numpy as np
# import rospy
# import os
# import json
# from std_msgs.msg import String
# from duckietown.dtros import DTROS, NodeType
# from sensor_msgs.msg import CompressedImage  # Import für das Kamerabild
# from duckietown_msgs.msg import LEDPattern, LEDPatternArray



# class RedLineListener(DTROS):

#     def __init__(self, node_name):
#         super(RedLineListener, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)

#         self._vehicle_name = os.environ['VEHICLE_NAME']
#         self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"

#         self.direction_redline_topic = f"/{self._vehicle_name}/red_line_info"
#         self.direction_redline = rospy.Subscriber(self.direction_redline_topic, String, self.callback, queue_size=10)
#         self.sub_image = rospy.Subscriber(self._camera_topic, CompressedImage, self.process_image, queue_size=1)

#         self.left_point = None
#         self.right_point = None

#         rospy.loginfo("Red Line Listener Node Initialized")

#     def callback(self, msg):
#         data = json.loads(msg.data)
#         direction = data["richtung"]
#         left = data["left"]
#         right = data["right"]
#         self.left_point = (int(left[0]), int(left[1]))
#         self.right_point = (int(right[0]), int(right[1]))
#         self.publish_turn_direction(direction)

#         rospy.loginfo_throttle(1, f"Empfangen: {direction} linker Punkt {self.left_point} rechter Punkt {self.right_point}")

#     def process_image(self, msg):
#         np_arr = np.frombuffer(msg.data, np.uint8)
#         frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

#         if self.left_point is not None and self.right_point is not None:
#             self.ellipseForTurning(frame, self.left_point, self.right_point)

#         cv2.imshow("Abbiegelinien", frame)
#         cv2.waitKey(1)  # nicht 0, sonst blockiert es den Node


#     def ellipseForTurning(self, frame, left, right):
#         # Halbachsen als Tupel (x-Achse, y-Achse)
#         left_axes = (150, 300)
#         right_axes = (350, 80)

#         color = (255, 0, 0)
#         thickness = 2

#         # Ellipse links (z.B. 0° bis 90°)
#         cv2.ellipse(frame, center=left, axes=left_axes, angle=90, startAngle=200, endAngle=270, color=color, thickness=thickness)

#         # Ellipse rechts (z.B. 90° bis 180°)
#         cv2.ellipse(frame, center=right, axes=right_axes, angle=0, startAngle=180, endAngle=250, color=color, thickness=thickness)

#         height, width = frame.shape[:2]
#         third_height, half_width = height // 3, width // 2
        
#         # Vertikale Linie für geradeaus fahren
#         cv2.line(frame, (half_width, 400), (half_width, third_height), (0, 255, 0), 2)  # grün, 2 Pixel dick

#     def publish_turn_direction(direction):
#         if direction == "left":
#             #Publishe den Punkt bei x=174, y=230
#             print()
#             set_blinker({self._vehicle_name}, "left", farbe = "gelb", frequenz = 2.0, duty = 0.5)
#         elif direction == "geradeaus":
#             #Publishe einen Punkt geradeaus
#             print()
#         elif direction == "right":
#             #Publishe den rechten Punkt bei x= , y=((((((((((((((((((((((((((((((((===============================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================e=))))))))))))))))))))))))))))))))
#             print()
 
#     def set_blinker(bot_name: str, seite: str, farbe: str = "gelb", frequenz: float = 2.0, duty: float = 0.5):
#         """
#         Aktiviert einen Blinker auf dem Duckiebot.
#         Parameters:
#         - bot_name: Name des Duckiebots (z. B. "duckiebot1")
#         - seite: "left", "right", "front", "back"
#         - farbe: "gelb", "rot", "blau", "grün", "weiß"
#         - frequenz: Blinkfrequenz in Hz
#         - duty: Verhältnis von An-Zeit zu Gesamtzeit (0–1)
#         """
#         farben = {
#             "gelb": [1.0, 1.0, 0.0],
#             "rot": [1.0, 0.0, 0.0],
#             "grün": [0.0, 1.0, 0.0],
#             "blau": [0.0, 0.0, 1.0],
#             "weiß": [1.0, 1.0, 1.0]
#         }
    
#         if farbe not in farben or seite not in ["left", "right", "front", "back"]:
#             rospy.logwarn("Ungültige Farbe oder Seite für Blinker.")
#             return
    
#         pub = rospy.Publisher(f"/{bot_name}/led_emitter_node/led_pattern", LEDPatternArray, queue_size=1)
#         rospy.sleep(0.5)  # Warten, bis Publisher bereit ist
    
#         pattern = LEDPattern()
#         pattern.color = farben[farbe]
#         pattern.frequency = frequenz
#         pattern.duty_cycle = duty
#         pattern.led = seite
    
#         msg = LEDPatternArray()
#         msg.patterns = [pattern]
    
#         pub.publish(msg)
#         rospy.loginfo(f"Blinker an {seite} aktiviert ({farbe})")


# if __name__ == "__main__":
#     #rospy.init_node("red_line_listener")
#     listener = RedLineListener(node_name='turning_process_node')
#     rospy.spin()


import cv2
import numpy as np
import rospy
import os
import json
from std_msgs.msg import String, ColorRGBA
from geometry_msgs.msg import Point, Twist
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage, Image
from duckietown_msgs.msg import LEDPattern #, LEDPatternArray

#für das Anfahren eines Punktes
from cv_bridge import CvBridge
 
class RedLineListener(DTROS):
 
    def __init__(self, node_name):
        super(RedLineListener, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)
 
        self._vehicle_name = os.environ['VEHICLE_NAME']
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
 
        self.direction_redline_topic = f"/{self._vehicle_name}/red_line_info"
        self.direction_redline = rospy.Subscriber(self.direction_redline_topic, String, self.callback, queue_size=10)
        self.sub_image = rospy.Subscriber(self._camera_topic, CompressedImage, self.process_image, queue_size=1)
 
        self.cmd_pub = rospy.Publisher(f"/{self._vehicle_name}/zielpunkt", Point, queue_size=1)
        self.left_point = None
        self.right_point = None
 
        self.led_pub = rospy.Publisher(f"/{self._vehicle_name}/led_emitter_node/led_pattern", LEDPattern, queue_size=1)
        # Timer für Blink-Callback, alle 0.5 Sekunden (2 Hz Blinkfrequenz)
        #self.blink_timer = rospy.Timer(rospy.Duration(0.5), self.blink_all_leds)
 
        self.blink_on = True
        #self.blink_timer = rospy.Timer(rospy.Duration(1), self.blink_all_leds)  # alle 0.5 Sekunden blinken

        rospy.loginfo("Red Line Listener Node Initialized")
 
    def callback(self, msg):
        data = json.loads(msg.data)
        direction = data["richtung"]
        left = data["left"]
        right = data["right"]
        self.left_point = (int(left[0]), int(left[1]))
        self.right_point = (int(right[0]), int(right[1]))
        self.publish_turn_direction(direction)
        rospy.loginfo_throttle(1, f"Empfangen: {direction} | links: {self.left_point} | rechts: {self.right_point}")
 
    def process_image(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
 
        if self.left_point and self.right_point:
            self.ellipseForTurning(frame, self.left_point, self.right_point)
 
        cv2.imshow("Abbiegelinien", frame)
        cv2.waitKey(1)
        self.blink_all_leds()
 
    def ellipseForTurning(self, frame, left, right):
        left_axes = (150, 300)
        right_axes = (350, 80)
        color = (255, 0, 0)
        thickness = 2
 
        cv2.ellipse(frame, center=left, axes=left_axes, angle=90, startAngle=200, endAngle=270, color=color, thickness=thickness)
        cv2.ellipse(frame, center=right, axes=right_axes, angle=0, startAngle=180, endAngle=250, color=color, thickness=thickness)
 
        height, width = frame.shape[:2]
        third_height, half_width = height // 3, width // 2
        cv2.line(frame, (half_width, 400), (half_width, third_height), (0, 255, 0), 2)
 
    def publish_turn_direction(self, direction):
        zielpunkt = Point()
        pattern_on = LEDPattern()
        #self.blink_all_leds()
 
        pattern_on.frequency = 2.0
        if direction == "links":
            pattern_on.color_mask = [True, False, False, True]
            pattern_on.frequency_mask = [True, False, False, True]
            pattern_on.rgb_vals = [
                ColorRGBA(1.0, 1.0, 0.0, 1.0),  # Front Left - gelb
                ColorRGBA(0, 0, 0, 1),          # Front Right - aus
                ColorRGBA(0, 0, 0, 1),          # Back Right - aus
                ColorRGBA(1.0, 1.0, 0.0, 1.0)   # Back Left - gelb
            ]
            # zielpunkt.x = 174.0
            # zielpunkt.y = 230.0
            # zielpunkt.z = 0.0
           # self.blink_all_leds()
        elif direction == "rechts":
            pattern_on.color_mask = [False, True, True, False]
            pattern_on.frequency_mask = [False, True, True, False]
            pattern_on.rgb_vals = [
                ColorRGBA(0, 0, 0, 1),  # Front Left - aus
                ColorRGBA(1.0, 1.0, 0.0, 1.0),  # Front Right - gelb
                ColorRGBA(1.0, 1.0, 0.0, 1.0),  # Back Right - gelb
                ColorRGBA(0, 0, 0, 1)   # Back Left - aus
            ]
            # zielpunkt.x = 460.0
            # zielpunkt.y = 230.0
            # zielpunkt.z = 0.0
            #self.blink_all_leds()
        elif direction == "geradeaus":
            # zielpunkt.x = 320.0
            # zielpunkt.y = 180.0
            zielpunkt.z = 0.0
 
        self.led_pub.publish(pattern_on)
        self.cmd_pub.publish(zielpunkt)
        rospy.loginfo(f"Zielpunkt publiziert: {zielpunkt}")
 

    def drive_to_pixel(self):


        # Testfunktion: Fahre gezielt zu einem festen Punkt im Bild.
        # Diese Methode abonniert den Kamera-Stream und steuert den Bot
        # proportional zum Abstand des Zielpixels.
        ziel_x = 174
        ziel_y = 230
        toleranz_x = 20
        toleranz_y = 20
        fertig = False
        bridge = CvBridge()
        pub_cmd = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
    
        def image_callback(data):
            nonlocal fertig
            if fertig:
                return
    
            frame = bridge.imgmsg_to_cv2(data, 'bgr8')
            h, w, _ = frame.shape
            dx = ziel_x - (w // 2)
            dy = ziel_y - (h - 10)

            if abs(dx) < toleranz_x and abs(dy) < toleranz_y:
                twist = Twist()
                pub_cmd.publish(twist)
                fertig = True
                rospy.loginfo(" Ziel erreicht, Bot gestoppt.")
                return
            twist = Twist()
            twist.linear.x = 0.2
            twist.angular.z = -0.003 * dx
            pub_cmd.publish(twist)
    
        sub_img = rospy.Subscriber(f"/{self._vehicle_name}/camera_node/image/raw", Image, image_callback)

        rospy.loginfo("Starte Zielansteuerung zu (174, 230) im Bild ...")


    def blink_all_leds(self):
        pattern_on = LEDPattern()    

        if self.blink_on:
        # Index:     0       1       2       3
        # LEDs:   FRONT_LEFT  FRONT_RIGHT  BACK_RIGHT  BACK_LEFT
            pattern_on.rgb_vals = [
                ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0),
                ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0),
                ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0),
                ColorRGBA(r=0.0, g=1.0, b=1.0, a=1.0)
                ]
            
            # color_mask sagt, welche LEDs an sind (alle 4 LEDs an)
            pattern_on.color_mask = [True, True, True, True]
            
            # frequency gibt die Blinkfrequenz an
            pattern_on.frequency = 0  # 2 Hz blink
            
            # frequency_mask sagt, für welche LEDs die Frequenz gilt (alle 4)
            pattern_on.frequency_mask = [True, True, True, True]
        else :
            pattern_on.rgb_vals = [
                ColorRGBA(r=0.0, g=0.0, b=0.0, a=1.0),
                ColorRGBA(r=0.0, g=0.0, b=0.0, a=1.0),
                ColorRGBA(r=0.0, g=0.0, b=0.0, a=1.0),
                ColorRGBA(r=0.0, g=0.0, b=0.0, a=1.0)
                ]
            
            # color_mask sagt, welche LEDs an sind (alle 4 LEDs an)
            pattern_on.color_mask = [False, False, False, False]
            
            # frequency gibt die Blinkfrequenz an
            pattern_on.frequency = 0  # 2 Hz blink
            
            # frequency_mask sagt, für welche LEDs die Frequenz gilt (alle 4)
            pattern_on.frequency_mask = [False, False, False, False]            
            
        rospy.loginfo_throttle(3,f"Jetzt wird gepublisht! {pattern_on}")
        #self.led_pub.publish(pattern_on)
        self.blink_on = not self.blink_on
        # Optional könntest du hier noch auskommentierte Schleifen oder ein Aus-Pattern ergänzen.
    
 
if __name__ == "__main__":
    listener = RedLineListener(node_name='turning_process_node')
    rospy.spin()