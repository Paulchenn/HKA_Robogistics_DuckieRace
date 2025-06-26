#!/usr/bin/env python3

import os
import rospy
import cv2
import yaml
import numpy as np
import time
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from std_msgs.msg import Float64


class CameraReaderNode(DTROS):
    def __init__(self, node_name):
        super(CameraReaderNode, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)

        self._vehicle_name = os.environ['VEHICLE_NAME']
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        self._bridge = CvBridge()
        self._window = "camera-reader"
        self._config_path = 'packages/followlane/config/detect_lane.yaml'

        with open(self._config_path, 'r') as f:
            self.conf = yaml.safe_load(f)

        self.debug = self.conf.get('show_debug', False)
        self.target_x_buffer = []
        self.image = None

        rospy.Subscriber(self._camera_topic, CompressedImage, self.image_callback)
        self.pub_lane = rospy.Publisher(f"/{self._vehicle_name}/detect/lane", Float64, queue_size=1)
        rospy.on_shutdown(self.fnShutDown)
        self.pub_left_x = rospy.Publisher(f"/{self._vehicle_name}/detect/lane/left_x", Float64, queue_size=1)
        self.pub_right_x = rospy.Publisher(f"/{self._vehicle_name}/detect/lane/right_x", Float64, queue_size=1)


    def image_callback(self, msg):
        self.image = self._bridge.compressed_imgmsg_to_cv2(msg)
        self._timestamp = msg.header.stamp

    def create_polygon(self):
        return np.array([[
            [self.conf['lane_image']['top_left_x'], self.conf['lane_image']['top_left_y']],
            [self.conf['lane_image']['top_right_x'], self.conf['lane_image']['top_right_y']],
            [self.conf['lane_image']['bottom_right_x'], self.conf['lane_image']['bottom_right_y']],
            [self.conf['lane_image']['bottom_left_x'], self.conf['lane_image']['bottom_left_y']],
        ]], dtype=np.int32)

    def compute_target_x_from_polygon(self, polygon, mask_white, mask_yellow, image):
        min_area = 100
        mask_poly = np.zeros_like(mask_white)
        cv2.fillPoly(mask_poly, polygon, 255)
        mw = cv2.bitwise_and(mask_white, mask_poly)
        my = cv2.bitwise_and(mask_yellow, mask_poly)
        # mr = cv2.bitwise_and(mask_red, mask_poly)

        # Farbige Darstellung im Bild (optional für Debug)
        image[mw > 0] = (0, 255, 0)     # Grün für weiße Maske
        image[my > 0] = (255, 0, 0)     # Blau für gelbe Maske
        # image[mr > 0] = (0, 0, 255)     # Rot für rote Maske

        # Linien erkennen mit Hough-Transformation für Weiß und Gelb
        def get_lines(mask):
            blurred = cv2.GaussianBlur(mask, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            return cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=15, minLineLength=10, maxLineGap=300)

        lines_white = get_lines(mw)
        lines_yellow = get_lines(my)

        # Visualisierung Weiß
        if lines_white is not None:
            for x1, y1, x2, y2 in lines_white[:, 0]:
                cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Visualisierung Gelb
        if lines_yellow is not None:
            for x1, y1, x2, y2 in lines_yellow[:, 0]:
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 255), 2)
        

        # Berechnung der Zielposition aus weißen und gelben Linien
        x_white = None
        x_yellow = None

        if lines_white is not None:
            white_xs = [(x1 + x2) / 2 for [[x1, _, x2, _]] in lines_white if ((x1 + x2) / 2) >= 310]
            if white_xs:
                x_white = min(white_xs)

        if lines_yellow is not None:
            yellow_xs = [(x1 + x2) / 2 for [[x1, _, x2, _]] in lines_yellow if ((x1 + x2) / 2) <= 340]
            if yellow_xs:
                x_yellow = max(yellow_xs)

        # Ziel-x berechnen basierend auf gefundenen Linien
        if x_white is not None and x_yellow is not None:
            return (x_white + x_yellow + 120) / 2
        elif x_yellow is not None:
            return x_yellow + 260
        elif x_white is not None:
            offset = 180 - 4.17 * idx ** 1.4
            return (x_white - offset)

        edges_white = cv2.Canny(cv2.GaussianBlur(mw, (5, 5), 0), 50, 150)
        edges_yellow = cv2.Canny(cv2.GaussianBlur(my, (5, 5), 0), 50, 150)

        if self.debug:
            cv2.imshow("edges-white", edges_white)
            cv2.imshow("edges-yellow", edges_yellow)

        contours_white, _ = cv2.findContours(edges_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_yellow, _ = cv2.findContours(edges_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        leftmost_x = None
        for i, cnt in enumerate(contours_white):
            area = cv2.contourArea(cnt)
            if area <= min_area:
                if self.debug:
                    rospy.loginfo(f"[Weiß] {i}: zu klein (Fläche = {area:.1f})")
                continue
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                if self.debug:
                    rospy.loginfo(f"[Weiß] {i}: Schwerpunkt nicht berechenbar (m00 = 0)")
                continue
            cx = int(M['m10'] / M['m00'])
            if self.debug:
                rospy.loginfo(f"[Weiß] {i}: Schwerpunkt X = {cx}, Fläche = {area:.1f}")
            if leftmost_x is None or cx < leftmost_x:
                leftmost_x = cx
                cv2.drawContours(image, [cnt], -1, (0, 255, 0), 2)

        rightmost_x = None
        for i, cnt in enumerate(contours_yellow):
            area = cv2.contourArea(cnt)
            if area <= min_area:
                if self.debug:
                    rospy.loginfo(f"[Gelb] {i}: zu klein (Fläche = {area:.1f})")
                continue
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                if self.debug:
                    rospy.loginfo(f"[Gelb] {i}: Schwerpunkt nicht berechenbar (m00 = 0)")
                continue
            cx = int(M['m10'] / M['m00'])
            if self.debug:
                rospy.loginfo(f"[Gelb] {i}: Schwerpunkt X = {cx}, Fläche = {area:.1f}")
            if rightmost_x is None or cx > rightmost_x:
                rightmost_x = cx
                cv2.drawContours(image, [cnt], -1, (0, 255, 255), 2)
        if self.debug:
            rospy.loginfo(f"[Auswertung] Weiß X: {leftmost_x}, Gelb X: {rightmost_x}")

        if leftmost_x is not None and rightmost_x is not None:
            self.pub_right_x.publish(Float64(rightmost_x))
            self.pub_left_x.publish(Float64(leftmost_x))
            return ((leftmost_x + rightmost_x) / 2)
        elif rightmost_x is not None:
            self.pub_right_x.publish(Float64(rightmost_x))
            return rightmost_x + 200
        elif leftmost_x is not None:
            self.pub_left_x.publish(Float64(leftmost_x))
            return leftmost_x - 200
        else:
            return None


    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.image is None:
                rate.sleep()
                continue

            start = time.time()
            image = self.image.copy()
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            wh = self.conf['white']
            gh = self.conf['gelb']

        rhl1, rhh1 = 0, 10
        rhl2, rhh2 = 170,180
        rsl,rsh = 100, 255
        rvl, rvh = 100, 255
        # Maske für Bereich 1
        lower_red1 = np.array([rhl1, rsl, rvl])
        upper_red1 = np.array([rhh1, rsh, rvh])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)

        # Maske für Bereich 2
        lower_red2 = np.array([rhl2, rsl, rvl])
        upper_red2 = np.array([rhh2, rsh, rvh])
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

        # Beide Masken kombinieren
        mask_red = cv2.bitwise_or(mask1, mask2)
        
        mask_white = cv2.inRange(hsv, (wh['hl'], wh['sl'], wh['vl']), (wh['hh'], wh['sh'], wh['vh']))
        mask_yellow = cv2.inRange(hsv, (gh['hl'], gh['sl'], gh['vl']), (gh['hh'], gh['sh'], gh['vh']))

        if self.debug:
            cv2.imshow("hsv-white", mask_white)
            cv2.imshow("hsv-yellow", mask_yellow)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, kernel)
        mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, kernel)
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel)
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, kernel)

        red_pixels = cv2.countNonZero(mask_red)
        threshold = 5000 #Schwellenwert für die rote Line
        
        if red_pixels > threshold:
            #rospy.loginfo("Rote Linie erkannt, stoppe den Duckiebot!")
            self.pub_lane.publish(Float64(0)) #Geschwindigkeit auf 0 setzen
            self.pub_redline.publish(Bool(True))
            if self.debug:
                cv2.imshow("kernel-white", mask_white)
                cv2.imshow("kernel-yellow", mask_yellow)

            polygon = self.create_polygon()
            target_x = self.compute_target_x_from_polygon(polygon, mask_white, mask_yellow, image)

            if target_x is not None:
                self.target_x_buffer.append(target_x)
                if len(self.target_x_buffer) > 2:
                    self.target_x_buffer.pop(0)

                smoothed_x = int(np.mean(self.target_x_buffer))
                target_y = image.shape[0] - 50

            
                cv2.circle(image, (smoothed_x, target_y), 6, (255, 0, 255), -1)
                cv2.putText(image, "Target", (smoothed_x - 20, target_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

                self.pub_lane.publish(Float64(smoothed_x))

            # Mittelpunkt markieren (immer)
            center_x = int((640 / 2))
            center_y = image.shape[0] - 50  # gleiche Höhe wie Target
            cv2.circle(image, (center_x, center_y), 6, (0, 0, 255), -1)
            cv2.putText(image, "Center", (center_x - 25, center_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            cv2.polylines(image, polygon, isClosed=True, color=(255, 255, 255), thickness=2)
            cv2.imshow(self._window, image)
            cv2.waitKey(1)

            if self.debug:
                rospy.loginfo(f"[Latenz] Jetzt: {rospy.Time.now().to_sec():.3f}, Bild-Zeitstempel: {self._timestamp.to_sec():.3f}")
                rospy.loginfo(f"[Verzögerung] {(rospy.Time.now() - self._timestamp).to_sec():.3f} Sekunden")
                rospy.loginfo(f"[Timer] Gesamtzeit: {(time.time() - start):.3f}s")

            rate.sleep()

    def fnShutDown(self):
        with open(self._config_path, 'w') as f:
            yaml.dump(self.conf, f)
        print("Config saved")

if __name__ == '__main__':
    node = CameraReaderNode(node_name='camera_reader_node')
    node.run()
