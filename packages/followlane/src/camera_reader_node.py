#!/usr/bin/env python3

import os
import rospy
import cv2
import yaml
import numpy as np
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from std_msgs.msg import Float64
from std_msgs.msg import Bool

class CameraReaderNode(DTROS):

    def __init__(self, node_name):
        super(CameraReaderNode, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)

        self._vehicle_name = os.environ['VEHICLE_NAME']
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        self._bridge = CvBridge()
        self._window = "camera-reader"

        self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.callback)
        self.pub_lane = rospy.Publisher(f"/{self._vehicle_name}/detect/lane", Float64, queue_size=1)
        self.pub_redline = rospy.Publisher(f"/{self._vehicle_name}/stop_line_detected",  Bool, queue_size=1)

        with open('packages/followlane/config/detect_lane.yaml','r') as f:
            self.conf = yaml.safe_load(f)

        self.target_x_buffers = {i: [] for i in range(5)}  # 5 Boxen (idx 0–4)
        rospy.on_shutdown(self.fnShutDown)

    def create_polygon_offset(self, offset_y):
        shrink = offset_y * 0.9  # je höher offset, desto schmaler die Box
        return np.array([[
            [self.conf['lane_image']['top_left_x'] + shrink, self.conf['lane_image']['top_left_y'] - offset_y],
            [self.conf['lane_image']['top_right_x'] - shrink, self.conf['lane_image']['top_right_y'] - offset_y],
            [self.conf['lane_image']['bottom_right_x'] - shrink, self.conf['lane_image']['bottom_right_y'] - offset_y],
            [self.conf['lane_image']['bottom_left_x'] + shrink, self.conf['lane_image']['bottom_left_y'] - offset_y],
        ]], dtype=np.int32)

    # def compute_target_x_from_polygon(self, polygon, mask_white, mask_yellow, mask_red, image, idx):
    #     mask_poly = np.zeros_like(mask_white)
    #     cv2.fillPoly(mask_poly, polygon, 255)

    #     mw = cv2.bitwise_and(mask_white, mask_poly)
    #     my = cv2.bitwise_and(mask_yellow, mask_poly)
    #     mr = cv2.bitwise_and(mask_red, mask_poly)

    #     image[mw > 0] = (0, 255, 0)
    #     image[my > 0] = (255, 0, 0)
    #     image[mr > 0] = (0, 0, 255)

    #     def get_lines(mask):
    #         blurred = cv2.GaussianBlur(mask, (5, 5), 0)
    #         edges = cv2.Canny(blurred, 50, 150)
    #         return cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=15, minLineLength=10, maxLineGap=300)


    #     lines_white = get_lines(mw)
    #     lines_yellow = get_lines(my)
    #     lines_red = get_lines(mr)

    #     # Linien für Visualisierung einzeichnen
    #     if lines_white is not None:
    #         for x1, y1, x2, y2 in lines_white[:, 0]:
    #             cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    #     if lines_yellow is not None:
    #         for x1, y1, x2, y2 in lines_yellow[:, 0]:
    #             cv2.line(image, (x1, y1), (x2, y2), (0, 255, 255), 2)

    #     x_white = None
    #     x_yellow = None

    #     if lines_white is not None:
    #         white_xs = [(x1 + x2) / 2 for [[x1, _, x2, _]] in lines_white if ((x1 + x2) / 2) >= 310]
    #         if white_xs:
    #             x_white = min(white_xs)

    #     if lines_yellow is not None:
    #         yellow_xs = [(x1 + x2) / 2 for [[x1, _, x2, _]] in lines_yellow if ((x1 + x2) / 2) <= 340]
    #         if yellow_xs:
    #             x_yellow = max(yellow_xs)

    #     if x_white is not None and x_yellow is not None:
    #         return (x_white + x_yellow + 120) / 2
    #     elif x_yellow is not None:
    #         return x_yellow + 260
    #     elif x_white is not None:
    #         offset = 180 - 4.17 * idx ** 1.4
    #         return (x_white - offset)
    #     else:
    #         return None

    def compute_target_x_from_polygon(self, polygon, mask_white, mask_yellow, mask_red, image, idx):
        # Maske auf Polygonbereich begrenzen
        mask_poly = np.zeros_like(mask_white)
        cv2.fillPoly(mask_poly, polygon, 255)

        # Masken im Polygonbereich isolieren
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
        else:
            return None



    def callback(self, msg):
        whl, whh = self.conf['white']['hl'], self.conf['white']['hh']
        wsl, wsh = self.conf['white']['sl'], self.conf['white']['sh']
        wvl, wvh = self.conf['white']['vl'], self.conf['white']['vh']

        ghl, ghh = self.conf['gelb']['hl'], self.conf['gelb']['hh']
        gsl, gsh = self.conf['gelb']['sl'], self.conf['gelb']['sh']
        gvl, gvh = self.conf['gelb']['vl'], self.conf['gelb']['vh']

        rhl1, rhh1 = 0, 10
        rhl2, rhh2 = 170,180
        rsl,rsh = 100, 255
        rvl, rvh = 100, 255
        

        image = self._bridge.compressed_imgmsg_to_cv2(msg)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        mask_white = cv2.inRange(hsv, (whl, wsl, wvl), (whh, wsh, wvh))
        mask_yellow = cv2.inRange(hsv, (ghl, gsl, gvl), (ghh, gsh, gvh))
        mask_red1 = cv2.inRange(hsv, (rhl1,rsl,rvl), (rhh1, rsh, rvh))
        mask_red2 = cv2.inRange(hsv, (rhl2,rsl,rvl), (rhh2, rsh, rvh))
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)

        red_pixels = cv2.countNonZero(mask_red)
        threshold = 5000 #Schwellenwert für die rote Line
        
        if red_pixels > threshold:
            rospy.loginfo("Rote Linie erkannt, stoppe den Duckiebot!")
            self.pub_lane.publish(Float64(0)) #Geschwindigkeit auf 0 setzen
            self.pub_redline.publish(Bool(True))


        polygons = [self.create_polygon_offset(offset_y=i*40) for i in range(5)] #offset kann geändert werden

        target_xs_raw = []

        for idx, poly in enumerate(polygons):
            tx = self.compute_target_x_from_polygon(poly, mask_white, mask_yellow, mask_red, image, idx=idx)
            if tx is not None:
                target_xs_raw.append((idx, tx))

        for idx, (poly, tx_raw) in enumerate(zip(polygons, [tx for _, tx in target_xs_raw])):
            self.target_x_buffers[idx].append(tx_raw)
            if len(self.target_x_buffers[idx]) > 15:
                self.target_x_buffers[idx].pop(0)

            smoothed_x = int(np.mean(self.target_x_buffers[idx]))


            # Y-Mitte der Polygonbox berechnen
            y_coords = [point[1] for point in poly[0]]
            target_y = int(sum(y_coords) / len(y_coords))

            # Visualisierung
            cv2.circle(image, (smoothed_x, target_y), 6, (255, 0, 255), -1)
            cv2.putText(image, f"Target{idx+1}", (smoothed_x - 30, target_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

            if idx == 0:
                self.pub_lane.publish(Float64(smoothed_x))

        for poly in polygons:
            cv2.polylines(image, poly, isClosed=True, color=(255, 255, 255), thickness=2)


        cv2.imshow(self._window, image)
        cv2.waitKey(1)

    def fnShutDown(self):
        with open('packages/followlane/config/detect_lane.yaml','w') as f:
            yaml.dump(self.conf, f)
        print("Config saved")

def nothing(x): pass

if __name__ == '__main__':
    node = CameraReaderNode(node_name='camera_reader_node')
    rospy.spin()