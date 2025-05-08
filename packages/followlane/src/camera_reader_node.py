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

class CameraReaderNode(DTROS):

    def __init__(self, node_name):
        super(CameraReaderNode, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)

        self._vehicle_name = os.environ['VEHICLE_NAME']
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        self._bridge = CvBridge()
        self._window = "camera-reader"

        self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.callback)
        self.pub_lane = rospy.Publisher(f"/{self._vehicle_name}/detect/lane", Float64, queue_size=1)

        with open('packages/followlane/config/detect_lane.yaml','r') as f:
            self.conf = yaml.safe_load(f)

        self.names = ['white','yellow','duck','lane image']
        self.name = self.names[0]

        rospy.on_shutdown(self.fnShutDown)

    def callback(self, msg):
        
        # ROI und HSV-Werte laden
        x1 = self.conf['lane_image']['top_left_x']
        y1 = self.conf['lane_image']['top_left_y']
        x2 = self.conf['lane_image']['bottom_right_x']
        y2 = self.conf['lane_image']['bottom_right_y']

        hl = self.conf['white']['hl']
        hh = self.conf['white']['hh']
        sl = self.conf['white']['sl']
        sh = self.conf['white']['sh']
        vl = self.conf['white']['vl']
        vh = self.conf['white']['vh']

        image = self._bridge.compressed_imgmsg_to_cv2(msg)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        roi = hsv[y1:y2, x1:x2]
        mask = cv2.inRange(roi, (hl, sl, vl), (hh, sh, vh))

        roi_color = image[y1:y2, x1:x2]
        roi_color[mask > 0] = (0, 255, 0)

        # === HoughLines-Auswertung ===
        blurred = cv2.GaussianBlur(mask, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=10)

        # Linienmittelpunkt berechnen
        line_centers = []
        if lines is not None:
            roi_color = image[y1:y2, x1:x2]  # nur einmal holen
            for line in lines:
                x1_l, y1_l, x2_l, y2_l = line[0]
                cv2.line(roi_color, (x1_l, y1_l), (x2_l, y2_l), (0, 0, 255), 2)
                mid_x = (x1_l + x2_l) / 2 + x1  # globale Koordinate
                line_centers.append(mid_x)

        if line_centers:
            avg_x = float(np.mean(line_centers))
            self.pub_lane.publish(Float64(avg_x))

        # Box zeichnen
        x_alt, y_alt = 0, 0
        for point in ['top_left', 'top_right', 'bottom_left', 'bottom_right', 'top_left']:
            x = self.conf['lane_image'][f'{point}_x']
            y = self.conf['lane_image'][f'{point}_y']
            if x_alt != 0 or y_alt != 0:
                image = cv2.line(image, (x_alt, y_alt), (x, y), (255, 255, 255), 2)
            x_alt, y_alt = x, y

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