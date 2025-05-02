#!/usr/bin/env python3

import os
import rospy
import numpy as np
import cv2
from sensor_msgs.msg import CompressedImage
import yaml
from duckietown.dtros import DTROS, NodeType

class DetectLaneNode(DTROS):
    def __init__(self, node_name):
        super(DetectLaneNode, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)

        self.load_conf('packages/followlane/config/detect_lane.yaml')
        self._vehicle_name = os.environ['VEHICLE_NAME']
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"

        self.sub_image_original = rospy.Subscriber(
            self._camera_topic, CompressedImage, self.cbFindLane, queue_size=1
        )

        self.counter = 0

    def crop_img(self, img):
        img = img.copy()
        pts1 = np.float32([
            [self.conf['lane_image']['top_left_x'],     self.conf['lane_image']['top_left_y']],
            [self.conf['lane_image']['top_right_x'],    self.conf['lane_image']['top_right_y']],
            [self.conf['lane_image']['bottom_right_x'], self.conf['lane_image']['bottom_right_y']],
            [self.conf['lane_image']['bottom_left_x'],  self.conf['lane_image']['bottom_left_y']],
        ])
        pts2 = np.float32([[0,0],[100,0],[0,100],[100,100]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        return cv2.warpPerspective(img, M, (100,100))

    def cbFindLane(self, image_msg):
        if self.counter % 3 != 0:
            self.counter += 1
            return
        self.counter += 1

        np_arr = np.frombuffer(image_msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        img = self.crop_img(cv_image)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        mask_yellow = cv2.inRange(
            hsv,
            (self.hue_yellow_l, self.saturation_yellow_l, self.lightness_yellow_l),
            (self.hue_yellow_h, self.saturation_yellow_h, self.lightness_yellow_h)
        )

        mask_white = cv2.inRange(
            hsv,
            (self.hue_white_l, self.saturation_white_l, self.lightness_white_l),
            (self.hue_white_h, self.saturation_white_h, self.lightness_white_h)
        )

        yellow_detected = np.count_nonzero(mask_yellow) > 50
        white_detected = np.count_nonzero(mask_white) > 50

        if yellow_detected and white_detected:
            print("Both yellow and white detected")
        elif yellow_detected:
            print("Yellow detected")
        elif white_detected:
            print("White detected")
        else:
            print("No yellow or white detected")

    def load_conf(self, path):
        with open(path, 'r') as f:
            text = f.read()
        self.conf = yaml.safe_load(text)

        # Farbwerte laden
        self.hue_white_l = self.conf['white']['hl']
        self.hue_white_h = self.conf['white']['hh']
        self.saturation_white_l = self.conf['white']['sl']
        self.saturation_white_h = self.conf['white']['sh']
        self.lightness_white_l = self.conf['white']['vl']
        self.lightness_white_h = self.conf['white']['vh']

        self.hue_yellow_l = self.conf['yellow']['hl']
        self.hue_yellow_h = self.conf['yellow']['hh']
        self.saturation_yellow_l = self.conf['yellow']['sl']
        self.saturation_yellow_h = self.conf['yellow']['sh']
        self.lightness_yellow_l = self.conf['yellow']['vl']
        self.lightness_yellow_h = self.conf['yellow']['vh']

        # Duck-Erkennung aktuell ungenutzt
        self.hue_duck_l = self.conf['duck']['hl']
        self.hue_duck_h = self.conf['duck']['hh']
        self.saturation_duck_l = self.conf['duck']['sl']
        self.saturation_duck_h = self.conf['duck']['sh']
        self.lightness_duck_l = self.conf['duck']['vl']
        self.lightness_duck_h = self.conf['duck']['vh']

if __name__ == '__main__':
    node = DetectLaneNode(node_name='detect_lane_node')
    rospy.spin()

