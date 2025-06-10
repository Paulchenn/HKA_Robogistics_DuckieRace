#!/usr/bin/env python3

import cv2
import rospy
import numpy as np
import os
import yaml

from cv_bridge import CvBridge
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Float64
from ultralytics import YOLO


class DetectDuckieNode(DTROS):
    def __init__(self, node_name):
        # initialize the DTROS parent class
        super(DetectDuckieNode, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)
        
        self._model = YOLO("packages/followlane/assets/model.pt") # copy of assets/yolo_duckies/results/duckies/duckie-train2/weights/best.pt

        self._vehicle_name = os.environ['VEHICLE_NAME']
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        self.sub_image = rospy.Subscriber(self._camera_topic, CompressedImage, self.cbDetectObjects, queue_size = 1)
        self._yolo_topic = f"/{self._vehicle_name}/detect/duckie/image"
        self.pup_image = rospy.Publisher(self._yolo_topic,Image,queue_size = 1)
        self._duckie_topic = f"/{self._vehicle_name}/detect/duckie"
        self.pup_duckie = rospy.Publisher(self._duckie_topic,Float64,queue_size = 1)

        with open('packages/followlane/config/detect_lane.yaml','r') as f:
            self.conf = yaml.safe_load(f)

        self.counter = 0
        self.bridge = CvBridge()

        self._window = "duckie-detection"


    def cbDetectObjects(self, image_msg):
        '''
        if self.counter % 3 != 0:
            self.counter += 1
            return
        else:
            self.counter += 1
        '''

        # 1. get image from camera
        np_arr = np.frombuffer(image_msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # 2. Create an empty mask with the same size as the image
        mask = np.zeros(cv_image.shape[:2], dtype=np.uint8)

        # 3.1. Create a polygon that defines the area of interest
        polygon = np.array([
            [self.conf['lane_image']['top_left_x'], self.conf['lane_image']['top_left_y']],
            [self.conf['lane_image']['top_right_x'], self.conf['lane_image']['top_right_y']],
            [self.conf['lane_image']['bottom_right_x'], self.conf['lane_image']['bottom_right_y']],
            [self.conf['lane_image']['bottom_left_x'], self.conf['lane_image']['bottom_left_y']],
        ], dtype=np.int32)
        # 3.2. Create a polygon that defines the area of interest and draw it on the mask
        cv2.fillPoly(mask, [polygon], 255)

        # 4. Apply the mask to the image
        masked_image = cv2.bitwise_and(cv_image, cv_image, mask=mask)

        # 5. Run YOLOv8 model on the masked image
        results = self._model(masked_image, conf=0.5, iou=0.5, agnostic_nms=True)

        # 6. Get bounding boxes from YOLO        
        boxes = results.boxes.xyxy   # <---- ERROR HEERE !!!

        # 7. Filter bounding boxes: keep only those whose center is inside the polygon
        inside_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # center of the box
            if mask[cy, cx] == 255:
                inside_boxes.append(box)
                
        # 8. Optional: Show image and polygon
        #img_poly = cv_image.copy()
        #cv2.polylines(img_poly, [polygon], isClosed=True, color=(0, 255, 0), thickness=2)
        #cv2.imshow("Polygon", img_poly)
        #cv2.imshow("Mask", mask)
        #cv2.waitKey(1)
        #cv2.destroyAllWindows()

        '''
        nearest_box = get_nearest_box(results)
        x = int(nearest_box.xyxy[0][2])
        y = int(nearest_box.xyxy[0][3])
        rospy.loginfo(f"nearest duckie: ({x}, {y})")

        image = draw_bounding_boxes(results,cv_image)

        msg = self.bridge.cv2_to_imgmsg(image, "bgr8")
        self.pup_image.publish(msg)

        cv2.imshow(self._window, image)
        cv2.waitKey(1)
        '''
       


def draw_bounding_boxes(results,img):
    for result in results:
        for box in result.boxes:
            cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                          (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), 1)
            cv2.putText(img, f"{result.names[int(box.cls[0])]}",
                        (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
    return img


def get_nearest_box(results):
    y_min = float('-inf')
    box_with_min_y = None

    for result in results:
        for box in result.boxes:
            if box.xyxy[0][3] >= y_min:
                box_with_min_y = box
                y_min = box.xyxy[0][3]
                
    return box_with_min_y




if __name__ == '__main__':

    node = DetectDuckieNode(node_name='detect_duckie_node')
    rospy.spin()