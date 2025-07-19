#!/usr/bin/env python3

import cv2
import rospy
import numpy as np
import os
import yaml
import threading

from cv_bridge import CvBridge
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Float64, Float64MultiArray
from ultralytics import YOLO


class FilteredResults:
    def __init__(self, names):
        self.names = names
        self.boxes = []
        self.nearest = None  # Store the nearest box here (not a list)


class DetectParkingSlotNode(DTROS):
    def __init__(self, node_name):
        # Initialize the DTROS parent class
        super(DetectParkingSlotNode, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)
        self._vehicle_name = os.environ['VEHICLE_NAME']
        
        self.frame_count = 0
        
        self._model = YOLO("packages/followlane/assets/model_detectDuckieBotSlot_V3.pt")  # YOLO model path

        # Subscriber for camera images
        # The camera topic is constructed using the vehicle name from the environment variable
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        self.sub_image = rospy.Subscriber(self._camera_topic, CompressedImage, self.cbDetectObjects, queue_size=1)

        # Publisher for image with bounding boxes
        self._yolo_topic = f"/{self._vehicle_name}/detect/object/image"
        self.pup_image = rospy.Publisher(self._yolo_topic, Image, queue_size=1)

        # Publisher for nearest duckie coordinates
        # Duckie nearest Bounding Box (BB) coordinates (x1, y1, x2, y2)
        self._duckieNearestBB_topic = f"/{self._vehicle_name}/detect/object/duckieNearestBB"
        self.pup_duckieNearestBB = rospy.Publisher(self._duckieNearestBB_topic, Float64MultiArray, queue_size=1)
        # Duckie nearest Bounding Box on right lane (BB) coordinates (x1, y1, x2, y2)
        self._duckieNearestRightBB_topic = f"/{self._vehicle_name}/detect/object/duckieNearestRightBB"
        self.pup_duckieNearestRightBB = rospy.Publisher(self._duckieNearestRightBB_topic, Float64MultiArray, queue_size=1)

        # Publisher for nearest bot coordinates
        # Duckie nearest Bounding Box (BB) coordinates (x1, y1, x2, y2)
        self._botNearestBB_topic = f"/{self._vehicle_name}/detect/object/botNearestBB"
        self.pup_botNearestBB = rospy.Publisher(self._botNearestBB_topic, Float64MultiArray, queue_size=1)

        # Publisher for nearest parking coordinates
        # Duckie nearest Bounding Box (BB) coordinates (x1, y1, x2, y2)
        self._parkingBB_topic = f"/{self._vehicle_name}/detect/object/parkingBB"
        self.pup_parkingBB = rospy.Publisher(self._parkingBB_topic, Float64MultiArray, queue_size=1)

        with open('packages/followlane/config/detect_duckieBotSlot.yaml', 'r') as f:
            self.conf = yaml.safe_load(f)

        self._bridge = CvBridge()
        

    def cbDetectObjects(self, image_msg):
        self.frame_count += 1
        if self.frame_count % 10 != 0:  # Take only every tenth picture
            return

        # Convert CompressedImage message to OpenCV image
        cv_image = self._bridge.compressed_imgmsg_to_cv2(image_msg)
        
        # Create an empty mask with the same size as the image
        mask = np.zeros(cv_image.shape[:2], dtype=np.uint8)
        # Define polygon points for the area of interest
        polygon = self.get_polygon("mask")
        # Draw the polygon on the mask
        cv2.fillPoly(mask, [polygon], 255)

        # Create an empty mask with the same size as the image
        mask_lane = np.zeros(cv_image.shape[:2], dtype=np.uint8)
        # Define polygon points for the area of interest
        polygon = self.get_polygon("mask_lane")
        # Draw the polygon on the mask
        cv2.fillPoly(mask_lane, [polygon], 255)

        # Create an empty mask with the same size as the image
        mask_laneRight = np.zeros(cv_image.shape[:2], dtype=np.uint8)
        # Shift the polygon of "mask_lane" to the right by 120 pixels
        polygon_lane = self.get_polygon("mask_lane")
        polygon_shifted = polygon_lane.copy()
        polygon_shifted[:, 0] += 180  # shift X coordinates only
        # Draw shifted polygon on right lane mask
        cv2.fillPoly(mask_laneRight, [polygon_shifted], 255)


        # Run YOLO model on the image
        results = self._model(cv_image, conf=0.65, iou=0.5, agnostic_nms=True, verbose=False)

        # Plot results on the image
        annotated_frame = results[0].plot()

        # Publish image mit Bounding Boxes
        if False: #annotated_frame is not None:
            img_msg = self._bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')
            self.pup_image.publish(img_msg)

        # Filter detected duckies inside polygon mask and find nearest one
        filteredResults_duckie, nearest_duckie = self.get_filteredResults_and_nearest(
            results,
            "duckie",
            mask_lane,
            filtered_results=[],
            y_lowest=0
        )
        filteredResults_duckieRight, nearest_duckieRight = self.get_filteredResults_and_nearest(
            results,
            "duckie",
            mask_laneRight,
            filtered_results=[],
            y_lowest=0
        )
        # Fil
        # Filter detected bots inside polygon mask and find nearest one
        filteredResults_bot, nearest_bot = self.get_filteredResults_and_nearest(
            results,
            "bot",
            mask,
            filtered_results=[],
            y_lowest=0
        )
        # Filter detected free Slots inside polygon mask and find nearest one
        filteredResults_freeSlot, nearest_freeSlot = self.get_filteredResults_and_nearest(
            results,
            "freeSlot",
            mask,
            filtered_results=[],
            y_lowest=0
        )
        # Filter detected bots inside polygon mask and find nearest one
        filteredResults_occSlot, nearest_occSlot = self.get_filteredResults_and_nearest(
            results,
            "occupiedSlot",
            mask,
            filtered_results=[],
            y_lowest=0
        )

        # === Slot Evaluation with IoU (ALL bots & duckies) ===
        def compute_max_iou(slot_box, object_list):
            """Compute maximum IoU between slot_box and all objects in list"""
            max_iou = 0.0
            for obj in object_list:
                obj_box = [int(x) for x in obj.boxes[0].xyxy[0]]
                iou = self.compute_iou(slot_box, obj_box)
                max_iou = max(max_iou, iou)
            return max_iou

        best_slot_box = None
        best_slot_type = None  # "freeSlot" or "occupiedSlot"
        best_iou = 0.0

        iou_threshold = self.conf["iou_thresh"]  # Tunable

        # Collect all detected slot boxes (free + occupied)
        all_slots = []
        if nearest_freeSlot:
            all_slots.append(("freeSlot", nearest_freeSlot))
        if nearest_occSlot:
            all_slots.append(("occupiedSlot", nearest_occSlot))

        # Prepare Duckie- und Bot-Listen for IoU-Calc
        duckie_objects = filteredResults_duckie
        bot_objects = filteredResults_bot

        for slot_type, slot_box in all_slots:
            box_slot = [int(x) for x in slot_box.xyxy[0]]

            if not duckie_objects==None:
                iou_duckie = compute_max_iou(box_slot, duckie_objects)
            if not duckie_objects==None:
                iou_bot = compute_max_iou(box_slot, bot_objects)
            max_iou = max(iou_duckie, iou_bot)

            if max_iou > best_iou:
                best_iou = max_iou
                best_slot_box = box_slot
            elif best_slot_box is None:
                best_slot_box = box_slot
            # best_slot_type = slot_type

        if self.conf["debugPrints_detectObject"]:
            rospy.loginfo(f"[OBJECT DETECTION] {best_iou}")

        if best_slot_box is not None:
            is_occupied = best_iou > iou_threshold

            if self.conf["debugPrints_detectObject"]:
                rospy.loginfo(f"[OBJECT DETECTION] {best_slot_box}")

            # Publish slot position (always)
            if not is_occupied:
                msg_parkingNearestBB = Float64MultiArray(data=best_slot_box)
                self.pup_parkingBB.publish(msg_parkingNearestBB)

            color = (0, 0, 255) if is_occupied else (0, 255, 0)
            label = 'Slot: OCCUPIED' if is_occupied else 'Slot: FREE'
            cv2.rectangle(annotated_frame, (best_slot_box[0], best_slot_box[1]),
                          (best_slot_box[2], best_slot_box[3]),
                          color, 2)
            cv2.putText(annotated_frame, label, (best_slot_box[0], best_slot_box[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
        # Finales Bild mit Slot-Zustand senden
        # Zeichne Masken-Konturen ein (nur visuell, ändert die Maske nicht)

        cv2.polylines(annotated_frame, [polygon_lane], isClosed=True, color=(0, 255, 255), thickness=2)
        cv2.polylines(annotated_frame, [polygon_shifted], isClosed=True, color=(255, 0, 255), thickness=2)


        my_img_msg = self._bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')
        self.pup_image.publish(my_img_msg)

        # === Publish ===
        if nearest_duckie is not None:
            x1, y1, x2, y2 = map(int, nearest_duckie.xyxy[0])
            msg_duckieNearestBB = Float64MultiArray(data=[x1, y1, x2, y2])
            self.pup_duckieNearestBB.publish(msg_duckieNearestBB)
        if nearest_duckieRight is not None:
            x1, y1, x2, y2 = map(int, nearest_duckieRight.xyxy[0])
            msg_duckieNearestRightBB = Float64MultiArray(data=[x1, y1, x2, y2])
            self.pup_duckieNearestRightBB.publish(msg_duckieNearestRightBB)
        if nearest_bot is not None:
            x1, y1, x2, y2 = map(int, nearest_bot.xyxy[0])
            msg_botNearestBB = Float64MultiArray(data=[x1, y1, x2, y2])
            self.pup_botNearestBB.publish(msg_botNearestBB)
        
        if False: #self.conf['debugPrints_detectObject']:
            print('==========')
            print('Duckies:')
            print(' ', filteredResults_duckie)
            print(' ', nearest_duckie)
            print('----------')
            print('Bots:')
            print(' ', filteredResults_bot)
            print(' ', nearest_bot)
            print('----------')
            print('free Slot:')
            print(' ', filteredResults_freeSlot)
            print(' ', nearest_freeSlot)
            print('----------')
            print('occupied Slot:')
            print(' ', filteredResults_occSlot)
            print(' ', nearest_occSlot)
                
                
    def get_polygon(self, mask_name):
        return np.array([
            [self.conf[mask_name]['top_left_x'], self.conf[mask_name]['top_left_y']],
            [self.conf[mask_name]['top_right_x'], self.conf[mask_name]['top_right_y']],
            [self.conf[mask_name]['bottom_right_x'], self.conf[mask_name]['bottom_right_y']],
            [self.conf[mask_name]['bottom_left_x'], self.conf[mask_name]['bottom_left_y']],
        ], dtype=np.int32)
        
        
    def get_filteredResults_and_nearest(
            self,
            results,
            name,
            mask,
            filtered_results=[],
            y_lowest=0
        ):
        myClass = self.conf['classes'][name]
        
        for result in results:
            filtered = FilteredResults(names=name)

            for box in result.boxes:
                if box.cls[0] == myClass:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    if mask[cy, cx] == 255:
                        filtered.boxes.append(box)
                        if y2 >= y_lowest:
                            filtered.nearest = box
                            y_lowest = y2

            # Only add if at least one box found
            if filtered.boxes:
                filtered_results.append(filtered)

        return filtered_results, filtered.nearest
    

    def compute_iou(self, box1, box2):
        """Compute IoU between two bounding boxes (format: [x1, y1, x2, y2])"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area

        if union_area == 0:
            return 0.0
        return inter_area / union_area


if __name__ == '__main__':
    node = DetectParkingSlotNode(node_name='detect_duckieBotSlot_node')
    rospy.spin()
