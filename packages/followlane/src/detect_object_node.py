#!/usr/bin/env python3

import cv2
import rospy
import numpy as np
import os
import yaml
import threading

from collections import deque
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
        
        # start frame_count --> will be deletet
        self.frame_count = 0

        self.cv_image = None

        # start cv bridge
        self._bridge = CvBridge()
        
        # Creating YOLO interface
        self._model = YOLO("packages/followlane/assets/model_detectDuckieBotSlot_V3.pt")  # YOLO model path

        # Read config file
        with open('packages/followlane/config/detect_duckieBotSlot.yaml', 'r') as f:
            self.conf = yaml.safe_load(f)

        # Buffer size in frames
        HISTORY_LENGTH = 5

        # for image annotating
        self.last_boxes = None
        self.frames_since_last = 100

        # Confidence and Bridging for duckie
        self.history_duckie = deque(maxlen=HISTORY_LENGTH)
        self.last_duckie_bbox = None
        self.frames_since_last_duckie = 100  # initial large value
        self.DUCKIE_BRIDGE_LIMIT = self.conf["bridge_limit"]

        # Confidence and Bridging for duckie right
        self.history_duckieRight = deque(maxlen=HISTORY_LENGTH)
        self.last_duckieRight_bbox = None
        self.frames_since_last_duckieRight = 100  # initial large value
        self.DUCKIERIGHT_BRIDGE_LIMIT = self.conf["bridge_limit"]

        # Confidence and Bridging for bot
        self.history_bot = deque(maxlen=HISTORY_LENGTH)
        self.last_bot_bbox = None
        self.frames_since_last_bot = 100  # initial large value
        self.BOT_BRIDGE_LIMIT = self.conf["bridge_limit"]

        # Confidence and Bridging for slot
        self.history_slot = deque(maxlen=HISTORY_LENGTH)
        self.last_slot_bbox = None
        self.frames_since_last_slot = 100  # initial large value
        self.SLOT_BRIDGE_LIMIT = self.conf["bridge_limit"]

        # === Subscribers ===
        # for camera images
        self.sub_image = rospy.Subscriber(f"/{self._vehicle_name}/camera_node/image/compressed", CompressedImage, self.cbDetectObjects, queue_size=1)

        # === Publishers ===
        # for image with bounding boxes
        self.pup_image = rospy.Publisher(f"/{self._vehicle_name}/detect/object/image", Image, queue_size=1)
        # for nearest duckie BBox coordinates (x1, y1, x2, y2)
        self.pup_duckieNearestBB = rospy.Publisher(f"/{self._vehicle_name}/detect/object/duckieNearestBB", Float64MultiArray, queue_size=1)
        # for nearest duckie on right lane BBox coordinates (x1, y1, x2, y2)
        self.pup_duckieNearestRightBB = rospy.Publisher(f"/{self._vehicle_name}/detect/object/duckieNearestRightBB", Float64MultiArray, queue_size=1)
        # for nearest bot BBox coordinates (x1, y1, x2, y2)
        self.pup_botNearestBB = rospy.Publisher(f"/{self._vehicle_name}/detect/object/botNearestBB", Float64MultiArray, queue_size=1)
        # for nearest parking BBox coordinates (x1, y1, x2, y2)
        self.pup_parkingBB = rospy.Publisher(f"/{self._vehicle_name}/detect/object/parkingBB", Float64MultiArray, queue_size=1)



    ##### ===== CALLBACK FUNCTIONS OF SUBSRIBERS ===== #####
    def cbDetectObjects(self, image_msg):
        # Convert CompressedImage message to OpenCV image
        if image_msg is not None:
            self.cv_image = self._bridge.compressed_imgmsg_to_cv2(image_msg)



    ##### ===== OTHER FUNCTIONS ===== #####
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

    

    ##### ========== MAIN RUN FUNCTION ========== #####
    def run(self):
        '''
        Main run function. Running the whole time.
         
        Args:
            self

        Returns:
            none
        '''
        rate = rospy.Rate(5)
        while not rospy.is_shutdown():
            if self.cv_image is None:
                rate.sleep()
                continue

            # Create an empty mask with the same size as the image
            mask = np.zeros(self.cv_image.shape[:2], dtype=np.uint8)
            # Define polygon points for the area of interest
            polygon = self.get_polygon("mask")
            # Draw the polygon on the mask
            cv2.fillPoly(mask, [polygon], 255)

            # Create an empty mask with the same size as the image
            mask_lane = np.zeros(self.cv_image.shape[:2], dtype=np.uint8)
            # Define polygon points for the area of interest
            polygon = self.get_polygon("mask_lane")
            # Draw the polygon on the mask
            cv2.fillPoly(mask_lane, [polygon], 255)

            # Create an empty mask with the same size as the image
            mask_laneRight = np.zeros(self.cv_image.shape[:2], dtype=np.uint8)
            # Shift the polygon of "mask_lane" to the right by 120 pixels
            polygon_lane = self.get_polygon("mask_lane")
            polygon_shifted = polygon_lane.copy()
            polygon_shifted[:, 0] += 180  # shift X coordinates only
            # Draw shifted polygon on right lane mask
            cv2.fillPoly(mask_laneRight, [polygon_shifted], 255)

            # Run YOLO model on the image
            results = self._model(self.cv_image, conf=0.7, iou=0.5, agnostic_nms=True, verbose=False)

            # Plot results on the image
            boxes = results[0].boxes
            annotated_boxes = boxes if boxes is not None and len(boxes) > 0 else None
            # Neue oder überbrückte Boxen verwenden
            if annotated_boxes is not None:
                self.last_boxes = annotated_boxes
                self.frames_since_last = 0
            else:
                if self.last_boxes is not None and self.frames_since_last < self.conf["bridge_limit"]:
                    annotated_boxes = self.last_boxes
                    self.frames_since_last += 1
                else:
                    self.last_boxes = None
                    self.frames_since_last = self.conf["bridge_limit"]
                    annotated_boxes = None
            annotated_frame = self.cv_image.copy()
            if annotated_boxes is not None:
                for box in annotated_boxes:
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    class_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())

                    label = f"{results[0].names[class_id]} {conf:.2f}"
                    cv2.rectangle(annotated_frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0,255,0), 2)
                    cv2.putText(annotated_frame, label, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)


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

            is_occupied = None
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

                # PUBLISHER for free Slot

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

            # === Confidence Updates ===
            self.history_duckie.append(1 if nearest_duckie is not None else 0)
            self.history_duckieRight.append(1 if nearest_duckieRight is not None else 0)
            self.history_bot.append(1 if nearest_bot is not None else 0)
            self.history_slot.append(1 if best_slot_box is not None else 0)

            # Confidence Scores
            conf_duckie = sum(self.history_duckie) / len(self.history_duckie)
            conf_duckieRight = sum(self.history_duckieRight) / len(self.history_duckieRight)
            conf_bot = sum(self.history_bot) / len(self.history_bot)
            conf_slot = sum(self.history_slot) / len(self.history_slot)

            # === Publish ===
            # image
            my_img_msg = self._bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')
            self.pup_image.publish(my_img_msg)

            # duckies
            if nearest_duckie is not None and conf_duckie>self.conf["confidence_temporal"]:
                # New valid Duckie BBox --> save
                x1, y1, x2, y2 = map(int, nearest_duckie.xyxy[0])
                msg_duckieNearestBB = Float64MultiArray(data=[x1, y1, x2, y2])
                self.last_duckie_bbox = msg_duckieNearestBB
                self.frames_since_last_duckie = 0
            elif self.last_duckie_bbox is not None and self.frames_since_last_duckie<self.DUCKIE_BRIDGE_LIMIT:
                # Bridging active
                self.frames_since_last_duckie += 1
            else:
                # No valid detection & no bridging anymore --> Reset
                self.last_duckie_bbox = None
                self.frames_since_last_duckie = 100
                msg_duckieNearestBB = Float64MultiArray(data=None)
            if self.conf['debugPrints_detectObject']:
                rospy.loginfo(f"[nearest Duckie] {msg_duckieNearestBB}")
            self.pup_duckieNearestBB.publish(msg_duckieNearestBB)

            # duckies right
            if nearest_duckieRight is not None and conf_duckieRight>self.conf["confidence_temporal"]:
                # New valid Duckie Right BBox --> save
                x1, y1, x2, y2 = map(int, nearest_duckieRight.xyxy[0])
                msg_duckieNearestRightBB = Float64MultiArray(data=[x1, y1, x2, y2])
                self.last_duckieRight_bbox = msg_duckieNearestRightBB
                self.frames_since_last_duckieRight = 0
            elif self.last_duckieRight_bbox is not None and self.frames_since_last_duckieRight<self.DUCKIERIGHT_BRIDGE_LIMIT:
                # Bridging active
                self.frames_since_last_duckieRight += 1
            else:
                # No valid detection & no bridging anymore --> Reset
                self.last_duckieRight_bbox = None
                self.frames_since_last_duckieRight = 100
                msg_duckieNearestRightBB = Float64MultiArray(data=None)
            if self.conf['debugPrints_detectObject']:
                rospy.loginfo(f"[nearest Duckie Right] {msg_duckieNearestRightBB}")
            self.pup_duckieNearestRightBB.publish(msg_duckieNearestRightBB)

            # bots
            if nearest_bot is not None and conf_bot>self.conf["confidence_temporal"]:
                # New valid Bot BBox --> save
                x1, y1, x2, y2 = map(int, nearest_bot.xyxy[0])
                msg_botNearestBB = Float64MultiArray(data=[x1, y1, x2, y2])
                self.last_bot_bbox = msg_botNearestBB
                self.frames_since_last_bot = 0
            elif self.last_bot_bbox is not None and self.frames_since_last_bot<self.BOT_BRIDGE_LIMIT:
                # Bridging active
                self.frames_since_last_bot += 1
            else:
                # No valid detection & no bridging anymore --> Reset
                self.last_bot_bbox = None
                self.frames_since_last_bot = 100
                msg_botNearestBB = Float64MultiArray(data=None)
            if self.conf['debugPrints_detectObject']:
                rospy.loginfo(f"[nearest Bot] {msg_botNearestBB}")
            self.pup_botNearestBB.publish(msg_botNearestBB)

            # free slot
            if best_slot_box is not None and not is_occupied and conf_slot>self.conf["confidence_temporal"]:
                # New valid Slot BBox --> save
                msg_parkingNearestBB = Float64MultiArray(data=best_slot_box)
                self.last_slot_bbox = msg_parkingNearestBB
                self.frames_since_last_slot = 0
            elif self.last_slot_bbox is not None and self.frames_since_last_slot<self.SLOT_BRIDGE_LIMIT:
                # Bridging active
                self.frames_since_last_slot += 1
            else:
                # No valid detection & no bridging anymore --> Reset
                self.last_slot_bbox = None
                self.frames_since_last_slot = 100
                msg_parkingNearestBB = Float64MultiArray(data=None)
            if self.conf['debugPrints_detectObject']:
                rospy.loginfo(f"[nearest Slot] {msg_parkingNearestBB}")
            self.pup_parkingBB.publish(msg_parkingNearestBB)
            
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


if __name__ == '__main__':
    node = DetectParkingSlotNode(node_name='detect_duckieBotSlot_node')
    node.run()
