#!/usr/bin/env python3

import cv2
import yaml
import numpy as np
import os
import rospy
import random
from duckietown.dtros import DTROS, NodeType
from std_msgs.msg import Bool, Int32, Float64, ColorRGBA, Float64MultiArray
from sensor_msgs.msg import CompressedImage
from duckietown_msgs.msg import LEDPattern
import time

class RedLineDetector(DTROS):
    def __init__(self, node_name):
        super(RedLineDetector, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)

        self._vehicle_name = os.environ['VEHICLE_NAME']
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        self._redLine_topic = f"/{self._vehicle_name}/stop_line_detected"
        self._config_path = 'packages/followlane/config/detect_lane.yaml'
        with open(self._config_path, 'r') as f:
            self.conf = yaml.safe_load(f)

        self.pub_info = rospy.Publisher(f"/{self._vehicle_name}/abfrage_info", Int32, queue_size=1)
        self.pub_target_x = rospy.Publisher(f"/{self._vehicle_name}/target_x", Int32, queue_size=1)
        self.led_pub = rospy.Publisher(f"/{self._vehicle_name}/led_emitter_node/led_pattern", LEDPattern, queue_size=1)

        self.sub_image = rospy.Subscriber(self._camera_topic, CompressedImage, self.cb_image, queue_size=1)
        self.sub_redline = rospy.Subscriber(self._redLine_topic, Bool, self.process_stop_line, queue_size=1)
        self.sub_left_x = rospy.Subscriber(f"/{self._vehicle_name}/detect/lane/left_x", Float64, self.cb_left_x)
        self.sub_bots = rospy.Subscriber(f"/{self._vehicle_name}/detect/object/botNearestBB",Float64MultiArray,self.cb_bots,queue_size=1)

        self.detected_bots = []  # Liste der aktuell erkannten Bot-Bounding-Boxes

        # Blinken initialisieren
        self.pattern_on = LEDPattern()
        self.pattern_on.frequency = 2.0
        self.blink_on = True
        self.blink_timer = None  # wird dynamisch erstellt

        self.current_image = None

        self.waiting_at_line = False
        self.wait_start_time = None
        self.abbiegephase_gestartet = False
        self.abgeschlossen = False
        self.chosen_direction = None
        self.left_x_value = None
        self.abbiege_start_time = None
        self.last_turn_completed_time = None
        self.rechts_vor_links_freigegeben = False

        # Polygon für die Kreuzungsfläche im Kamerabild
        self.intersection_area = np.array([
            [130, 260],
            [370, 260],
            [520, 400],
            [0, 400]
        ], dtype=np.int32)


        self.debug = False
        self.debug_important = True
        self.debug_window = True

        if self.debug:
            rospy.loginfo("[RedLineDetector] Debug-Modus aktiviert")

    def cb_image(self, msg):
        # Bild puffern
        np_arr = np.frombuffer(msg.data, np.uint8)
        self.current_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    def cb_left_x(self, msg):
        self.left_x_value = msg.data
        # if self.debug:
        #     rospy.loginfo(f"[Abbiegen] Empfange weißen Left-X: {self.left_x_value}")

    def cb_bots(self, msg):
        """
        Callback für erkannte Duckiebots – wandelt Flat-Array in Bounding-Box-Listen um.
        """
        data = msg.data
        boxes = []

        if len(data) % 4 != 0:
            rospy.logwarn("[RedLineDetector] Ungültige Bot-Koordinaten empfangen (nicht durch 4 teilbar)")
            return

        for i in range(0, len(data), 4):
            x1 = int(data[i])
            y1 = int(data[i + 1])
            x2 = int(data[i + 2])
            y2 = int(data[i + 3])
            boxes.append((x1, y1, x2, y2))

        self.detected_bots = boxes


    def process_stop_line(self, msg):
        current_time = rospy.get_time()
        in_cooldown = (
            self.last_turn_completed_time is not None and
            (current_time - self.last_turn_completed_time) < 2.0
        )

        if msg.data and not in_cooldown:
            self.abgeschlossen = False
            if self.debug:
                rospy.loginfo_throttle(5, "[STOP] Stoplinie erkannt")
            if not self.abbiegephase_gestartet and not self.waiting_at_line:
                self.stop_line_detected = True
                self.waiting_at_line = True
                self.wait_start_time = current_time
                self.rechts_vor_links_freigegeben = False
        elif msg.data and in_cooldown:
            if self.debug:
                rospy.loginfo_throttle(5, "[STOP] Stoplinie ignoriert – Cooldown aktiv")
    
    def get_box_area(self, x1, y1, x2, y2):
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        return width * height

    def check_rechts_vor_links(self):
        """
        Überprüft die 'Rechts vor Links'-Regel basierend auf der gewählten Richtung.
        Gibt True zurück, wenn das Fahrzeug fahren darf, sonst False.
        """

        # --- Sicherheitsprüfung: Ist bereits ein Bot auf der Kreuzung? ---
        for bot in self.detected_bots:
            x1, y1, x2, y2 = bot

            # Prüfe alle 4 Ecken der Bounding Box
            corners = [
                (x1, y1),  # oben links
                (x2, y1),  # oben rechts
                (x2, y2),  # unten rechts
                (x1, y2)   # unten links
            ]

            for corner in corners:
                if cv2.pointPolygonTest(self.intersection_area, corner, False) >= 0:
                    if self.debug:
                        rospy.loginfo("[Abbiegen] Bot-Ecke auf Kreuzung erkannt – Warten")
                    return False  # Eine Ecke innerhalb → Bot blockiert Kreuzung

        # --- Jetzt geht's weiter mit richtungsabhängiger Prüfung ---
        if self.chosen_direction == "rechts":
            # Regel: Solange kein Bot auf der Kreuzung ist, dürfen wir rechts abbiegen
            if self.debug:
                rospy.loginfo("[Rechts vor Links] Rechtsabbiegen – Kein Bot auf Kreuzung → Erlaubt")
            return True

        elif self.chosen_direction == "links":
            for bot in self.detected_bots:
                x1, y1, x2, y2 = bot
                center_x = (x1 + x2) // 2
                area = self.get_box_area(x1, y1, x2, y2)

                # Regel 1: Bot rechts von uns?
                if center_x > 350 and area >= 6000:
                    if self.debug:
                        rospy.loginfo("[Rechts vor Links] Großer Bot rechts (x=%d, area=%d) – Warten (links)", center_x, area)
                    return False

                # Regel 2: Bot direkt gegenüber (Gegenverkehr)?
                if 100 < center_x <= 350 and area >= 2000:
                    if self.debug:
                        rospy.loginfo("[Rechts vor Links] Großer Bot gegenüber (x=%d, area=%d) – Warten (links)", center_x, area)
                    return False

            # Kein relevanter Bot rechts oder gegenüber → Linksabbiegen erlaubt
            if self.debug:
                rospy.loginfo("[Rechts vor Links] Kein großer Bot rechts oder gegenüber – Linksabbiegen erlaubt")
            return True

        elif self.chosen_direction == "geradeaus":
            for bot in self.detected_bots:
                x1, y1, x2, y2 = bot
                center_x = (x1 + x2) // 2
                area = self.get_box_area(x1, y1, x2, y2)

                if center_x > 350 and area >= 6026:
                    if self.debug:
                        rospy.loginfo("[Rechts vor Links] Großer Bot rechts (x=%d, area=%d) – Warten (geradeaus)", center_x, area)
                    return False

            if self.debug:
                rospy.loginfo("[Rechts vor Links] Kein großer Bot rechts – Geradeaus erlaubt")
            return True

        else:
            # Keine Richtung gewählt → Sicherheitshalber nicht freigeben
            return False





    def blink_all_leds(self, event):
        if not hasattr(self, 'pattern_on'):
            return

        pattern_for_publish = LEDPattern()

        if self.blink_on:
            pattern_for_publish = self.pattern_on
        else:
            # → ALLE 5 LEDs ausschalten, inkl. LED 4 (vorne rechts)
            pattern_for_publish.color_mask = [False] * 5
            pattern_for_publish.frequency = 0.0
            pattern_for_publish.frequency_mask = [False] * 5
            pattern_for_publish.rgb_vals = [ColorRGBA(0, 0, 0, 1.0)] * 5

        self.led_pub.publish(pattern_for_publish)
        self.blink_on = not self.blink_on


    def turn_off_leds(self):
        pattern_default = LEDPattern()

        # LED 0, 1, 3, 4: nutzen; LED 2: ignorieren
        pattern_default.color_mask = [True, True, False, True, True]
        pattern_default.frequency_mask = [False, False, False, False, False]
        pattern_default.frequency = 0.0

        # Mapping nach deiner Beobachtung
        pattern_default.rgb_vals = [
            ColorRGBA(1.0, 1.0, 1.0, 1.0),  # [0] Front links → weiß
            ColorRGBA(1.0, 0.0, 0.0, 1.0),  # [1] Hinten rechts → rot
            ColorRGBA(0.0, 0.0, 0.0, 1.0),  # [2] Ignorieren (kein Effekt)
            ColorRGBA(1.0, 0.0, 0.0, 1.0),  # [3] Hinten links → rot
            ColorRGBA(1.0, 1.0, 1.0, 1.0)   # [4] Vorne rechts → weiß
        ]

        self.led_pub.publish(pattern_default)



    def run(self):
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            if self.current_image is None:
                rate.sleep()
                continue

            if self.blink_timer is not None and not self.abbiegephase_gestartet and not self.waiting_at_line:
                self.blink_timer.shutdown()
                self.blink_timer = None
                self.turn_off_leds()

            frame = self.current_image.copy()
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            rd = self.conf['red']
            lower_red1 = np.array([rd['hl'], rd['sl'], rd['vl']])
            upper_red1 = np.array([rd['hh'], rd['sh'], rd['vh']])
            lower_red2 = np.array([140, rd['sl'], rd['vl']])
            upper_red2 = np.array([255, rd['sh'], rd['vh']])

            mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask_red = cv2.bitwise_or(mask_red1, mask_red2)

            contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            min_area = 150
            filtered_contours_red = [cnt for cnt in contours_red if cv2.contourArea(cnt) > min_area]

            height, width = frame.shape[:2]
            half_width = width // 2
            fifth_width = width // 5
            quarter_height = height // 4
            y_horizontal = 3 * quarter_height

            options = []
            for contour in filtered_contours_red:
                x, y, w, h = cv2.boundingRect(contour)
                center_x = x + w // 2
                center_y = y + h // 2
                if center_y < y_horizontal:
                    if center_x <= fifth_width:
                        options.append("links")
                    elif center_x <= half_width:
                        options.append("geradeaus")
                    else:
                        options.append("rechts")

            current_time = rospy.get_time()

            if self.waiting_at_line:
                self.pub_info.publish(Int32(3))  # Signalisiert: Warte an Stopplinie

                # >>> Richtungswahl NUR hier durchführen <<<
                if options and self.chosen_direction is None:
                    if self.debug_important:
                        rospy.loginfo(f"[Abbiegen] Mögliche Richtungen erkannt (beim Warten): {options}")
                    self.chosen_direction = random.choice(options)
                    if self.debug_important:
                        rospy.loginfo(f"[Abbiegen] Richtung gewählt: {self.chosen_direction}")

                    # LED-Muster setzen nach Wahl
                    if self.chosen_direction == "links":
                        self.pattern_on.color_mask = [True, False, False, True, False]
                        self.pattern_on.frequency_mask = [True, False, False, True, False]
                        self.pattern_on.rgb_vals = [
                            ColorRGBA(1.0, 1.0, 0.0, 1.0),
                            ColorRGBA(0, 0, 0, 1.0),
                            ColorRGBA(0, 0, 0, 1.0),
                            ColorRGBA(1.0, 1.0, 0.0, 1.0),
                            ColorRGBA(0, 0, 0, 1.0)
                        ]

                    elif self.chosen_direction == "rechts":
                        self.pattern_on.color_mask = [False, True, False, False, True]
                        self.pattern_on.frequency_mask = [False, True, False, False, True]
                        self.pattern_on.rgb_vals = [
                            ColorRGBA(0, 0, 0, 1.0),
                            ColorRGBA(1.0, 1.0, 0.0, 1.0),
                            ColorRGBA(0, 0, 0, 1.0),
                            ColorRGBA(0, 0, 0, 1.0),
                            ColorRGBA(1.0, 1.0, 0.0, 1.0)
                        ]

                # Blinker aktivieren (nur wenn Richtung gewählt und Timer noch nicht läuft)
                if self.blink_timer is None and self.chosen_direction in ["links", "rechts"]:
                    self.blink_timer = rospy.Timer(rospy.Duration(0.5), self.blink_all_leds)
                    if self.debug:
                        rospy.loginfo("[Abbiegen] Blinken gestartet beim Warten an Linie")

                # Nach Wartezeit prüfen, ob wir abbiegen dürfen
                if current_time - self.wait_start_time >= 7.0:
                    if (current_time - self.wait_start_time) % 1 < 0.1:
                        if self.check_rechts_vor_links():
                            self.waiting_at_line = False
                            self.abbiegephase_gestartet = True
                            self.abbiege_start_time = current_time
                            if self.debug:
                                rospy.loginfo("[Abbiegen] Abbiegevorgang gestartet")


            if self.abbiegephase_gestartet and not self.abgeschlossen:
                self.pub_info.publish(Int32(4))
                if self.chosen_direction == "links":
                    if self.debug:
                        rospy.loginfo("[Abbiegen] Linksabbiegen aktiv")
                    if filtered_contours_red:
                        leftmost = min(filtered_contours_red, key=lambda cnt: cv2.boundingRect(cnt)[0])
                        x, y, w, h = cv2.boundingRect(leftmost)
                        y_clamped = max(100, min(y, 400))
                        offset = int(np.interp(y_clamped, [100, 300], [250, 0]))
                        target_x = x + w + offset + 150
                        if self.debug_window:
                            cv2.circle(frame, (target_x, y), 6, (255, 0, 255), -1)
                    else:
                        target_x = 200
                        rospy.logwarn_throttle(2, "[Abbiegen] Keine rote Kontur gefunden")
                    self.pub_target_x.publish(Int32(target_x))

                elif self.chosen_direction == "geradeaus":
                    if self.debug:
                        rospy.loginfo_throttle(1, "[Abbiegen] Geradeaus aktiv")

                    if filtered_contours_red:
                        # Nur Konturen mit ausreichend Fläche betrachten
                        valid_contours = [cnt for cnt in filtered_contours_red if cv2.contourArea(cnt) > 200]

                        if valid_contours:
                            # Kontur mit kleinster Unterkante (y + h) wählen
                            def lowest_bottom(cnt):
                                x, y, w, h = cv2.boundingRect(cnt)
                                return y + h  # untere Kante

                            target_cnt = min(valid_contours, key=lowest_bottom)
                            x, y, w, h = cv2.boundingRect(target_cnt)

                            # Unterkante-Linie einzeichnen
                            cv2.line(frame, (x, y + h), (x + w, y + h), (255, 0, 0), 2)
                            cv2.putText(frame, f"y+h={y+h}", (x, y + h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                            y_clamped = max(100, min(y, 400))
                            offset = int(np.interp(y_clamped, [200, 370], [70, 250]))
                            target_x = x + w + offset

                            if self.debug_window:
                                cv2.circle(frame, (target_x, y), 6, (0, 255, 0), -1)

                        else:
                            rospy.logwarn_throttle(2, "[Abbiegen] Keine geeignete Kontur für Geradeaus gefunden")
                            target_x = 320
                    else:
                        rospy.logwarn_throttle(2, "[Abbiegen] Keine Kontur gefunden")
                        target_x = 320

                    self.pub_target_x.publish(Int32(target_x))

                
                elif self.chosen_direction == "rechts":
                    if self.debug:
                        rospy.loginfo("[Abbiegen] Rechtsabbiegen aktiv")

                    if filtered_contours_red:
                        # Rechtseste Kontur finden (größter x-Wert)
                        rightmost = max(filtered_contours_red, key=lambda cnt: cv2.boundingRect(cnt)[0])
                        x, y, w, h = cv2.boundingRect(rightmost)

                        # Nutze rechte untere Ecke der Box
                        corner_x = x + w
                        corner_y = y + h

                        # Offset basierend auf y-Distanz zur Kamera (weiter weg = kleiner)
                        y_clamped = max(100, min(corner_y, 400))
                        offset = int(np.interp(y_clamped, [100, 300], [0, 200]))

                        target_x = corner_x + offset - 200  # Nach links korrigieren für Rechtskurve

                        if self.debug_window:
                            cv2.circle(frame, (target_x, corner_y), 6, (0, 0, 255), -1)
                            cv2.putText(frame, f"Target (rechts)", (target_x - 30, corner_y - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                            cv2.putText(frame, f"Offset: {offset}", (target_x - 30, corner_y + 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    else:
                        target_x = 440  # Fallback weit rechts
                        rospy.logwarn_throttle(2, "[Abbiegen] Keine rote Kontur für Rechtsabbiegen gefunden")

                    self.pub_target_x.publish(Int32(target_x))



                # Richtungsabhängiger Zeitwert
                required_time = 1.0 if self.chosen_direction == "rechts" else 3.0

                if (
                    self.abbiege_start_time is not None and
                    (current_time - self.abbiege_start_time) >= required_time and
                    self.left_x_value is not None
                ):
                    if self.debug:
                        rospy.loginfo("[Abbiegen] Weiße Linie erkannt – Abbiegevorgang abgeschlossen")
                    if self.blink_timer is not None:
                        self.blink_timer.shutdown()
                        self.blink_timer = None
                    self.turn_off_leds()

                    self.abbiegephase_gestartet = False
                    self.abgeschlossen = True
                    self.left_x_value = None
                    self.abbiege_start_time = None
                    self.last_turn_completed_time = current_time
                    self.chosen_direction = None

            # --- NEU: Nur tiefste rote Box überwachen ---
            if not self.abbiegephase_gestartet and not self.waiting_at_line:
                if filtered_contours_red:
                    bottommost = max(filtered_contours_red, key=lambda cnt: cv2.boundingRect(cnt)[1])
                    x, y, w, h = cv2.boundingRect(bottommost)
                    if y > 210:
                        self.pub_info.publish(Int32(1))
                        if self.debug:
                            rospy.loginfo(f"[Info] Unterste rote Box bei y={y} → sende Int32(1)")
                else:
                    if self.debug:
                        rospy.loginfo("[Info] Keine rote Box erkannt → sende Int32(0)")

            if self.debug_window:
                # Kreuzungs-Polygon einzeichnen
                cv2.polylines(frame, [self.intersection_area], isClosed=True, color=(255, 0, 255), thickness=2)

                for cnt in filtered_contours_red:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

                # Optional: Detektierte Bots markieren
                for bot in self.detected_bots:
                    x1, y1, x2, y2 = bot
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, "Bot", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                # Richtung einblenden
                richtung_text = f"Richtung: {self.chosen_direction if self.chosen_direction else 'keine'}"
                cv2.putText(frame, richtung_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

                cv2.imshow("Rote Linien", frame)
                cv2.waitKey(1)

            rate.sleep()


if __name__ == "__main__":
    node = RedLineDetector(node_name="red_line_detector")
    node.run()
