import cv2
import numpy as np
import random
import rospy
from duckietown.dtros import DTROS, NodeType
from std_msgs.msg import String  # Verwenden einer simplen Message für die Ausgabe
from std_msgs.msg import Bool 
 


def process_image():
    #if msg ==True:
        rospy.init_node("red_line_detector", anonymous=True)
    
        # Publisher für die Ergebnisse
        pub = rospy.Publisher(f"/{vehicle_name}/red_line_info", String, queue_size=10)
        sub_redline = rospy.Subscriber(f"/{vehicle_name}/stop_line_detected", Bool, process_image)
        
    
    #TODO Andere Kamera-Zugriffe raussuchen
        # Kamera-Stream öffnen
        cap = cv2.VideoCapture(0)  # Ändere die Quelle je nach Kamerainput
    
        while not rospy.is_shutdown():
            ret, frame = cap.read()
            if not ret:
                break
    
            # Bildgröße bestimmen
            height, width, _ = frame.shape
            half_height, half_width = height // 2, width // 2
    
            # Bild in HSV umwandeln
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
            # Rote Farbe im HSV-Bereich definieren
            lower_red1 = np.array([0, 120, 70])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 120, 70])
            upper_red2 = np.array([180, 255, 255])
    
            # Maske für rote Farbe erstellen
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask = mask1 + mask2
    
            # Konturen finden
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
            # Anzahl der erkannten roten Linien berechnen
            num_red_lines = len(contours)
            num_path_options = max(0, num_red_lines - 1)
    
            # Variablen für Richtungsentscheidungen initialisieren
            turn_straight = False
            turn_right = False
            turn_left = False
    
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
    
                if x < half_width and y < half_height:
                    turn_straight = True
                elif x >= half_width and y < half_height:
                    turn_right = True
                elif x < half_width and y >= half_height:
                    turn_left = True
    
            chosen_direction = None
            if num_path_options == 3:
                chosen_direction = random.choice(["left", "straight", "right"])
                rospy.loginfo(f"Drei Optionen gefunden: Zufällig ausgewählt -> {chosen_direction}")
            elif num_path_options == 2:
                rospy.loginfo("Kreuzung mit zwei Optionen erkannt.")
                if (turn_left and  turn_straight == True):
                    chosen_direction = random.choice(["left","straight"])
                elif (turn_left and turn_right ==True):
                    chosen_direction = random.choice(["left","right"])
                elif (turn_straight and turn_right==True):
                    chosen_direction = random.choice(["straight","right"])
            elif num_path_options == 1:
                rospy.loginfo("Nur eine Möglichkeit vorhanden.")
                if turn_left ==True:
                    chosen_direction = "left"
                elif turn_straight == True:
                    chosen_direction = "straight"
                elif turn_right ==True:
                    chosen_direction = "right" 
    
            #message = f"Linien: {num_red_lines}, Optionen: {num_path_options}, Turn Straight: {turn_straight}, Turn Right: {turn_right}, Turn Left: {turn_left}, Chosen: {chosen_direction}"
            message = f"{chosen_direction}"
            pub.publish(message)  # Nachricht senden
    
            cv2.imshow("Original", frame)
            cv2.imshow("Maske", mask)
    
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
        cap.release()
        cv2.destroyAllWindows()

 
if __name__ == "__main__":
    try:
        process_image()
    except rospy.ROSInterruptException:
        pass