#!/usr/bin/env python3

import os
import rospy
from std_msgs.msg import String, ColorRGBA
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import Twist2DStamped, LEDPattern
import time

class RemoteControlNode(DTROS):

    def __init__(self, node_name):
        # initialize the DTROS parent class
        super(RemoteControlNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)

        # static parameters        
        vehicle_name = os.environ['VEHICLE_NAME']
        twist_topic = f"/{vehicle_name}/car_cmd_switch_node/cmd"
        self.pub_cmd_vel = rospy.Publisher(twist_topic, Twist2DStamped, queue_size = 1)
        led_topic = f"/{vehicle_name}/led_emitter_node/led_pattern"
        self.pub_led = rospy.Publisher(twist_topic, LEDPattern, queue_size = 1)
         
    def update_robot_spped(self,v,a):
        twist = Twist2DStamped(v=v, omega=a)
        print(f'moving v: {v}, a: {a}')
        self.pub_cmd_vel.publish(twist)
        time.sleep(0.1)

    def run(self):
        v = 0.2
        a = 0
        c = ''
        

        print('press w,a,s,d for moving the Robot. q to end.')


        # Ändere den Code zwischen den 
        # Das ziel ist es, in einer while Schleife auszulesen, welche Taste gedrückt wird
        # Wenn die Gedrückte Taste w ist, so soll die Geschwindigkeit v um 0.1 erhöht werden bei x um 0.1 reduziert
        # Ist die Taste a soll die Drehung a um 0.3 erhöht werden bei d um 0.3 veringert
        # Ist di Taste s sollen v und a auf 0 gesetzt werden
        # Nach jedem Tastendruck soll die Geschwindigkeit des Roboters aktualisiert werden
        # Wenn q gedrückt wird soll die schleife beendet werden
        #-----------------------------------

        #Lese welche Taste gedrückt wurde
        x = 0
        y = 0
        g = 1
        while c != "q": 
            c = getch()
            if isinstance(c, int):
                g = c
            while c != "q": 
                c = getch()
                if isinstance(c, int):
                    g = c
                if c == "w":
                    x = 0.2*g
                if c == "s":
                    x = -0.2*g
                if c == "a":
                    y = 3
                if c == "d":
                    y = -3
                if c == "e":
                    x = 0
                    y = 0
                print(g)
                self.update_robot_spped(v=x, a=y)
                y = 0
            
            
            
            else:
                print("Geben Sie einen Gang an")
            print(c)

        time.sleep(1)
        #------------------------------------
        # Am Ende Programm Beenden und Bewegung anhalten
        self.update_robot_spped(v=0,a=0)
        rospy.signal_shutdown('User endet Programm')


        


def getch():
    import sys, termios, tty

    fd = sys.stdin.fileno()
    orig = termios.tcgetattr(fd)

    try:
        tty.setcbreak(fd)  
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSAFLUSH, orig)

if __name__ == '__main__':
    # create the node
    node = RemoteControlNode(node_name='remote_control_node')
    # run node
    node.run()
    # keep the process from terminating
    rospy.spin()