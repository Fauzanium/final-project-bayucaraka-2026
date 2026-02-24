# generate ros2 publisher for controlling frame kit CNC machine publishing X, Y, Z coordinates
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import serial
arduino = serial.Serial('/dev/ttyACM0', 9600)  # Adjust the port and baud rate as needed

"""
Data:
    Coordinates in format [X, Y, Z]
"""

class ControlPublisher(Node):
    def __init__(self):
        super().__init__('control_publisher')
        self.publisher_ = self.create_publisher(Float32MultiArray, '/cnc_control', 10)
        timer_period = 1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.get_logger().info('Trolley Control Publisher has been started.')

    def timer_callback(self):
        msg = Float32MultiArray()
        # Example coordinates for X, Y, Z
        msg.data = [1.0, 2.0, 3.0]  # Replace with actual coordinates
        self.publisher_.publish(msg)
        self.get_logger().info(f'coordinates: {msg.data}')
def main(args=None):
    rclpy.init(args=args)
    control_publisher = ControlPublisher()
    rclpy.spin(control_publisher)
    control_publisher.destroy_node()
    rclpy.shutdown()
if __name__ == '__main__':  
    main()
    