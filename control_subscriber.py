# generate ros2 subscriber for controlling frame kit CNC machine subscribing X, Y, Z coordinates implementing in Arduino
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
class CNCControlSubscriber(Node):
    def __init__(self):
        super().__init__('control_subscriber')
        self.subscription_ = self.create_subscription(
            Float32MultiArray,
            '/cnc_control',
            self.listener_callback,
            10)
        self.get_logger().info('CNC Control Subscriber has been started.')

    def listener_callback(self, msg):
        self.get_logger().info(f'Received CNC control coordinates: {msg.data}')
def main(args=None):
    rclpy.init(args=args)
    control_subscriber = CNCControlSubscriber()
    rclpy.spin(control_subscriber)
    control_subscriber.destroy_node()
    rclpy.shutdown()
if __name__ == '__main__':    
    main()