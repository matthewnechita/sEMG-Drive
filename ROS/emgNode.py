#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
import time


def get_fake_gesture(elapsed_time):
    """
    Returns a fake gesture label based on time.
    Simulating a real-time ML model output.
    """

    if elapsed_time < 5:
        return "FORWARD"
    elif elapsed_time < 10:
        return "LEFT"
    elif elapsed_time < 15:
        return "RIGHT"
    elif elapsed_time < 20:
        return "STOP"
    else:
        return "FORWARD"


def main():
    # Initialize this Python file as a ROS node
    rospy.init_node("emg_fake_node")

    # Create a publisher that sends String messages on /emg/gesture
    pub = rospy.Publisher("/emg/gesture", String, queue_size=10)

    # Control how often we publish (Hz)
    rate = rospy.Rate(10)  # 10 messages per second

    start_time = time.time()

    rospy.loginfo("EMG fake node started. Publishing fake gestures...")

    while not rospy.is_shutdown():
        elapsed_time = time.time() - start_time

        # Reset cycle every 20 seconds
        elapsed_time = elapsed_time % 20

        # Get fake gesture label
        gesture = get_fake_gesture(elapsed_time)

        # Create ROS message
        msg = String()
        msg.data = gesture

        # Publish message
        pub.publish(msg)

        rospy.loginfo(f"Published gesture: {gesture}")

        rate.sleep()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
