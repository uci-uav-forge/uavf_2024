"""
@file test_follow_mode.py
@Description: This is a test script for using the ZR10 SDK Python implementation to set follow mode
@Author: Mohamed Abdelkader
@Contact: mohamedashraf123@gmail.com
All rights reserved 2022
"""

import sys
import os
from time import sleep
  
from siyi_sdk import SIYISDK

def test():
    cam = SIYISDK(server_ip="192.168.144.25", port=37260)

    if not cam.connect():
        print("No connection ")
        exit(1)

    cam.requestFollowMode()
    sleep(2)
    print("Current motion mode: ", cam._motionMode_msg.mode)

    cam.disconnect()

if __name__ == "__main__":
    test()