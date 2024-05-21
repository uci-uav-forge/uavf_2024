import sys
import os
from time import sleep, time
from siyi_sdk import SIYISDK

def test():
    cam = SIYISDK(server_ip="192.168.144.25", port=37260, debug=False)

    if not cam.connect():
        print("No connection ")
        exit(1)

    sleep(1)
    cam.requestCenterGimbal()
    sleep(2)
    cam.setAbsoluteZoom(1)
    sleep(3)
    cam.setAbsoluteZoom(5)

    cam.disconnect()

if __name__ == "__main__":
    test()