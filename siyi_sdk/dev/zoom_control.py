import sys
import os
from time import sleep
from siyi_sdk import SIYISDK

def test():
    cam = SIYISDK(server_ip="192.168.144.25", port=37260)

    if not cam.connect():
        print("No connection ")
        exit(1)

    state = True
    while state:
        val = input("w to zoom, s to zoom out, q to exit: ")
        if val == "q":
            state = False
        elif val == "w":
            val = cam.requestZoomIn()
            sleep(1)
            val = cam.requestZoomHold()
            sleep(1)
            print("Zoom level: ", cam.getZoomLevel())
        elif val == "s":
            val = cam.requestZoomOut()
            sleep(1)
            val = cam.requestZoomHold()
            sleep(1)
            print("Zoom level: ", cam.getZoomLevel())
        cam.requestAutoFocus()
        print("Autofocused")

    cam.disconnect()

if __name__ == "__main__":
    test()