from time import sleep
from siyi_sdk import SIYISDK

def test():
    cam = SIYISDK(server_ip="192.168.144.25", port=37260, debug=True)

    if not cam.connect():
        print("No connection ")
        exit(1)

    sleep(2)
    cam.requestCenterGimbal()
    sleep(5)
    cam.requestAbsolutePosition(135, -90)
    sleep(3)
    cam.requestAbsolutePosition(0, 25)
    sleep(3)
    cam.requestAbsolutePosition(-135, -90)
    sleep(3)
    cam.requestAbsolutePosition(0, -90)
    sleep(3)
    cam.requestAbsolutePosition(0, 0)
    sleep(3)
    cam.disconnect()

if __name__ == "__main__":
    test()