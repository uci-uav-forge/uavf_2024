from time import sleep
from siyi_sdk import SIYISDK, SIYISTREAM
import cv2

def test():
    stream = SIYISTREAM()

    if not stream.connect():
        print("No stream connection")
        exit(1)

    sleep(1)
    while 1:
        frame = stream.get_frame()
        cv2.putText(frame, "Resolution: {}".format(frame.shape), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        elif key & 0xFF == ord(' '):
            cv2.imwrite("frame.jpg", frame)
            
    stream.disconnect()

if __name__ == "__main__":
    test()