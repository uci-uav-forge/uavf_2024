import numpy as np

class MockSIYISTREAM:
    def __init__(self, server_ip : str = "192.168.144.25", port : int = 8554, name : str = "main.264", debug : bool = False):
        """
        
        Params
        --
        - server_ip [str] IP address of the camera
        - port: [int] UDP port of the camera
        - name: [str] name of the stream
        """
        pass

    def connect(self):
        return True

    def disconnect(self):
        return

    def get_frame(self) -> np.ndarray | None:
        return np.random.randint(0,255, (1080, 1920, 3)).astype(np.float32)
    
class MockSIYISDK:
    def __init__(self, server_ip="192.168.144.25", port=37260, debug=False):
        pass

    def resetVars(self):
        return True

    def connect(self, maxWaitTime=3.0):
        return True

    def disconnect(self):
        pass

    def checkConnection(self):
        pass

    def connectionLoop(self, t):
        pass

    def isConnected(self):
        return True

    def requestFirmwareVersion(self):
        return True

    def requestHardwareID(self):
        return True

    def requestGimbalAttitude(self):
        return True

    def requestGimbalInfo(self):
        return True

    def requestFunctionFeedback(self):
        return True

    def requestAutoFocus(self):
        return True

    def requestAbsoluteZoom(self, level, msg=None):
        return True

    def requestAbsolutePosition(self, yaw, pitch):
        return True

    def requestZoomIn(self):
        return True

    def requestZoomOut(self):
        return True

    def requestZoomHold(self):
        return True

    def requestLongFocus(self):
        return True

    def requestCloseFocus(self):
        return True

    def requestFocusHold(self):
        return True

    def requestCenterGimbal(self):
        return True

    def requestGimbalSpeed(self, yaw_speed:int, pitch_speed:int):
        return True

    def requestPhoto(self):
        return True

    def requestRecording(self):
        return True

    def requestFPVMode(self):
        return True

    def requestLockMode(self):
        return True

    def requestFollowMode(self):
        return True
    def getAttitude(self):
        return (0,0,0)

    def getAttitudeSpeed(self):
        return (0,0,0)

    def getFirmwareVersion(self):
        return (0,0,0)

    def getHardwareID(self):
        return 0

    def getRecordingState(self):
        return 0

    def getMotionMode(self):
        return 0

    def getMountingDirection(self):
        return 0

    def getFunctionFeedback(self):
        return 0

    def getZoomLevel(self):
        return 1

    def setAbsoluteZoom(self, level : float) -> bool:
        return True