from typing import List, Tuple

class DropzonePlanner:
    # Handles all logic related to controlling drone motion during the payload drop.
    def __init__(self, commander, dropzone_coords: List[Tuple[float, float]], image_width_m: float, image_height_m: float):
        self.commander = commander
        self.dropzone_coords = dropzone_coords
        self.image_width_m = image_width_m
        self.image_height_m = image_height_m
    
    def send_release_payload_signal(self):
        # mocked out for now.
        print("WOULD RELEASE PAYLOAD")

    def conduct_air_drop(self,
                        payload_color_id: int,
                        payload_shape_id: int,
                        payload_letter_id: int):
        # Called when a waypoint lap has been finished.
        # Expects that the drone has taken off and is in GUIDED mode.
        # Moves to drop zone from current position,
        # scans drop zone for targets if it's the first time,
        # navigates to the target best matching the current payload,
        # and releases it.

        pass