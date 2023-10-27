from typing import List

class DropzonePlanner:
    # Handles all logic related to controlling drone motion during the payload drop.
    def ___init__(self, commander, dropzone_coords: List[(float, float)], image_width_m, image_height_m):
        self.commander = commander
        self.dropzone_coords = dropzone_coords
        self.image_width_m = image_width_m
        self.image_height_m = image_height_m
    
    def send_release_payload_signal(self):
        print("WOULD RELEASE PAYLOAD")

    def conduct_air_drop(self,
                        payload_color_id: int,
                        payload_shape_id: int):
        # Called when a waypoint lap has been finished.
        # Moves to drop zone from current position,
        # scans drop zone for targets if necessary,
        # navigates to the target matching the current payload,
        # and releases it.

        pass