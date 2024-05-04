Steps:

* Turn on mavlink forwarding in QGC (Under `Application Settings > MAVLink`)

* Set MAV_{0,1,2}_FORWARD in PX4.
 (Edit /PX4-Autopilot/build/px4_sitl_default/etc/init.d-posix/px4-rc.params )

* Run mavlink_sender.py to send statustext on GCS.

* `commander_node.py` uses MAVROS to send/recieve statustext.

* TODO: wire this all up into functionality!
