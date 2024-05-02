Steps:

* Turn on mavlink forwarding in QGC (Under `Application Settings > MAVLink`)

* Set MAV_{0,1,2}_FORWARD in PX4.
 (Edit /PX4-Autopilot/build/px4_sitl_default/etc/init.d-posix/px4-rc.params )

* Run mavlink_reciever.py to listen for statustext on a comm link.

* Run mavlink_sender.py to listen for statustext on a comm link.

* Todo: get this to play nicely with MAVROS, and test in a non-SITL environment -- currently this breaks MAVROS while we're running this.
* There should be some sort of mavlink splitter we can use to allow both this script and MAVROS to connect.