from pymavlink import mavutil

conn = mavutil.mavlink_conn('udpin:localhost:14551')

conn.wait_heartbeat()

message = conn.mav.command_long_encode(
        conn.target_system,  # Target system ID
        conn.target_component,  # Target component ID
        mavutil.mavlink.MAV_CMD_USER_1,  # ID of command to send
        0,  # Confirmation
        0,  # param1
        0,  # param2
        0,  # param3 (unused)
        0,  # param4 (unused)
        0,  # param5 (unused)
        0,  # param5 (unused)
        0   # param6 (unused)
        )

# Send the COMMAND_LONG     
conn.mav.send(message)

# Wait for a response (blocking) to the MAV_CMD_SET_MESSAGE_INTERVAL command and print result
response = conn.recv_match(type='STATUSTEXT', blocking=True)
if response:
    response_txt = response.decode()
    print("response message recieved from drone.")
    print(f"response contents: {response_txt}" )

    
else:
    print("response message not recieved.")

# types = set()
# while True:
#     msg = conn.recv_match(blocking=True)
#     if msg._type not in types:
#         print(msg._type)
#         types.add(msg._type)