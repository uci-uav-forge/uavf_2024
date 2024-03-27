from pymavlink import mavutil

conn = mavutil.mavlink_connection('udpin:localhost:14551')
print("created connection object")

print("waiting for heartbeat")
conn.wait_heartbeat()
print("detected heartbeat.")

message = conn.mav.command_long_encode(
        0,  # Target system ID
        0,  # Target com
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
print("message sent.")

# Wait for a response (blocking) to the MAV_CMD_SET_MESSAGE_INTERVAL command and print result
response = conn.recv_match(type='STATUSTEXT', blocking=True)
print("waiting for response")
if response:
    print(f"response: {response}")
    # response_txt = response.decode()
    print("response message recieved from drone.")
    # print(f"response contents: {response_txt}" )

    
else:
    print("response message not recieved.")

# types = set()
# while True:
#     msg = conn.recv_match(blocking=True)
#     if msg._type not in types:
#         print(msg._type)
#         types.add(msg._type)