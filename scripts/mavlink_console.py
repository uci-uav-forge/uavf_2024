#!/usr/bin/env python3

import asyncio
from mavsdk import System
from mavsdk.server_utility import StatusTextType
import aiostream
import aioconsole
from uavf_2024.gnc.mission_messages import *

async def listen():
    drone = System(sysid=1)
    print("Awaiting connection")
    await drone.connect(system_address="udp://:14445") # todo cli arg :V
    print("Awaiting state update")
    
    async for state in drone.core.connection_state():
        if state.is_connected:
            print(f"Connected to drone!")
            break
    
    cur_msg = ""
    async for msg in drone.telemetry.status_text():
        if msg.type.value == StatusTextType.NOTICE.value and (cur_msg != "" or is_uavf_message(msg.text)):
          cur_msg += msg.text
          if '}' in cur_msg:
              print("recieved message ", cur_msg)
              cur_msg = ""
    

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(asyncio.wait([listen()]))