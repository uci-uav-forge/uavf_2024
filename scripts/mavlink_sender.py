#!/usr/bin/env python3

import asyncio
from mavsdk import System
from mavsdk.server_utility import StatusTextType
import aiostream
import aioconsole
from uavf_2024.gnc.mission_messages import *

got_connection = False

async def listen():
    while not got_connection:
        await asyncio.sleep(0.1)
    while True:
        cmd = await aioconsole.ainput("uavf> ")
        await drone.server_utility.send_status_text(StatusTextType.NOTICE, BumpLap(int(cmd)).to_string())
            

async def send():
    global drone, got_connection
    drone = System(sysid=1)
    print("Awaiting connection")
    await drone.connect(system_address="udp://:14445") # todo cli arg :V
    print("Awaiting state update")
    
    async for state in drone.core.connection_state():
        if state.is_connected:
            print(f"Connected to drone!")
            got_connection = True
            break
            
    async for msg in drone.telemetry.status_text():
        if is_uavf_message(msg.text):
          print(msg.text + "\nuavf> ",end='')
    

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(asyncio.wait([listen(), send()]))