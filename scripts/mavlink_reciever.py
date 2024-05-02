#!/usr/bin/env python3

import asyncio
from mavsdk import System
from mavsdk.server_utility import StatusTextType

async def run():

    drone = System(sysid=1)
    await drone.connect(system_address="udp://:14540")

    print("Waiting for drone to connect...")
    async for msg in drone.telemetry.status_text():
        print(str(msg.type) + ": " + msg.text)

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run())



