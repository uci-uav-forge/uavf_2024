# import csv
# from io import TextIOWrapper
# from pathlib import Path
# import time
# from psutil import cpu_percent, virtual_memory
# import asyncio

# async def periodic():
#     while True:
#         print('periodic')
#         await asyncio.sleep(1)

# def stop():
#     task.cancel()

# loop = asyncio.get_event_loop()
# task = loop.create_task(periodic())

# try:
#     loop.run_until_complete(task)
# except asyncio.CancelledError:
#     pass

# class SysUtilLogger:
#     """
#     Class to asynchronously log the system utilization
#     """
#     def __init__(self, logs_dir: Path, period_seconds: float):
#         self.logs_dir = logs_dir
#         self._period_seconds = period_seconds
        
#         self._event_loop = asyncio.get_event_loop()
#         self.task: asyncio.Task | None = None
#         self._log_file: TextIOWrapper | None
    
#     def _log_utilization(self):
#         while True:
#             print('logging sysutil')
#             await asyncio.sleep(1)

#     async def _log_preiodically(self):
#         while True:
#             print('logging sysutil')
#             await asyncio.sleep(1)
        
#     def start(self):
#         with open(self.logs_dir / f'{time.strftime("%m-%d %Hh%Mm")}.csv', "w") as f:
#             writer = csv.writer(f)
            
        
#         loop = asyncio.get_event_loop()
#         task = loop.create_task(periodic())
#         try:
#             loop.run_until_complete(task)
#         except asyncio.CancelledError:
#             print("Stopped sysutil logging logging")
#             pass
    