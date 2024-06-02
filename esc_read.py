import serial
import base64
import time
from time import strftime
import os

# on the jetson orin, you plug the esc telem wire into pin 10,
# which is fifth from the USB ports on the side closer to the fan.
'''
Where the X is in this diagram:

            X o
            o o
fan side    o o   edge of board
            o o
            o o
       usb ports side
'''

ser = serial.Serial(
        '/dev/ttyTHS1', 
        baudrate = 115200, 
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE
)

buff = []

def update_crc8(crc, crc_seed):
    crc_u = 0
    crc_u = crc
    crc_u ^= crc_seed
    for i in range(8):
        crc_u = 0x7 ^ (crc_u<<1) if (crc_u & 0x80) else (crc_u<<1)
        crc_u = crc_u & 0xFF
    return crc_u

def crc8(buff: list) -> bool:
    '''Calculate crc8 of buffer'''
    crc = 0
    for i in range(len(buff)):
        crc = update_crc8(buff[i], crc)
    return crc

first_byte = True
first_buff = True
first_valid_buff = True

os.makedirs("/home/forge/logs_esc", exist_ok=True)
fname = f"/home/forge/logs_esc/{strftime('%H:%M:%S')}.txt"

print(f"Logging to {fname}")
while 1:
    buff.append(ord(ser.read()))
    if first_byte:
        first_byte = False
        print("Got first byte")
    if len(buff) == 10:
        if first_buff:
            first_buff = False
            print("Got first full buffer")
        if crc8(buff[:-1]) == buff[-1]:
            if first_valid_buff:
                first_valid_buff = False
                print("Got first valid buffer", buff)
            temp = buff[0]
            voltage = (buff[1]<<8) + buff[2]
            current = (buff[3]<<8) + buff[4]
            consumption = (buff[5]<<8) + buff[6]
            rpm = (buff[7]<<8) + buff[8]

            t = int(time.time()*1e6)
            t_bytes = t.to_bytes((t.bit_length()+7)//8, byteorder='big')
            t_str = base64.b64encode(t_bytes).decode()

            with open(fname, "a") as f:
                f.write(f'{t_str} {temp} {voltage} {current} {consumption} {rpm}\n')

            buff = []
        else:
            del buff[0]
