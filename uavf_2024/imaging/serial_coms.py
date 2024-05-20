import serial, json, datetime, os

CURRENT_FILE_PATH = os.path.dirname(os.path.realpath(__file__))
#logging file location can be changed according to use
DRONE_LOG_PATH = os.path.join(CURRENT_FILE_PATH, "ESP32_Log")

class Drone_Communication:
    def __init__(self):
        os.makedirs(DRONE_LOG_PATH, exist_ok= True)
        current_date = datetime.datetime.now().date().strftime('%Y-%m-%d')
        self.file_path = os.path.join(DRONE_LOG_PATH, f"ESP32_log_{current_date}.txt")
        with open(self.file_path, 'a'):
            pass

    def drone_logging(self):
        #function that opens, read, store, and close the serial with its information on the drone ID
        current_time = datetime.datetime.now().time()
        current_time = current_time.replace(microsecond=0)

        with serial.Serial("COM6", 57600) as ser:
            drone_entry_tab = ser.readline().decode().strip()
            drone_entry = json.loads(drone_entry_tab)
            drone_entry["time stamp"] = current_time.strftime('%H:%M:%S')
            print(drone_entry)
            #logging in the same file according to date
            with open(self.file_path, 'a') as drone_log:
                drone_log.write(json.dumps(drone_entry) + "\n")

if __name__ == "__main__":
    drone_obj = Drone_Communication()
    drone_obj.drone_logging()
            