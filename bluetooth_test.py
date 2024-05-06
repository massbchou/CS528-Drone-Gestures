import asyncio
from bleak import BleakClient, BleakScanner
import struct

# async def run():
#     try:
#         scanner = BleakScanner()
#         devices = await scanner.discover()
#         for device in devices:
#             try:
#                 if device.name:
#                     print(device.name)
#             except ValueError as e:
#                 print(f"An error occurred while processing a device: {e}")
#     except ValueError as e:
#         print(f"An error occurred: {e}")

# loop = asyncio.get_event_loop()
# loop.run_until_complete(run())

async def connect_to_device():
    try:
        scanner = BleakScanner()
        devices = await scanner.discover()
        for device in devices:
            try:
                try:
                    if device.name:
                        print("found " + device.name)
                except ValueError as e:
                    print(f"An error occurred while processing a device: {e}")
                if device.name == "THE ESP32":
                    print(f"Connecting to device: {device.name}")
                    client = BleakClient(device)
                    await client.connect()
                    print("Connected successfully!")
                    # Perform operations with the connected device here
                    # Receive data from the connected device
                    services = client.services
                    for service in services:
                        print("Service:", service)
                        for characteristic in service.characteristics:
                            print("Characteristic:", characteristic)
                    
                    #print("First UUID:", first_uuid)
                    c = 0
                    while (c < 10):
                        data = await client.read_gatt_char("0000ff01-0000-1000-8000-00805f9b34fb") #Hardcoded UUID
                        # data = await client.start_notify("0000ff01-0000-1000-8000-00805f9b34fb", lambda c, x: print("Received data:", x.decode("utf-8")))
                        # while(True):
                        #     await asyncio.sleep(1)
                        print("Recieved ByteArray:", data.decode("utf-8"))
                        # data = struct.unpack('<f', data)
                        print("Received data:", data)
                        c += 1
                    # disconnect after 5 seconds
                    #await asyncio.sleep(5)
                        
                    #await client.disconnect()
                    print("Disconnected from device")
            except ValueError as e:
                print(f"An error occurred while processing a device: {e}")
    except ValueError as e:
        print(f"An error occurred: {e}")

loop = asyncio.get_event_loop()
loop.run_until_complete(connect_to_device())

# from adafruit_ble import BLERadio

# radio = BLERadio()
# print("scanning")
# found = set()
# for entry in radio.start_scan(timeout=60, minimum_rssi=-80):
#     addr = entry.address
#     if addr not in found:
#         if entry.complete_name is not None:
#             print(entry.complete_name)
#     found.add(addr)

# print("scan done")


# Code to be run on the esp32s3
# import time
# import busio
# import board
# import adafruit_mpu6050

# i2c = busio.I2C(scl=board.IO1, sda=board.IO0)

# mpu = adafruit_mpu6050.MPU6050(i2c)
# data = []

# now = time.time()
# start = time.time()
# while now < start + 4:
#     data.append((mpu.acceleration, mpu.gyro))
#     now = time.time()
#     time.sleep(0.04)

# print(len(data))


# from adafruit_ble import BLERadio
# from adafruit_ble.services.standard import device_info
# from adafruit_ble.advertising import Advertisement
# from adafruit_ble.advertising.standard import ProvideServicesAdvertisement
# from adafruit_ble.services.nordic import UARTService

# ble = BLERadio()
# uart = UARTService()
# ble.name = "THE ESP32"
# #advertisement = Advertisement()
# advertisement = ProvideServicesAdvertisement(uart)
# advertisement.short_name = "THE ESP32"
# advertisement.complete_name = "THE ESP32"
# advertisement.connectable = True

# startupAttempts = 0

# while True:
#     is_advertising = ble.advertising
#     print("Advertising: ", is_advertising)
#     if startupAttempts > 5:
#         print("Too many startup attempts, exiting")
#         break
    
#     try:
#         print("Advertisment object: " + str(advertisement))
#         ble.start_advertising(advertisement, scan_response=b"", timeout=100000)
#     except Exception as e:
#         print("error " + str(e))
#         ble.stop_advertising()
#         time.sleep(1)
#         startupAttempts += 1
#         continue
#     print("advertising as " + advertisement.complete_name)
#     while not ble.connected:
#         print("Waiting for connection... " + str(int(time.time() - start)) + " seconds elapsed")
#         time.sleep(1)
#     print("connected")
#     while ble.connected:
#         print("Still connected + " + str(int(time.time() - start)) + " seconds elapsed")
#         for connection in ble.connections:
#             print("connection: " + str(connection))
            
#             uart.write("Hello, world!\r\n")
#         time.sleep(1)
#         pass
#     print("disconnected")
#     ble.stop_advertising()
#     time.sleep(2)