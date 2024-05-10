import asyncio
import time
import numpy as np
import serial
import asyncio
from bleak import BleakClient, BleakScanner

async def connect_to_device():
    try:
        scanner = BleakScanner()
        devices = await scanner.discover()
        for device in devices:
            try:
                if device.name == "THE ESP32":
                    print(f"Connecting to device: {device.name}")
                    client = BleakClient(device)
                    await client.connect()
                    return client
            except ValueError as e:
                print(f"An error occurred while processing a device: {e}")
                # return None
    except ValueError as e:
        print(f"An error occurred: {e}")
        
async def read_from_connection(client):
  try:
    data = await client.read_gatt_char("0000ff01-0000-1000-8000-00805f9b34fb")
    # print(data)
    data = data.decode("utf-8").split('\n') #split the data based on the newline character
    data = data[:-1]
    finalData = []
    for movement in data:
      movement = ([float(string) for string in movement.split(',')])
      movement = np.array(movement).astype(np.float32)
      # print(movement, ' ', len(movement))
      finalData.append(movement)
    return finalData
  except Exception as e:
    print(e)
    return []
   #writing a random comment 
   
movementLen = 500
recordingsPerGesture = 62
gestureTypes = ['left']#, 'down', 'left', 'right']
recordedMovements = [] #3d array ==> [gesture type][gesture reading][single movement reading]

loop = asyncio.get_event_loop()
client = loop.run_until_complete(connect_to_device())
print('ready to run')
for gesture in gestureTypes:
    print('now recording movement for ', gesture)
    gestureGroupList = []
    time.sleep(3)
    for i in range(recordingsPerGesture):
        print('Starting reading ', i+1 , ' out of ', recordingsPerGesture)
        time.sleep(3)
        currGestureData = []
        
        while len(currGestureData) < movementLen:
            loop = asyncio.get_event_loop()
            data = loop.run_until_complete(read_from_connection(client))
            if len(data) == 0:
                continue
            for movement in data:
                print(movement)
                currGestureData.append(movement)
            
        #done with that particular file
        gestureGroupList.append(currGestureData)
    recordedMovements.append(gestureGroupList)
    
print('all readings done!')
print('\n\n\n', recordedMovements[0][0][0][0])
for i in range(len(recordedMovements)):
    gestureType = gestureTypes[i]
    for j in range(len(recordedMovements[i])):
        # /Users/nicolekaldus/esp/final-project/CS528-Drone-Gestures/bluetooth-data/down_0.csv
        storageFile = '/Users/nicolekaldus/esp/final-project/CS528-Drone-Gestures/bluetooth-data/' + gestureType + '_' + str(j) + '.csv'
        myFile = open(storageFile, 'w')
        myFile.write('acce_x,acce_y,acce_z,gyro_x,gyro_y,gyro_z\n')
        
        for measurement in recordedMovements[i][j]:
            resultingCSVString = ""
            for axis in measurement:
                resultingCSVString += str(axis) + ','
                
            resultingCSVString = resultingCSVString[:-1]
            resultingCSVString += '\n'
            myFile.write(resultingCSVString)
            
        myFile.close()

print('creating files done!')