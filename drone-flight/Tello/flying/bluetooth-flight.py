from tello import *
import asyncio
import serial
import torch
import pandas as pd
import numpy as np
import asyncio
from bleak import BleakClient, BleakScanner
import time
import torch.nn as nn
import matplotlib.pyplot as plt

# to run this file, have gatts_demo.c running on your esps3

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
class MotionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.0):
        super(MotionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

if torch.cuda.is_available():
    print("cuda available")
    device = torch.device('cuda')
    print("cuda")
else:
    print("CPU only")
    device = torch.device('cpu')
    print("CPU")

model = torch.load("/Users/nicolekaldus/esp/final-project/CS528-Drone-Gestures/models/CNN-bluetooth.pth", map_location="cpu")
model.eval()

def checkForMovement(paddingWindow, numActivationMovements, accelerationThreshold):
  movementCount = 0
  # print("\n\n\n")
  for movement in paddingWindow[-numActivationMovements:]: #check for movement
    isMovement = False
    # print("Movement detected: ", movement)
    for axis in movement: #only checking the acceleration points
      if axis > accelerationThreshold or axis < -accelerationThreshold:
        isMovement = True
        break
    if isMovement:
      movementCount += 1
  if movementCount >= numActivationMovements:
    print('Starting to record your gesture...')
    return True
  return False

def normalize(data):
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    data = (data - min_val) / (max_val - min_val)
    return torch.from_numpy(data)

def decideMovement(data): #0 = up, 1 = down, 2 = left, 3 = right

  # data = interpolated
  data = normalize(data)
  data.to(device)
  with torch.no_grad(): # no gradient update
    yhat = model(data.permute(1,0).unsqueeze(0)) # Inference with reshaped data
    pred = torch.max(yhat, 1) # get max probability from model output


    match pred.indices[0].item(): # get the index of the prediction
        case 0:
            print("Predict: up")
            return 0
        case 1:
            print("Predict: down")
            return 1
        case 2:
            print("Predict: left")
            return 2
        case 3:
            print("Predict: right")
            return 3
  return -1



paddingWindowSize = 100 #keep a sliding window of 10 points as padding
movementSize = 400
paddingWindow = []
recordedMovement = []
recording = False
numActivationMovements = 10
accelerationThreshold = 20
upThreshold = 4
upCount = 0
distCM = 20
numGestures = 0


# start()
# takeoff() 

#connect to the esp
loop = asyncio.get_event_loop()
client = loop.run_until_complete(connect_to_device())

readyToRecord = False
print('ready to run')
while numGestures < 10: #program will run 10 gestures
  loop = asyncio.get_event_loop()
  data = loop.run_until_complete(read_from_connection(client))
  # data = read_from_connection(client)
  if len(data) == 0:
    print('could not read data')
    continue

  if not recording: #need to add to window and check for movement
    if len(paddingWindow) < paddingWindowSize:
      for movement in data:
        paddingWindow.append(movement)
    else:
      paddingWindow = paddingWindow[4:]
      for movement in data:
        paddingWindow.append(movement)
      if not readyToRecord:
        readyToRecord = True
        print('ready to record')
      recording = checkForMovement(paddingWindow, numActivationMovements, accelerationThreshold)
  
  else: #else we are recording a movement
    # print('recording a movement...size: ', len(recordedMovement))
    if len(recordedMovement) < movementSize:
      for movement in data:
        recordedMovement.append(movement)
    else: #time to detect what the movement is
      print('DECIDING THE MOVEMENT')
      movementType = decideMovement(paddingWindow + recordedMovement)
      recordedMovement = []
      paddingWindow = []
      numGestures += 1
    
      recording = False
      readyToRecord = False
      
      if movementType == 0: #up
        if upCount < upThreshold:
          upCount += 1
          # up(distCM)
          print('up')
        else:
          print('too high...spin')
          # clockwise(360)
      
      elif movementType == 1: #down
        if upCount > 1:
          upCount -= 1
          print('down')
          # down(distCM)
        else:
          print('too low...spin')
          # clockwise(360)
          
      elif movementType == 2: #left
        print('left')
        # left(distCM)
      
      elif movementType == 3: #right
        print('right')
        # right(distCM)
        
# land() #safely lands the drone
print('landing')