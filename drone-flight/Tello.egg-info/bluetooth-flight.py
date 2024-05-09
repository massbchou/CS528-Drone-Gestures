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
    print(data)
    data = [float(string) for string in data.decode("utf-8").split(',')]
    data = np.array(data).astype(np.float32)
    return data
  except:
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

#/Users/nicolekaldus/esp/final-project/drone-flight/Tello/tests/LSTM.pth
model = torch.load("/Users/nicolekaldus/esp/final-project/CS528-Drone-Gestures/drone-flight/Tello/tests/CNN-L.pth", map_location="cpu")
model.eval()

def checkForMovement(paddingWindow, numActivationMovements, accelerationThreshold):
  movementCount = 0
  for movement in paddingWindow[:-numActivationMovements]: #check for movement
    isMovement = False
    for axisIndex in range(1,len(movement)):
      if movement[axisIndex] > accelerationThreshold or movement[axisIndex] < -accelerationThreshold:
        isMovement = True
        break
    if isMovement:
      movementCount += 1
  if movementCount >= numActivationMovements:
    return True
  return False

def normalize(data):
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    data = (data - min_val) / (max_val - min_val)
    return torch.from_numpy(data)

def decideMovement(data): #0 = up, 1 = down, 2 = left, 3 = right
  #we would run the neural net here
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



paddingWindowSize = 100 #keep a sliding window of 100 points as padding
movementSize = 400
paddingWindow = []
recordedMovement = []
recording = False
numActivationMovements = 10
accelerationThreshold = 20
upThreshold = 4
upCount = 0
distCM = 20

# start()
# takeoff() 
startTime = int(round(time.time() * 1000)) #current time in millisecondsz

loop = asyncio.get_event_loop()
client = loop.run_until_complete(connect_to_device())

print('ready to run')
while int(round(time.time() * 1000)) < startTime + 60000: #program will run for 5 minutes
  loop = asyncio.get_event_loop()
  data = loop.run_until_complete(read_from_connection(client))
  # data = read_from_connection(client)
  if len(data) == 0:
    print('could not read data')
    continue

  if not recording: #need to add to window and check for movement
    if len(paddingWindow) < paddingWindowSize:
      paddingWindow.append(data)
      print(len(paddingWindow))
    else:
      paddingWindow = paddingWindow[1:]
      paddingWindow.append(data)
      print(len(paddingWindow))
    
    # if len(paddingWindow) > numActivationMovements: #can we start checking for movement
      recording = checkForMovement(paddingWindow, numActivationMovements, accelerationThreshold)
      # print('recording: ', recording)
  
  else: #else we are recording a movement
    print('recording a movement...size: ', len(recordedMovement))
    if len(recordedMovement) < movementSize:
      recordedMovement.append(data)
    else: #time to detect what the movement is
      print('DECIDING THE MOVEMEMNT')
      print('padding window len: ', len(paddingWindow))
      print('recorded movement len: ', len(recordedMovement))
      movementType = decideMovement(paddingWindow + recordedMovement)
      recordedMovement = []
      paddingWindow = []
    
      recording = False
      
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




# start()
# takeoff() 

# initial_tof = get_tof()

# for j in range(4):
#   clockwise(90)
#   for i in range(5):
#     forward(20)
#     result = get_tof()
#     if result < initial_tof - 100: 
#       flip_forward()
#     print(result)

# land() 
