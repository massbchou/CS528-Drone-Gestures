import torch
import pandas as pd
import numpy as np
import random as rand

if torch.cuda.is_available():
    device = torch.device('cuda')
    print("cuda")
else:
    device = torch.device('cpu')
    print("CPU")


def random_data():
    num = rand.randint(0, 61)
    labels = ["up", "down", "left", "right"]
    true_label = rand.choice(labels)
    return num, true_label

def normalize(data):
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    data = (data - min_val) / (max_val - min_val)
    return torch.from_numpy(data)


if __name__ == "__main__":

    num, true_label = random_data()
    df = pd.read_csv(f"data/{true_label}_{num:02d}.csv")

    data = df[['acce_x', 'acce_y', 'acce_z', 'gyro_x', 'gyro_y', 'gyro_z']].values.astype(np.float32)

    data = normalize(data)
    model = torch.load("models/CNN.pth", map_location="cpu") # assuming no cuda
    model.eval() # evaluation mode for inference -> no backward pass
    data.to(device) # if cuda

    with torch.no_grad(): # no gradient update
        yhat = model(data.permute(1,0).unsqueeze(0)) # Inference with reshaped data
        pred = torch.max(yhat, 1) # get max probability from model output

        match pred.indices[0].item(): # get the index of the prediction
            case 0:
                print("Predict: up")
            case 1:
                print("Predict: down")
            case 2:
                print("Predict: left")
            case 3:
                print("Predict: right")
        print(f"Actual: {true_label}_{num:02d}")