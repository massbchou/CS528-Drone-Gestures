import time
import busio
import board
import adafruit_mpu6050

i2c = busio.I2C(scl=board.IO1, sda=board.IO0)

mpu = adafruit_mpu6050.MPU6050(i2c)
data = []

now = time.time()
start = time.time()
while now < start + 4:
    data.append((mpu.acceleration, mpu.gyro))
    now = time.time()
    time.sleep(0.04)

print(len(data))
# print(data)