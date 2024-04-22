# CS528-Drone-Gestures
Semester Project for CS 528 at the University of Massachusetts Amherst

# Virtual Environment Activation
'source esptoolenv/bin/activate' for MacOS / Linux
'esptoolenv\Scripts\activate' for Windows

# Setting Up Python on Your Device
Instructions adapted from https://micropython.org/download/ESP32_GENERIC_S3/
1. Confirm your port (mine is /dev/cu.usbserial-10)
2. Confirm your connection works with 'esptool.py -p PORT flash_id' to check the flash info.
3. If this is the FIRST time installing python, clear the flash. 'esptool.py --chip esp32s3 --port PORT erase_flash' 
4. Program the firmware starting address 0 using 'esptool.py --chip esp32s3 --port PORT write_flash -z 0 FIRMWARE.bin'