# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/Users/benchou/esp/esp-idf/components/bootloader/subproject"
  "/Users/benchou/Desktop/GitHub/CS528-Drone-Gestures/Bluetooth-C/project-name/build/bootloader"
  "/Users/benchou/Desktop/GitHub/CS528-Drone-Gestures/Bluetooth-C/project-name/build/bootloader-prefix"
  "/Users/benchou/Desktop/GitHub/CS528-Drone-Gestures/Bluetooth-C/project-name/build/bootloader-prefix/tmp"
  "/Users/benchou/Desktop/GitHub/CS528-Drone-Gestures/Bluetooth-C/project-name/build/bootloader-prefix/src/bootloader-stamp"
  "/Users/benchou/Desktop/GitHub/CS528-Drone-Gestures/Bluetooth-C/project-name/build/bootloader-prefix/src"
  "/Users/benchou/Desktop/GitHub/CS528-Drone-Gestures/Bluetooth-C/project-name/build/bootloader-prefix/src/bootloader-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/Users/benchou/Desktop/GitHub/CS528-Drone-Gestures/Bluetooth-C/project-name/build/bootloader-prefix/src/bootloader-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/Users/benchou/Desktop/GitHub/CS528-Drone-Gestures/Bluetooth-C/project-name/build/bootloader-prefix/src/bootloader-stamp${cfgdir}") # cfgdir has leading slash
endif()
