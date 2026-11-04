// serial_cmd.h - line commands from the laptop (115200 baud, newline-terminated):
//   T=YYYY-MM-DD HH:MM:SS     set the DS1307 from the PC clock
//   F=<id>,pm25,pm10,temp,rh  classify one raw feature vector through the live input path; reply
//                             C=<id>,<class>,<conf>,<lat_us>,<p0>..<p5> (probabilities exact, %.8f),
//                             C=<id>,-1 when Invoke() failed, C=<id>,-2 when the values did not parse.
//                             The id lets code/08_on_device/verify_on_device.py reject stale replies.
//   BENCH                     run the latency / RAM / energy protocol now
//   MODEL                     reply M=<model bytes>,<model crc32>,<input constants crc32> (hex)
// While F= vectors keep arriving the sensor cycle pauses (resumes TEST_IDLE_MS after the last one),
// so a verification run is not throttled to one reply per 3 s cycle.
#pragma once
#include <Arduino.h>

void serialService();      // handles every complete line waiting in the RX buffer
bool serialTestActive();   // true while a test-vector run is in progress
