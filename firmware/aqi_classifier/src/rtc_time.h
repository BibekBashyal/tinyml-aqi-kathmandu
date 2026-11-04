// rtc_time.h - DS1307 timestamps that never put a bad value into the CSV.
// Every read is a raw register read with a STOP between pointer write and read (the library's
// repeated-start read failed on the bench) and is range-checked; otherwise the row carries an
// uptime_<s> stamp that a PC-timestamped serial capture can align later.
#pragma once
#include <Arduino.h>

extern bool rtcOk;   // DS1307 answered at the last check

void rtcBegin();                           // detect at boot, run the module diagnostic
void rtcService();                         // re-detect every cycle after a flaky bus
bool rtcTimestamp(char* ts, size_t n);     // "YYYY-MM-DD HH:MM:SS" (true) or "uptime_<s>" (false)
void rtcSetFromSerial(const char* line);   // "T=YYYY-MM-DD HH:MM:SS" from the laptop
// After a failed rtcTimestamp(): true when the chip answered with a lost/halted time (T= fixes it),
// false when the read itself failed (bus problem, T= would not help). rtcLastError() says which.
bool rtcTimeLost();
const char* rtcLastError();
