// sd_store.h - microSD CSV store that survives a card or 5 V rail dropout.
// Bench 2026-11-02: a 5-minute mains cut took the card offline and the logger never remounted it,
// so 2.5 h of rows were lost. Every row now waits in a RAM ring buffer until the card has confirmed
// it (fflush + fsync succeeded and the file grew by exactly those bytes); the card is remounted at
// most every 10 s while rows are waiting. The SD driver in ../lib/SD is patched so that a block the
// card rejects is reported as a failure instead of as written (see ../lib/README.md).
#pragma once
#include <Arduino.h>

extern bool sdOk;   // card mounted AND every row so far is confirmed on it (no backlog)

// Mount the card (retrying forever with a hint on LCD and serial; boot only) and pick the file:
// path itself if it is new or already has this header, else the first free "<stem>_N.csv".
// whileWaiting (may be null) is called every few ms during the retries, e.g. to answer serial.
void sdMountBlocking(const char* path, const char* csvHeader, void (*whileWaiting)());
// Queue one CSV row (with its newline) and write the queue to the card, or try to remount.
void sdStore(const char* row);
// Call every cycle: writes a waiting backlog (or retries the card) even when no new row was made.
void sdService();
size_t sdPendingCount();    // rows waiting in RAM
size_t sdDroppedCount();    // rows lost because the RAM buffer overflowed (since boot)
const char* sdFileName();   // the file actually in use
