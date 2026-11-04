#include "sd_store.h"
#include <SPI.h>
#include <SD.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include "config.h"
#include "node_hw.h"

// Code review 2026-11-03 (rounds 2 and 3):
//  - appends go through POSIX stdio on the VFS path, because fflush()/fsync() return the errors that
//    the Arduino File::flush() throws away. A batch is confirmed only if both succeed and stat() shows
//    the file grew by exactly the bytes written; otherwise every row stays queued (a later retry may
//    duplicate rows, which carry their timestamp; it never loses them).
//  - the log is never opened with "w": nothing can truncate it. The header goes into an empty file
//    only, also when the file vanished while running (card swapped).
//  - after a remount the file's last byte tells whether a half-written row needs a newline first.
//  - (final review) the log stays open for the whole mount: reopening it every cycle made FatFs walk
//    the whole cluster chain on each append, slower every day. Any size change we did not make (card
//    swapped) sends the write through a full remount, which re-checks the header.
static const char*    SD_MOUNT_POINT   = "/sd";   // SD.begin() default
static const uint32_t SD_REMOUNT_MS    = 10000;
static const uint32_t SD_RETRY_MS      = 3000;
static const uint32_t SD_WAIT_POLL_MS  = 5;
static const int      SD_MAX_VARIANT   = 99;
static const size_t   SD_PATH_MAX      = 32;
static const size_t   SD_FLUSH_BATCH   = 32;     // rows per confirmed write
static const int      SD_READ_ATTEMPTS = 3;

static char     sdPending[SD_PENDING_ROWS][SD_ROW_BYTES];
static size_t   sdHead = 0, sdCount = 0, sdDropped = 0;
static uint32_t sdLastRemountMs = 0;
static bool     sdNeedNewline = false;   // the file ends inside a row
static const char* sdHeader = "";
static char     sdPath[SD_PATH_MAX] = "";
static char     sdFull[SD_PATH_MAX + 4] = "";   // SD_MOUNT_POINT + sdPath, for stdio
static const char* sdBasePath = "";              // the file asked for; variants are chosen from it
static FILE*    sdFp = nullptr;                  // open for the whole mount
static size_t   sdKnownSize = 0;                 // file size after our last confirmed write
bool sdOk = false;

static bool fileSize(size_t& size) {
  struct stat st;
  if (stat(sdFull, &st) != 0) return false;
  size = (size_t)st.st_size;
  return true;
}

static void sdCloseLog() {
  if (sdFp) { fclose(sdFp); sdFp = nullptr; }
}

static bool sdOpenLog() {
  if (sdFp) return true;
  sdFp = fopen(sdFull, "a");   // "a" never truncates
  if (sdFp && fileSize(sdKnownSize)) return true;
  sdCloseLog();
  return false;
}

static bool sdMount() {
  SPI.begin(SD_SCK_PIN, SD_MISO_PIN, SD_MOSI_PIN, SD_CS_PIN);
  if (SD.begin(SD_CS_PIN, SPI, SD_SPI_HZ)) return true;
  SD.end(); SPI.end();
  return false;
}

enum class HeaderState { Fits, Foreign, Unreadable };

// Fits: missing, empty or starts with sdHeader. Foreign: another header. Unreadable: cannot be read.
static HeaderState headerState(const char* path) {
  for (int attempt = 0; attempt < SD_READ_ATTEMPTS; attempt++) {
    File f = SD.open(path, FILE_READ);
    if (!f) { if (!SD.exists(path)) return HeaderState::Fits; continue; }
    size_t bytes = f.size();
    if (bytes == 0) { f.close(); return HeaderState::Fits; }
    if (f.peek() < 0) { f.close(); continue; }   // non-empty file but the read failed: try again
    String first = f.readStringUntil('\n');
    f.close();
    first.trim();   // also drops a CR from a CRLF file
    return first == sdHeader ? HeaderState::Fits : HeaderState::Foreign;
  }
  return HeaderState::Unreadable;
}

// Never append rows under a different header (a stage 1 file, an older column layout).
static bool chooseFile(const char* path) {
  const char* dot = strrchr(path, '.');
  int stem = dot ? (int)(dot - path) : (int)strlen(path);
  for (int n = 0; n <= SD_MAX_VARIANT; n++) {
    if (n == 0) strlcpy(sdPath, path, sizeof(sdPath));
    else snprintf(sdPath, sizeof(sdPath), "%.*s_%d%s", stem, path, n, dot ? dot : "");
    HeaderState st = headerState(sdPath);
    if (st == HeaderState::Unreadable) { Serial.printf("SD: cannot read %s, remounting\n", sdPath); return false; }
    if (st == HeaderState::Foreign) continue;
    if (n > 0) Serial.printf("SD: %s has a different header, writing to %s\n", path, sdPath);
    snprintf(sdFull, sizeof(sdFull), "%s%s", SD_MOUNT_POINT, sdPath);
    return true;
  }
  Serial.printf("SD: no free variant of %s\n", path);
  return false;
}

// Write the header into an empty file, or note whether the file ends inside a row. True on success.
static bool sdPrepareFile() {
  sdCloseLog();
  FILE* fp = fopen(sdFull, "a+");   // "a+" creates, never truncates, and allows reading the tail
  if (!fp) return false;
  bool ok = fseek(fp, 0, SEEK_END) == 0;
  long size = ok ? ftell(fp) : -1;
  if (size == 0) {
    ok = fputs(sdHeader, fp) >= 0 && fputc('\n', fp) != EOF && fflush(fp) == 0 && fsync(fileno(fp)) == 0;
    sdNeedNewline = false;
  } else if (size > 0) {
    ok = fseek(fp, -1, SEEK_END) == 0;
    int last = ok ? fgetc(fp) : EOF;
    ok = ok && last != EOF;
    sdNeedNewline = ok && last != '\n';
  } else {
    ok = false;
  }
  return (fclose(fp) == 0) && ok;
}

static void mountHint() {
  pinMode(SD_MISO_PIN, INPUT_PULLDOWN); delay(2);
  bool misoHigh = digitalRead(SD_MISO_PIN);   // a powered, seated card holds MISO high
  pinMode(SD_MISO_PIN, INPUT);
  Serial.printf("SD init failed: MISO %s. Retrying in 3 s\n",
                misoHigh ? "high -> card powered, check CS/SCK/MOSI wires"
                         : "low -> no card, card not seated, module unpowered, or MISO wire open");
  lcd.clear(); lcd.print("SD init failed");
  lcd.setCursor(0, 1); lcd.print(misoHigh ? "chk CS/SCK/MOSI" : "card/power/MISO");
}

void sdMountBlocking(const char* path, const char* csvHeader, void (*whileWaiting)()) {
  sdHeader = csvHeader;
  sdBasePath = path;
  while (true) {   // a reseated card or wire is picked up without a reset
    if (sdMount() && chooseFile(path) && sdPrepareFile()) break;
    sdCloseLog();
    SD.end(); SPI.end();
    mountHint();
    for (uint32_t t0 = millis(); millis() - t0 < SD_RETRY_MS; delay(SD_WAIT_POLL_MS))
      if (whileWaiting) whileWaiting();
  }
  sdOk = true;
  Serial.printf("SD: logging to %s\n", sdPath);
}

static void sdQueueRow(const char* row) {
  size_t slot = (sdHead + sdCount) % SD_PENDING_ROWS;
  strlcpy(sdPending[slot], row, SD_ROW_BYTES);
  if (sdCount < SD_PENDING_ROWS) sdCount++;
  else { sdHead = (sdHead + 1) % SD_PENDING_ROWS; sdDropped++; }   // full: forget the oldest row
}

// Drop from the queue exactly the rows the confirmed growth covers; a row cut in half stays queued
// and the next write starts on a new line so the fragment cannot merge with it.
static void dequeueConfirmed(size_t grown, size_t batch) {
  for (size_t j = 0; j < batch; j++) {
    size_t len = strlen(sdPending[sdHead]);
    if (grown < len) { sdNeedNewline = grown > 0; return; }
    grown -= len;
    sdHead = (sdHead + 1) % SD_PENDING_ROWS;
    sdCount--;
  }
}

// One batch: header or separating newline if needed, rows, fflush + fsync, then confirm via stat().
static bool flushBatch(FILE* fp) {
  size_t before, extra = 0, bytes = 0, batch = min(sdCount, SD_FLUSH_BATCH);
  if (fflush(fp) != 0 || !fileSize(before)) return false;
  if (before != sdKnownSize) return false;   // changed behind our back (card swapped): remount and re-check
  bool wrote = true;
  if (before == 0) {   // new, or recreated while running (card swapped): it needs its header
    wrote = fputs(sdHeader, fp) >= 0 && fputc('\n', fp) != EOF;
    extra = strlen(sdHeader) + 1;
  } else if (sdNeedNewline) {
    wrote = fputc('\n', fp) != EOF;
    extra = 1;
  }
  for (size_t j = 0; wrote && j < batch; j++) {
    const char* row = sdPending[(sdHead + j) % SD_PENDING_ROWS];
    size_t len = strlen(row);
    wrote = fwrite(row, 1, len, fp) == len;
    if (wrote) bytes += len;
  }
  size_t onCard;
  if (fflush(fp) != 0 || fsync(fileno(fp)) != 0 || !fileSize(onCard)) return false;   // keep all queued
  if (onCard < before + extra) return false;
  sdKnownSize = onCard;
  sdNeedNewline = false;
  dequeueConfirmed(onCard - before - extra, batch);
  return wrote && onCard - before - extra == bytes;
}

// Write the whole queue in confirmed batches through the open log. True when the queue is empty.
static bool sdFlushToCard() {
  if (!sdOpenLog()) return false;
  bool ok = true;
  while (ok && sdCount) ok = flushBatch(sdFp);
  if (!ok) sdCloseLog();   // the next attempt starts from a fresh handle after a remount
  return ok && sdCount == 0;
}

// Called after a failed write. Rate-limited so a dead card cannot stall the 3 s cycle.
static void sdRecover() {
  if (millis() - sdLastRemountMs < SD_REMOUNT_MS) return;
  sdLastRemountMs = millis();
  sdCloseLog();   // its FatFs object dies with the unmount
  SD.end(); SPI.end(); delay(20);
  // the header check again: the card may have been swapped for one with another layout
  bool mounted = sdMount() && chooseFile(sdBasePath) && sdPrepareFile();
  if (mounted) sdOk = sdFlushToCard();   // flush now, not next cycle
  Serial.printf("SD: %s (%s), %u rows waiting in RAM, %u dropped since boot\n",
                !mounted ? "remount failed (card power or wires)" : sdOk ? "remounted, backlog confirmed on card" : "remounted, write not confirmed",
                sdPath, (unsigned)sdCount, (unsigned)sdDropped);
}

void sdService() {
  if (sdCount == 0) return;
  if (sdOk && sdFlushToCard()) return;
  sdOk = false;
  sdRecover();
}

void sdStore(const char* row) {
  sdQueueRow(row);   // every row goes through the queue and leaves it only once confirmed
  sdService();
}

size_t sdPendingCount() { return sdCount; }
size_t sdDroppedCount() { return sdDropped; }
const char* sdFileName() { return sdPath; }
