#!/bin/bash
# Double-click this file to build and open Smooth60.
cd "$(dirname "$0")" || exit 1
# Only one build at a time (a double-click can start two).
LOCK="/tmp/smooth60-build.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
  if [ -n "$(find "$LOCK" -maxdepth 0 -mmin +5 2>/dev/null)" ]; then rmdir "$LOCK"; mkdir "$LOCK"; else echo "A build is already running."; exit 0; fi
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT
LOG="build.log"
{
  echo "=== Smooth60 build — $(date) ==="
  echo "macOS: $(sw_vers -productVersion) ($(sw_vers -buildVersion))"
  echo "Chip: $(sysctl -n machdep.cpu.brand_string 2>/dev/null)"
  echo "Memory: $(( $(sysctl -n hw.memsize) / 1073741824 )) GB"
  echo "Developer dir: $(xcode-select -p 2>&1)"
  echo "SDK: $(xcrun --sdk macosx --show-sdk-version 2>&1)"
  echo "Swift: $(xcrun swiftc --version 2>&1 | head -1)"
  echo "--- Capture devices (USB) ---"
  system_profiler SPCameraDataType 2>/dev/null | grep -E "^    [^ ].*:$|Model ID" || true
  echo "-----------------------------"
  echo "--- VTFrameProcessor API ---"
  HDR="$(xcrun --sdk macosx --show-sdk-path)/System/Library/Frameworks/VideoToolbox.framework/Headers"
  grep -n -B3 -A3 "processWithParameters\|processWithCommandBuffer" "$HDR/VTFrameProcessor.h" 2>&1 | head -80
  grep -n "interpolationPhase\|initWithSourceFrame" "$HDR"/VTLowLatencyFrameInterpolation*.h 2>&1 | head -20
  echo "-----------------------------"

  MAJOR=$(sw_vers -productVersion | cut -d. -f1)
  if [ "$MAJOR" -lt 26 ]; then
    echo "ERROR: Smooth60 needs macOS 26 or later (this Mac has $(sw_vers -productVersion))."
    exit 1
  fi
  if ! xcrun --find swiftc >/dev/null 2>&1; then
    echo "ERROR: Swift compiler not found. Run: xcode-select --install"
    exit 1
  fi

  # Quit a running copy so the new build replaces it.
  pkill -x Smooth60 2>/dev/null && sleep 1

  APP="build/Smooth60.app"
  rm -rf "$APP"
  mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
  cp Info.plist "$APP/Contents/Info.plist"
  [ -f AppIcon.icns ] && cp AppIcon.icns "$APP/Contents/Resources/AppIcon.icns"

  echo "Compiling…"
  xcrun --sdk macosx swiftc -O -parse-as-library -swift-version 5 \
    -target arm64-apple-macos26.0 \
    Sources/main.swift \
    -o "$APP/Contents/MacOS/Smooth60" || { echo "BUILD FAILED"; exit 1; }

  codesign --force --sign - "$APP" || { echo "CODESIGN FAILED"; exit 1; }
  touch "$APP"   # so Finder and the Dock pick up the new icon
  echo "BUILD SUCCEEDED: $(pwd)/$APP"
} 2>&1 | tee "$LOG"

STATUS=${PIPESTATUS[0]}
if [ "$STATUS" -eq 0 ] && [ -d build/Smooth60.app ]; then
  # Install the fresh build into Applications and open it from there.
  bash ./Install.command
  exit 0
fi
echo
echo "Done. You can close this window."
