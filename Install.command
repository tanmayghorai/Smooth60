#!/bin/bash
# Double-click to install the built Smooth60 into /Applications (run Build.command first).
cd "$(dirname "$0")" || exit 1
{
  echo "=== Smooth60 install — $(date) ==="
  SRC="build/Smooth60.app"
  if [ ! -d "$SRC" ]; then
    echo "ERROR: $SRC not found — double-click Build.command first."
    exit 1
  fi

  DEST_DIR="/Applications"
  if [ ! -w "$DEST_DIR" ]; then
    DEST_DIR="$HOME/Applications"   # no permission for /Applications: use your personal Applications folder
    mkdir -p "$DEST_DIR"
  fi
  DEST="$DEST_DIR/Smooth60.app"

  # Quit any running copy so it can be replaced.
  pkill -x Smooth60 2>/dev/null && sleep 1

  rm -rf "$DEST"
  ditto "$SRC" "$DEST" || { echo "INSTALL FAILED"; exit 1; }
  xattr -dr com.apple.quarantine "$DEST" 2>/dev/null
  touch "$DEST"
  # Tell Launchpad/Spotlight about it.
  /System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister -f "$DEST" 2>/dev/null

  echo "INSTALLED: $DEST"
} 2>&1 | tee install.log

STATUS=${PIPESTATUS[0]}
if [ "$STATUS" -eq 0 ]; then
  DEST=$(grep '^INSTALLED: ' install.log | sed 's/^INSTALLED: //')
  [ -n "$DEST" ] && open "$DEST"
fi
echo
echo "Done. You can close this window."
