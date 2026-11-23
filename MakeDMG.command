#!/bin/bash
# Double-click to package build/Smooth60.app into a drag-to-Applications DMG in the dist folder.
cd "$(dirname "$0")" || exit 1
{
  APP="build/Smooth60.app"
  if [ ! -d "$APP" ]; then
    echo "ERROR: build/Smooth60.app not found — double-click Build.command first."
    exit 1
  fi
  VERSION=$(defaults read "$(pwd)/$APP/Contents/Info.plist" CFBundleShortVersionString 2>/dev/null || echo "0.1")
  OUT="dist/Smooth60-$VERSION.dmg"
  STAGE=$(mktemp -d /tmp/smooth60-dmg.XXXXXX)

  cp -R "$APP" "$STAGE/"
  ln -s /Applications "$STAGE/Applications"
  cat > "$STAGE/Read Me First.txt" <<'TXT'
Smooth60 (beta) — 30 fps console games at 60 fps on your Mac

1. Drag Smooth60 into the Applications folder.
2. Open it. If macOS says it can't verify the app:
   System Settings → Privacy & Security → scroll down → "Open Anyway".
3. Allow Camera and Microphone access (that's how it reads your capture card).

Needs: Apple Silicon Mac, macOS 26 or later, a USB capture card (UVC).
PS5: set video output to 1080p and turn HDR off.

Shortcuts: ⌘↩ start/stop · ⌃⌘F full screen · ⌘G frame generation on/off
           ⌘B before/after · ⇧⌘O stream output window · F8 record · F7 screenshot
Recordings and screenshots: Desktop › Smooth60 (change it in the Display tab).
4K capture cards work; if needed, frame generation runs at 1080p automatically.
TXT

  mkdir -p dist
  rm -f "$OUT"
  hdiutil create -volname "Smooth60" -srcfolder "$STAGE" -ov -format UDZO "$OUT" || { echo "DMG FAILED"; exit 1; }
  rm -rf "$STAGE"
  echo "DMG READY: $(pwd)/$OUT ($(du -h "$OUT" | cut -f1))"
} 2>&1 | tee dmg.log

[ -d dist ] && open dist
echo
echo "Done. You can close this window."
