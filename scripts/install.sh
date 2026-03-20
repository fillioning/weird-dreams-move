#!/bin/bash
set -e
MODULE_ID="weird-dreams"
MOVE_HOST="${1:-${MOVE_HOST:-move.local}}"
DEST="/data/UserData/move-anything/modules/sound_generators"

echo "Installing $MODULE_ID to $MOVE_HOST..."
scp -r "dist/$MODULE_ID" "root@$MOVE_HOST:$DEST/"
ssh root@$MOVE_HOST "chown -R ableton:users $DEST/$MODULE_ID"
echo "Done. Restart Move to load."
