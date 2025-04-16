#!/bin/bash

# Mount Google Drive
if [ ! -d ~/gdrive ]; then
    mkdir -p ~/gdrive
    chmod 775 ~/gdrive
fi

rclone mount drive: ~/gdrive --vfs-cache-mode full &

if mountpoint -q ~/gdrive; then
    echo "Google Drive mounted at ~/gdrive"
else
    echo "Failed to mount."
fi

# Instructions for making the script executable and running it
# Run 'chmod +x drive_mount.sh' to make the script executable
# Run './drive_mount.sh' to execute the script