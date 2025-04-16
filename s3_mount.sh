#!/bin/bash

# Mount S3 bucket
if [ ! -d ~/s3bucket ]; then
    mkdir -p ~/s3bucket
fi

s3fs fcp-indi ~/s3bucket \
    -o passwd_file=~/.passwd-s3fs \
    -o use_path_request_style \
    -o url=https://s3.amazonaws.com \
    -o allow_other

echo "S3 bucket mounted at ~/s3bucket"

#run 'chmod +x mount_s3.sh' to make the script executable
#'./s3_mount.sh' to run