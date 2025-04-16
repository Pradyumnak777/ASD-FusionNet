
'''
# Setup Instructions (for WSL/Ubuntu)
## 1. Create and activate the Conda environment

conda create --name <<name>> python=3.10 -y
conda activate <<>name>>

## 2. Install system dependencies (required for mounting S3 buckets)
# These are NOT Conda packages — install them using apt

sudo apt update
sudo apt install -y s3fs libfuse-dev

## 3. Configure AWS credentials for s3fs
# Replace YOUR_ACCESS_KEY and YOUR_SECRET_KEY with actual values

echo "YOUR_ACCESS_KEY:YOUR_SECRET_KEY" > ~/.passwd-s3fs
chmod 600 ~/.passwd-s3fs

## 4. Enable 'allow_other' option for s3fs (only needs to be done once)
# This allows s3fs-mounted folders to be accessible to your user

sudo nano /etc/fuse.conf
# In the opened file, uncomment the following line:
# user_allow_other
# Save and exit (Ctrl + O → Enter, then Ctrl + X)

## 5. Create the mount point inside WSL (DO NOT use /mnt/c/... for s3fs)

mkdir -p ~/s3bucket

## 6. Mount the S3 bucket using s3fs
# Replace 'fcp-indi' with your actual bucket name

s3fs fcp-indi ~/s3bucket \
    -o passwd_file=~/.passwd-s3fs \
    -o use_path_request_style \
    -o url=https://s3.amazonaws.com \
    -o allow_other

##  S3 bucket is now mounted at ~/s3bucket.

## 7. Navigate to the project directory (if not already there)

cd /mnt/c/Users/Lenovo/OneDrive/ASD-BranchNet

## 8. Create a symbolic link to the mounted S3 bucket inside the working directory
# This enables easier access to S3 content without changing directories

ln -s <path of actual> <path of symlink>

## You should now see 's3bucket_shortcut' inside your project directory.
# This can be accessed from scripts as a relative path, e.g., 's3bucket_shortcut/my_file.csv'

## 9. Confirm symlink creation

ls -l
# Output should include a line like:
# s3bucket_shortcut -> /home/pajju/s3bucket

##10. ignore this on git, as it doesnt need to be tracked.

##11. optionally, launch vscode via wsl for better folder visibility and compatibilty.
run (code .) in the wsl terminal to open the current directory in VSCode.
'''

#Mounting google drive
'''
sudo apt update
sudo apt install rclone -y

follow subsequent steps..

for project-2, root-folder-id -> 19_-IejK1SlQEzX1mADeu4ks05phhlRQo

mkdir -p ~/gdrive
chmod 775 ~/gdrive
rclone mount drive: ~/gdrive --vfs-cache-mode full
(keep this running and perform other tasks in a new terminal)
'''