
'''
# setup instructions (for wsl/ubuntu)
## 1. create and activate the conda environment

conda create --name <<name>> python=3.10 -y
conda activate <<name>>

## 2. install system dependencies (needed for mounting s3 buckets)
# these are NOT conda packages — install them using apt

sudo apt update
sudo apt install -y s3fs libfuse-dev

## 3. configure aws credentials for s3fs
# replace YOUR_ACCESS_KEY and YOUR_SECRET_KEY with actual values

echo "YOUR_ACCESS_KEY:YOUR_SECRET_KEY" > ~/.passwd-s3fs
chmod 600 ~/.passwd-s3fs

## 4. enable 'allow_other' option for s3fs (only needs to be done once)
# this lets s3fs-mounted folders be accessible to your user

sudo nano /etc/fuse.conf
# in the opened file, uncomment this line:
# user_allow_other
# save and exit (Ctrl + O → Enter, then Ctrl + X)

## 5. create the mount point inside wsl (don’t use /mnt/c/... for s3fs)

mkdir -p ~/s3bucket

## 6. mount the s3 bucket using s3fs
# replace 'fcp-indi' with your actual bucket name

s3fs fcp-indi ~/s3bucket \
    -o passwd_file=~/.passwd-s3fs \
    -o use_path_request_style \
    -o url=https://s3.amazonaws.com \
    -o allow_other

## s3 bucket is now mounted at ~/s3bucket.

## 7. navigate to the project directory (if not already there)

cd /mnt/c/Users/Lenovo/OneDrive/ASD-BranchNet

## 8. create a symbolic link to the mounted s3 bucket inside the working directory
# this makes it easier to access s3 content without switching directories

ln -s <path of actual> <path of symlink>

## you should now see 's3bucket_shortcut' inside your project directory.
# you can access it from scripts as a relative path, e.g., 's3bucket_shortcut/my_file.csv'

## 9. confirm symlink creation

ls -l
# output should include a line like:
# s3bucket_shortcut -> /home/<<user_name>>/s3bucket

## 10. ignore this on git, it doesn’t need to be tracked.

## 11. optionally, launch vscode via wsl for better folder visibility and compatibility.
run (code .) in the wsl terminal to open the current directory in vscode.
'''

#Mounting google drive
'''
sudo apt update
sudo apt install rclone -y

follow subsequent steps..

root-folder-id -> <<get from URL of the folder in gdrive>>

mkdir -p ~/gdrive
chmod 775 ~/gdrive
rclone mount drive: ~/gdrive --vfs-cache-mode full
(keep this running and perform other tasks in a new terminal)
'''