import os
import time

cwd = os.getcwd()

# use os.path for pathname manipulation
# to get last component of a pathname
os.path.basename(cwd)
# to get the directory name of a path
os.path.dirname(cwd)
# to join path components
os.path.join(cwd, 'test.txt')
# expand user's home directory
path = '~/Home/text.txt'
os.path.expanduser(path)
# split the file extension
os.path.splitext(path)

# file existence testing
# exist as a file
os.path.isfile('file_io/pathname_manipulation.py')
# exist as a directory
os.path.isdir('file_io')
# file metadata functions
os.path.getsize('file_io/pathname_manipulation.py')
os.path.getmtime('file_io/pathname_manipulation.py')
time.ctime(
    os.path.getmtime('file_io/pathname_manipulation.py'))

# file directory listing
# list all regular files in a directory
file_names = [name for name in os.listdir(cwd)
              if os.path.isfile(os.path.join(cwd, name))]
# list all directories in a directory
dir_names = [name for name in os.listdir(cwd)
             if os.path.isdir(os.path.join(cwd, name))]
