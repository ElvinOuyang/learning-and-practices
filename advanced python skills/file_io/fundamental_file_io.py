import os

# how to print to a file instead of the console:
# 't' mode opens the file as text file
with open('file_io/test.txt', 'wt') as f:
    print('Hello World!\nHello Again', file=f)

# use newline parameter to define new line signifier
with open('file_io/test.txt', 'rt', newline='') as f:
    g = f.read()

# write to a file only if it does not exist previously
# use 'x' mode in open()
with open('file_io/test_3.txt', 'xt') as f:
    print('another test!', file=f)
# above code is equal to
if not os.path.exists('file_io/test_3.txt'):
    with open('file_io/test_3.txt', 'wt') as f:
        f.write('hello!')
else:
    print('file already exists')
