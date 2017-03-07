#! python3
# mapIt.py - launches a map in the browser using an address from
# the command line or clipboard.

import webbrowser, sys, pyperclip
if len(sys.argv) > 1:
    # Get address from command line.
    address = ' '.join(sys.argv[1:])  # .join() method puts a list as a single string
else:
    # Get address from clipboard.
    address = pyperclip.paste()

webbrowser.open('http://www.google.com/maps/place/' + address)