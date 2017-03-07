import random  # import the module "random"
for i in range(5):
    print(random.randint(1,10))

import random, sys, os, math

# None value
spam = print('Hello!')
print(spam)  # for functions that does not return calculated values, they return None value

# Keyword arguments
print('Hello',end='')  # optional parameters defined by key words
print('World')
# by defining the end of previous print() argument, the string would not end with new line

print('cat','dog','mouse',sep=',')
# key word parameter sep defines what should come between each string


# global scope and local scope
def spam():
    global eggs  # this line tells Python not to create a local variable egg
    eggs = 'spam'

eggs = 'global'  # variable eggs is valued as 'global'
spam()  # function spam() gives eggs a new value, 'spam'
print(eggs)


# Handling error
def spam(divideby):
    try:
        return 42 / divideby
    except ZeroDivisionError:  # the program handles the error as if it is a condition
        print('Error: Invalid argument.')

print(spam(2))
print(spam(12))
print(spam(0))  # this line will return the printed warning since it will return ZeroDivisionError
print(spam(1))

# introducing module webbrowser
import webbrowser
webbrowser.open('http://inventwithpython.com')
