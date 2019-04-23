# test basic string decorator without arguments
# wrappers start from bottom to top
from functools import wraps


def uppercase(func):
    def wrapper(*args, **kwargs):
        print('creating uppercase string')
        modified_result = func(*args, **kwargs).upper()
        return modified_result
    print('created uppercase wrapper')
    return wrapper


def bold(func):
    def wrapper(*args, **kwargs):
        print('creating markdown string for bold')
        modified_result = '**' + func(*args, **kwargs) + '**'
        return modified_result
    print('created bold wrapper')
    return wrapper


def bullets(func):
    def wrapper(*args, **kwargs):
        print('creating markdown string for bullets')
        modified_result = '- ' + func(*args, **kwargs)
        return modified_result
    print('created bullet wrapper')
    return wrapper


# decorators below equals uppercase(bold(bullets(greet)))
@uppercase
@bold
@bullets
def greet():
    print("running greet()")
    return "hi!"


# test decorator that takes in arguments
def trace(func):
    def wrapper(*args, **kwargs):
        # taking the exact same arguments make the wrapper
        # feels the save as the func()
        print('TRACE: calling {0} with {1}, {2}'.format(
            func.__name__, args, kwargs))
        original_result = func(*args, **kwargs)
        print('TRACE: {0} returned {1}'.format(
            func.__name__, repr(original_result)))
        return original_result
        # returning original result makes the decorated function
        # works like the func()
    return wrapper


@trace
def arith(a, b):
    x = a + b
    return x


@trace
@uppercase
@bold
def greet_again(a):
    return a


# test decorator with parameters and maintain help info
def string_manipulation(wrap_string):
    # extra wrap just for input of customized string
    def string_wrapper(func):
        @wraps(func)
        # decorate the func to copy over the help info
        def wrapper(*args, **kwargs):
            updated_string = '{0}{1}{0}'.format(
                wrap_string, func(*args, **kwargs))
            print('updated string created')
            return updated_string
        print('wrapper created')
        return wrapper
    print('string_wrapper created')
    return string_wrapper


@string_manipulation('**')
def combine_strings(a, b):
    '''
Gets combined string from input
    '''
    combined = a + b
    print('string combined')
    return combined
