from functools import wraps

def my_wraps(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        print("Calling my wrappers")
        return f(*args, **kwargs)
    return wrapper

@my_wraps
def my_function():
    print("Calling my functions")
my_function()

