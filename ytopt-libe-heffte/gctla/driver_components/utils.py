import inspect

def pretty_function_print(func):
    if not callable(func):
        return f"{func}"
    if inspect.isbuiltin(func):
        return f"builtin_function {func.__name__}"
    if inspect.ismethod(func):
        class_name = func.__self__.__class__.__name__
        return f"class:{class_name} {func.__name__}"
    if inspect.isfunction(func):
        module_name = inspect.getmodule(func).__name__
        return f"module:{module_name} {func.__name__}"

