from .utils import pretty_function_print

def NoOp(*args,**kwargs):
    pass

# Useful for pre-launch / post-launch commands that cannot be executed by a subshell
class PythonEval():
    def __init__(self, func=NoOp, args=(), kwargs={}, debug_level=0, **other_attrs):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.debug_level = debug_level
        for (attr, value) in other_attrs.items():
            setattr(self, attr, value)
    def __str__(self):
        return f"Calls {pretty_function_print(self.func)} with:"+\
                "\n\targs:\n\t\t"+"\n\t\t".join([str(_) for _ in self.args])+\
                "\n\tkwargs:\n\t\t"+"\n\t\t".join([f"Key: '{k}'; Value: '{v}'" for (k,v) in self.kwargs.items()])
    def make_args(self, job):
        new_args = []
        for attr in self.args:
            if hasattr(attr, 'format') and callable(getattr(attr,'format')):
                attr = attr.format(**job.__dict__)
            new_args.append(attr)
        new_kwargs = {}
        for key, val in self.kwargs.items():
            if hasattr(val, 'format') and callable(getattr(val,'format')):
                val = val.format(**job.__dict__)
            new_kwargs[key] = val
        return new_args, new_kwargs
    def __call__(self, job):
        new_args, new_kwargs = self.make_args(job)
        print(pretty_function_print(self.func))
        if self.debug_level > 0:
            if len(new_args) > 0:
                print("\targs:\n\t\t"+"\n\t\t".join([f"#{idx}: {pretty_function_print(arg)}" for idx, arg in enumerate(new_args)]))
            if len(new_kwargs) > 0:
                print("\tkwargs:\n\t\t"+"\n\t\t".join([f"Key: '{k}'; Value: '{pretty_function_print(v)}'" for (k,v) in new_kwargs.items()]))
        self.func(*new_args, **new_kwargs)

