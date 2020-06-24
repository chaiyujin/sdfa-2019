
def add_methods_from(*modules):
    def decorator(Class):
        for module in modules:
            if hasattr(module, "__methods__"):
                for method in getattr(module, "__methods__"):
                    setattr(Class, method.__name__, method)
            if hasattr(module, "__classmethods__"):
                for method in getattr(module, "__classmethods__"):
                    setattr(Class, method.__name__, ClassMethod(method))
            if hasattr(module, "__staticmethods__"):
                for method in getattr(module, "__staticmethods__"):
                    setattr(Class, method.__name__, StaticMethod(method))
        return Class
    return decorator


def register_method(methods):
    def _register_method(method):
        methods.append(method)
        return method  # Unchanged
    return _register_method


def extend(Class):
    def decorator(method):
        setattr(Class, method.__name__, method)
        return method
    return decorator


def extend_classmethod(Class):
    def decorator(method):
        setattr(Class, method.__name__, ClassMethod(method))
        return method
    return decorator


def extend_staticmethod(Class):
    def decorator(method):
        setattr(Class, method.__name__, StaticMethod(method))
        return method
    return decorator


class StaticMethod(object):
    "Emulate PyStaticMethod_Type() in Objects/funcobject.c"

    def __init__(self, f):
        self.f = f

    def __get__(self, obj, objtype=None):
        return self.f


class ClassMethod(object):
    "Emulate PyClassMethod_Type() in Objects/funcobject.c"

    def __init__(self, f):
        self.f = f

    def __get__(self, obj, klass=None):
        if klass is None:
            klass = type(obj)

        def newfunc(*args, **kwargs):
            return self.f(klass, *args, **kwargs)

        return newfunc
