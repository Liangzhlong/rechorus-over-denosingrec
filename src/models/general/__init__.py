from os.path import dirname, basename, isfile, join
import glob
import importlib

modules = glob.glob(join(dirname(__file__), "*.py"))
__all__ = []

for f in modules:
    if isfile(f) and not f.endswith('__init__.py'):
        module_name = basename(f)[:-3]
        try:
            module = importlib.import_module('.' + module_name, package=__package__)
            if hasattr(module, module_name):
                globals()[module_name] = getattr(module, module_name)
                __all__.append(module_name)
        except Exception as e:
            pass
