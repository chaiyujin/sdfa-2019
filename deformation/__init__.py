import os

_dir = os.path.dirname(__file__)
_build = os.path.join(_dir, "cpp", "build")

try:
    from .cpp.build.deformation import *
except Exception:
    os.makedirs(_build, exist_ok=True)
    code = os.system(f"echo 'cd {_build} && cmake .. && make -j8' | bash")
    if code != 0:
        quit()
    from .cpp.build.deformation import *
