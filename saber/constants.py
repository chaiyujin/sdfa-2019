import os


class _Constants(object):
    def __init__(self):
        self._root = os.path.abspath(os.path.dirname(__file__))
        self._temp_dir = os.path.join(self._root, ".tmp")
        self._assets_dir = os.path.join(self._root, "assets")

    @property
    def temp_dir(self):
        return self._temp_dir

    @property
    def assets_dir(self):
        return self._assets_dir


constants = _Constants()
