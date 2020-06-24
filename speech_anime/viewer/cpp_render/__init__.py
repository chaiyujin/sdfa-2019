import os
import saber
import numpy as np
from . import obj_render

_root = os.path.join(os.path.abspath(os.path.dirname(__file__)))
# set snow root
obj_render.set_work_dir(os.path.abspath("."))
obj_render.set_snow_root(_root)
obj_render.set_msaa(1)

# set voca template
_template_verts, _template_faces = None, None


def set_template(template):
    global _template_verts
    global _template_faces
    assert isinstance(template, str)
    assert os.path.splitext(template)[1] == ".obj"

    obj_render.set_template(template)
    _template_verts, _template_faces =\
        saber.mesh.read_mesh(template, flatten=True)


def render_mesh(verts: np.ndarray, faces: np.ndarray = None):
    assert _template_verts is not None and _template_faces is not None,\
        "Template is not set yet! please call set_template()"
    # check same
    if faces is not None:
        assert np.all(faces.flatten() == _template_faces.flatten())
    verts = verts.flatten(order='C').astype(np.float32)
    assert len(verts) == len(_template_verts),\
        "given verts length should be {}! not {}".format(len(_template_verts), len(verts))
    return obj_render.render_verts(verts)
