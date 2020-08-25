import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import cv2
import saber
import trimesh
import pyrender
import numpy as np

# set voca template
_template_verts, _template_faces = None, None
_template_mesh = None
# scene
_cam = pyrender.PerspectiveCamera(yfov=(np.pi / 4.0))
_cam_pose = np.asarray([
    [ 9.84561989e-01, -1.14640632e-02,  1.74657155e-01,  7.99997887e-02],
    [-2.63421926e-08,  9.97852584e-01,  6.54966148e-02,  3.00000020e-02],
    [-1.75033820e-01, -6.44855109e-02,  9.82448868e-01,  4.49999897e-01],
    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00],
])
_scene = pyrender.Scene(ambient_light=np.array([0.02, 0.02, 0.02, 1.0]))
_scene.add(_cam, pose=_cam_pose)
_scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=3.5), pose=_cam_pose)
_scene.add(pyrender.PointLight      (color=np.ones(3), intensity=0.5), pose=_cam_pose)
# render
_renderer = None


def set_template(template):
    global _template_verts
    global _template_faces
    global _template_mesh
    assert isinstance(template, str)
    assert os.path.exists(template)

    _template_mesh = pyrender.Mesh.from_trimesh(trimesh.load(template, process=False))
    _template_verts, _template_faces = saber.mesh.read_mesh(template, flatten=True)
    _scene.add(_template_mesh)


def render_mesh(verts: np.ndarray, faces: np.ndarray = None, image_size: tuple = (512, 512)):
    global _renderer
    assert _template_verts is not None and _template_faces is not None,\
        "Template is not set yet! please call set_template()"
    # check same
    if faces is not None:
        assert np.all(faces.flatten() == _template_faces.flatten())
    verts = verts.flatten(order='C').astype(np.float32)
    assert len(verts) == len(_template_verts),\
        "given verts length should be {}! not {}".format(len(_template_verts), len(verts))
    # render
    if _renderer is None or image_size != (_renderer.viewport_width, _renderer.viewport_height):
        if _renderer is not None:
            _renderer.delete()
        _renderer = pyrender.OffscreenRenderer(viewport_width=image_size[0], viewport_height=image_size[1])
    _template_mesh.primitives[0].positions = np.reshape(verts, (-1, 3))
    for mesh in _renderer._renderer._meshes:
        for p in mesh.primitives:
            p.delete()
    _renderer._renderer._meshes = set()
    img = _renderer.render(_scene)[0]
    # make sure the size
    if image_size != (img.shape[1], img.shape[0]):
        img = cv2.resize(img, image_size)
    return img
