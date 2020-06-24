import os
import saber
import numpy as np
from . import get_dr
from . import get_mesh


def read_mesh(mesh_path):
    if os.path.splitext(mesh_path)[1] == ".ply":
        verts, faces = saber.mesh.io.read_ply(mesh_path)
    elif os.path.splitext(mesh_path)[1] == ".obj":
        verts, faces = saber.mesh.io.read_obj(mesh_path)
    verts = np.reshape(verts, (-1, 3))
    faces = np.reshape(faces, (-1, 3))
    return verts, faces


def write_obj(obj_path, verts, faces, eye_indices=None):
    with open(obj_path, "w") as fp:
        for vi, vert in enumerate(verts):
            if eye_indices is not None and vi in eye_indices:
                continue
            fp.write(f"v {vert[0]} {vert[1]} {vert[2]}\n")
        for face in faces:
            f0 = face[0]
            f1 = face[1]
            f2 = face[2]
            if eye_indices is not None:
                if eye_indices[0] <= f0 <= eye_indices[-1]: continue
                if f0 > eye_indices[-1]: f0 = f0 - eye_indices[-1] + eye_indices[0]
                if f1 > eye_indices[-1]: f1 = f1 - eye_indices[-1] + eye_indices[0]
                if f2 > eye_indices[-1]: f2 = f2 - eye_indices[-1] + eye_indices[0]
            fp.write(f"f {f0+1} {f1+1} {f2+1}\n")


def remove_eye_indices(indices, eye_indices):
    new_idx = []
    for idx in indices:
        if idx < eye_indices[0]:
            new_idx.append(idx)
        elif idx > eye_indices[-1]:
            new_idx.append(idx - eye_indices[-1] + eye_indices[0])
    return new_idx


assets_root = os.path.join(os.path.dirname(__file__), "assets")
_, all_faces = read_mesh(os.path.join(assets_root, "with_eyeballs.obj"))

with open(os.path.join(assets_root, "voca_eye.txt")) as fp:
    eye_indices = [int(x) for x in fp.readline().split()]
    for i in range(len(eye_indices) - 1):
        assert eye_indices[i] + 1 == eye_indices[i + 1]

with open(os.path.join(assets_root, "voca_eye_faces.txt")) as fp:
    eye_face_indices = [int(x) for x in fp.readline().split()]
    for i in range(len(eye_face_indices) - 1):
        assert eye_face_indices[i] + 1 == eye_face_indices[i + 1]

with open(os.path.join(assets_root, "voca_lower_face.txt")) as fp:
    lower_indices = [int(x) for x in fp.readline().split()]
    non_lower_indices = [x for x in range(5023) if x not in lower_indices]
    lower_indices = remove_eye_indices(lower_indices, eye_indices)
    non_lower_indices = remove_eye_indices(non_lower_indices, eye_indices)


Identity = np.asarray([1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def remove_eyeballs(src):
    fname = src
    if fname.find("_no_eyeballs.obj") < 0:
        _maybe = os.path.splitext(fname)[0] + "_no_eyeballs.obj"
        if not os.path.exists(_maybe):
            full_verts, full_faces = read_mesh(fname)
            if len(full_verts) == 5023:
                fname = _maybe
                write_obj(fname, full_verts, full_faces, eye_indices)
        else:
            fname = _maybe
    return fname


def deformation_transfer(srcA_fname, srcB_fname, tarA_fname, tarB_fname):
    # remove eyeballs
    _srca = remove_eyeballs(srcA_fname)
    _srcb = remove_eyeballs(srcB_fname)
    _tara = remove_eyeballs(tarA_fname)
    assert _tara != tarA_fname, "tarA frame must have eyeballs"

    deform_grad = get_dr.get_dr(_srca, _srcb)
    deform_grad = np.reshape(deform_grad, (-1, 9))

    deform_grad[non_lower_indices] = Identity
    deform_grad = deform_grad.flatten()

    m = get_mesh.get_mesh(_tara, deform_grad)
    verts = np.reshape(m, (-1, 3))

    # add eyeballs!
    all_verts, _ = read_mesh(tarA_fname)
    all_verts[:eye_indices[0]] = verts[:eye_indices[0]]
    all_verts[eye_indices[-1]+1:] = verts[eye_indices[0]:]

    tarB_fname = os.path.splitext(tarB_fname)[0] + '.obj'
    write_obj(tarB_fname, all_verts, all_faces)

    if _srcb != srcB_fname: os.remove(_srcb)
