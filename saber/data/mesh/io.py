import os
import plyfile
import numpy as np


def read_ply(ply_path, expect_indices=None, dtype=np.float32, flatten=True):
    plydata = plyfile.PlyData.read(ply_path)
    verts = np.stack((
        plydata["vertex"]["x"],
        plydata["vertex"]["y"],
        plydata["vertex"]["z"]
    ), axis=1).astype(dtype)
    faces = np.stack(plydata["face"]["vertex_indices"], axis=0)\
              .astype(np.uint32)
    if flatten:
        verts = verts.flatten(order='C')
        faces = faces.flatten(order='C')
    if expect_indices is not None:
        assert np.all(expect_indices == faces)
    return verts, faces


def read_obj(filename_obj, expect_indices=None, dtype=np.float32, flatten=True, normalization=False):
    # load vertices
    with open(filename_obj) as f:
        lines = f.readlines()

    vertices = []
    for line in lines:
        if len(line.split()) == 0:
            continue
        if line.split()[0] == 'v':
            vertices.append([float(v) for v in line.split()[1:4]])
    vertices = np.vstack(vertices)

    # load faces
    faces = []
    for line in lines:
        if len(line.split()) == 0:
            continue
        if line.split()[0] == 'f':
            vs = line.split()[1:]
            nv = len(vs)
            v0 = int(vs[0].split('/')[0])
            for i in range(nv - 2):
                v1 = int(vs[i + 1].split('/')[0])
                v2 = int(vs[i + 2].split('/')[0])
                faces.append((v0, v1, v2))
    faces = (np.vstack(faces) - 1).astype(np.uint32)

    # normalize into a unit cube centered zero
    if normalization:
        vertices -= vertices.min(0)[0][None, :]
        vertices /= np.abs(vertices).max()
        vertices *= 2
        vertices -= vertices.max(0)[0][None, :] / 2

    verts = vertices.astype(dtype)
    faces = faces.astype(np.uint32)

    if flatten:
        verts = verts.flatten(order='C')
        faces = faces.flatten(order='C')

    if expect_indices is not None:
        assert np.all(expect_indices == faces)

    return verts, faces


def write_obj(fname, verts, faces):
    with open(fname, "w") as fp:
        for vert in verts:
            fp.write(f"v {vert[0]} {vert[1]} {vert[2]}\n")
        for face in faces:
            fp.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def read_mesh(fname, dtype=np.float32, flatten=False):
    ext = os.path.splitext(fname)[1]
    if   ext == ".obj": return read_obj(fname, dtype=dtype, flatten=flatten)
    elif ext == ".ply": return read_ply(fname, dtype=dtype, flatten=flatten)
    else:
        raise NotImplementedError(f"Cannot read '{fname}'.")
