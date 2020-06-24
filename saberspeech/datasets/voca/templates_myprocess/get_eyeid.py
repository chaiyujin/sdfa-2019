import os
import numpy as np


def read_obj(obj_path):
    verts = []
    faces = []
    with open(obj_path) as fp:
        for line in fp:
            line = line.strip().split()
            if len(line) == 0:
                continue
            if line[0] == "v":
                verts.append(np.asarray([float(line[1]), float(line[2]), float(line[3])], np.float32))
            if line[0] == "f":
                faces.append(np.asarray([int(line[1]), int(line[2]), int(line[3])], np.int32) - 1)
    return verts, faces


def write_obj(obj_path, verts, faces, zero_indices):
    with open(obj_path, "w") as fp:
        for vi, vert in enumerate(verts):
            use_zero = vi in zero_indices
            v0 = vert[0] if not use_zero else 0
            v1 = vert[1] if not use_zero else 0
            v2 = vert[2] if not use_zero else 0
            fp.write(f"v {v0} {v1} {v2}\n")
        for face in faces:
            fp.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


full_verts, full_faces = read_obj("voca.obj")
part_verts, _ = read_obj("voca_eye.obj")
print(len(full_verts), len(part_verts))

# find the indices in original verts
indices = []
faces_idx = []
for vert in part_verts:
    found = False
    for vi, full in enumerate(full_verts):
        if np.all(vert == full):
            indices.append(vi)
            found = True
            break
    assert found
indices = sorted(indices)
for fi, face in enumerate(full_faces):
    f0 = face[0] in indices
    f1 = face[1] in indices
    f2 = face[2] in indices
    assert (f0 and f1 and f2) or not (f0 or f1 or f2)
    if f0 and f1 and f2:
        faces_idx.append(fi)
faces_idx = sorted(faces_idx)

print(len(indices))
with open("voca_eye.txt", "w") as fp:
    fp.write(" ".join(str(x) for x in indices))
print(len(faces_idx))
with open("voca_eye_faces.txt", "w") as fp:
    fp.write(" ".join(str(x) for x in faces_idx))


write_obj("remove_eye.obj", full_verts, full_faces, indices)
