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


full_verts, full_faces = read_obj("voca.obj")
part_verts, _ = read_obj("voca_nose.obj")
print(len(full_verts), len(part_verts))

# find the indices in original verts
indices = []
for vert in part_verts:
    found = False
    for vi, full in enumerate(full_verts):
        if np.all(vert == full):
            indices.append(vi)
            found = True
            break
    assert found
indices = sorted(indices)
non_indices = sorted([x for x in range(len(full_verts)) if x not in indices])
faces_idx = []
non_faces_idx = []
for fi, face in enumerate(full_faces):
    f0 = face[0] in indices
    f1 = face[1] in indices
    f2 = face[2] in indices
    if not (f0 or f1 or f2):
        non_faces_idx.append(fi)
    else:
        faces_idx.append(fi)
non_faces_idx = sorted(non_faces_idx)

print(len(indices))
with open("voca_nose.txt", "w") as fp:
    fp.write(" ".join(str(x) for x in indices))
print(len(faces_idx))
with open("voca_nose_faces.txt", "w") as fp:
    fp.write(", ".join(str(x) for x in faces_idx))

print(len(non_indices))
with open("voca_non_nose.txt", "w") as fp:
    fp.write(", ".join(str(x) for x in non_indices))
print(len(non_faces_idx))
with open("voca_non_nose_faces.txt", "w") as fp:
    fp.write(", ".join(str(x) for x in non_faces_idx))
