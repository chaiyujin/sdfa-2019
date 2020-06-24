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


with open("voca_eye.txt") as fp:
    eye_indices = [
        int(x)
        for x in fp.readline().split()
    ]

full_verts, full_faces = read_obj("voca.obj")
write_obj("remove_eye.obj", full_verts, full_faces, eye_indices)
