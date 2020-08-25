
import os
import cv2
import math
import torch
import saber
import numpy as np
from tqdm import tqdm
from ..tools import FaceDataType
from ..datasets.vocaset.mask import non_face
# try:
#     from . import render_cpp as renderer
# except ImportError:
#     from . import render_py as renderer
from . import render_py as renderer


_template_verts, _template_faces = None, None


def set_dgrad_static(verts, faces):
    global _template_verts
    global _template_faces
    _template_verts = verts
    _template_faces = faces

    c_indices = non_face.non_face_verts
    saber.log.info("deformation.set_target")
    saber.mesh.deformation.set_target(
        verts=np.reshape(verts, (-1, 3)),
        faces=np.reshape(faces, (-1, 3)),
        cnsts=c_indices
    )


def set_template_mesh(template_path):
    verts, faces = saber.mesh.read_mesh(template_path, dtype=np.float32)
    set_dgrad_static(verts, faces)
    if renderer is not None:
        if os.path.splitext(template_path)[1] != '.obj':
            template_path = os.path.splitext(template_path)[0] + ".obj"
            saber.mesh.write_obj(template_path, verts, faces)
        renderer.set_template(template_path)


def frame_to_mesh(data_frame, face_data_type):
    # check inputs
    if torch.is_tensor(data_frame):
        data_frame = data_frame.detach().cpu().numpy()
    assert FaceDataType.is_mesh(face_data_type)
    if isinstance(face_data_type, str):
        face_data_type = FaceDataType[face_data_type]
    # for different data type
    if face_data_type == FaceDataType.dgrad_3d:
        assert _template_verts is not None and _template_faces is not None
        # only support verts
        data_frame = data_frame.flatten(order='C').astype(np.float32)
        assert len(data_frame) in [89784],\
            "[voca] given verts should be {}! not {}".format([89784], len(data_frame))
        if len(data_frame) == 89784:
            c_indices = non_face.non_face_verts
            if not saber.mesh.deformation.is_same(
                _template_verts.shape[0],
                _template_faces.shape[0],
                len(c_indices)
            ):
                saber.log.info("deformation.set_target")
                saber.mesh.deformation.set_target(
                    verts=np.reshape(_template_verts, (-1, 3)),
                    faces=np.reshape(_template_faces, (-1, 3)),
                    cnsts=c_indices
                )
            data_frame = saber.mesh.deformation.get_mesh(
                deform_grad=data_frame.astype(np.float64),
                vert_cnsts=np.reshape(_template_verts, (-1, 3))[c_indices]
            )
        return (
            np.reshape(data_frame,      (-1, 3)),
            np.reshape(_template_faces, (-1, 3))
        )
    elif face_data_type == FaceDataType.verts_off_3d:
        return (
            np.reshape(data_frame,      (-1, 3)) + _template_verts,
            np.reshape(_template_faces, (-1, 3))
        )
    elif face_data_type == FaceDataType.verts_pos_3d:
        return (
            np.reshape(data_frame,      (-1, 3)),
            np.reshape(_template_faces, (-1, 3))
        )
    else:
        raise NotImplementedError(f"{face_data_type} is not supported!")


def render_frame(frame, face_data_type, image_size: tuple = (512, 512)):
    # check frame type
    if torch.is_tensor(frame):
        frame = frame.detach().cpu().numpy()
    assert isinstance(frame, np.ndarray)
    # check face_data_type
    if isinstance(face_data_type, str):
        face_data_type = FaceDataType[face_data_type]
    # render
    if FaceDataType.is_mesh(face_data_type):
        assert renderer is not None
        verts, _ = frame_to_mesh(frame, face_data_type)
        img = renderer.render_mesh(verts, faces=None, image_size=image_size)
    else:
        raise NotImplementedError()
    # resize and return
    return img
