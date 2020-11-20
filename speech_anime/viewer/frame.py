
import os
import cv2
import math
import torch
import saber
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import deformation
from ..tools import FaceDataType
from ..datasets.vocaset.mask import non_face
try:
    from . import render_cpp as renderer
except ImportError:
    from . import render_py as renderer


_template_verts, _template_faces = None, None
_template_c_indices = []
_template_corres = dict(
    corr_count = [],
    corr_faces = []
)


def set_dgrad_static(verts, faces, c_indices=None, corres=None):
    global _template_verts
    global _template_faces
    global _template_c_indices
    _template_verts = verts
    _template_faces = faces
    _template_c_indices = non_face.non_face_verts if c_indices is None else c_indices
    if corres is not None:
        for key in _template_corres:
            _template_corres[key] = deepcopy(corres[key])
    else:
        _template_corres['corr_count'] = []
        _template_corres['corr_faces'] = []

    saber.log.info("deformation.set_target")
    deformation.set_target(
        verts=np.reshape(verts, (-1, 3)),
        faces=np.reshape(faces, (-1, 3)),
        cnsts=_template_c_indices,
        corrs=_template_corres['corr_count']
    )


def set_template_mesh(template_path, constraints_path=None, corres_path=None):
    verts, faces = saber.mesh.read_mesh(template_path, dtype=np.float32)

    # read optional constraints and corres
    c_indices, corres = None, None
    if constraints_path is not None:
        with open(constraints_path) as fp:
            line = ' '.join(x.strip() for x in fp.readlines())
            c_indices = [int(x) for x in line.split()]
    if corres_path is not None:
        # get corres
        corres_dict = {}
        with open(corres_path) as fp:
            count = 0
            for i, line in enumerate(fp):
                if i == 0:
                    count = int(line.strip())
                    continue
                if count == 0:
                    break
                src_i, dst_i, _ = line.strip().split(",")
                src_i = int(src_i)
                dst_i = int(dst_i)
                if dst_i not in corres_dict:
                    corres_dict[dst_i] = []
                corres_dict[dst_i].append(src_i)
                count -= 1
        corres_count = []
        corres_faces = []
        for i in range(len(faces)):
            if i not in corres_dict:
                corres_count.append(0)
                corres_faces += [0]
            else:
                corrs = corres_dict[i]
                corres_count.append(len(corrs))
                corres_faces += corrs
        corres = dict(
            corr_count = corres_count,
            corr_faces = corres_faces,
        )

    # set static
    set_dgrad_static(verts, faces, c_indices, corres)

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
            c_indices = _template_c_indices
            if not deformation.is_same(
                _template_verts.shape[0],
                _template_faces.shape[0],
                len(c_indices)
            ):
                saber.log.info("deformation.set_target")
                deformation.set_target(
                    verts=np.reshape(_template_verts, (-1, 3)),
                    faces=np.reshape(_template_faces, (-1, 3)),
                    cnsts=c_indices
                )
            # get mesh
            vert_cnsts = []
            if c_indices is not None and len(c_indices) > 0:
                vert_cnsts = np.reshape(_template_verts, (-1, 3))[c_indices]
            data_frame = deformation.get_mesh(
                deform_grad=data_frame.astype(np.float64),
                vert_cnsts=vert_cnsts,
                **_template_corres
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
