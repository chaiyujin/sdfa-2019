import tqdm
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


def get_deform_mat(verts_a, verts_b, faces, eps=1e-6):
    """ return the dg from verts_a to verts_b """
    assert verts_a.shape == verts_b.shape,\
        "Given verts are not in same shape!"
    eye = np.eye(3, dtype=np.float64)
    verts_a = verts_a.astype(np.float64)
    verts_b = verts_b.astype(np.float64)
    dg_all = np.zeros((3 * faces.shape[0], 3), dtype=np.float64)

    for fi, f in enumerate(faces):
        i = fi * 3
        v0, v1, v2 = f

        v_src = np.zeros((3, 3))
        v_src[:, 0] = verts_a[v1] - verts_a[v0]
        v_src[:, 1] = verts_a[v2] - verts_a[v0]

        v_tar = np.zeros((3, 3))
        v_tar[:, 0] = verts_b[v1] - verts_b[v0]
        v_tar[:, 1] = verts_b[v2] - verts_b[v0]

        src_e2, small_src = _get_e2(v_src[:, 0], v_src[:, 1], eps=eps)
        tar_e2, small_tar = _get_e2(v_tar[:, 0], v_tar[:, 1], eps=eps)

        if small_src or small_tar:
            dg_all[i: i+3] = eye
        else:
            v_src[:, 2] = src_e2
            v_tar[:, 2] = tar_e2
            dg = np.matmul(v_tar[:], np.linalg.inv(v_src[:]))
            dg_all[i: i+3] = dg.T
    return dg_all


def _get_e2(e0, e1, eps):
    _edge = np.cross(e0, e1)
    _len0 = np.linalg.norm(e0)
    _len1 = np.linalg.norm(e1)
    abs_cos_theta = np.abs(np.dot(e0, e1) / (_len0 * _len1))
    small = int(abs_cos_theta > (1.0 - eps))
    if small == 1:
        return _edge, small
    _len2 = np.sqrt(np.linalg.norm(_edge))
    _edge /= _len2
    return _edge, small


def _get_part_A(tri_a):
    s1, s2, s3 = tri_a
    v = np.zeros((3, 2))
    v[:, 0] = s2 - s1
    v[:, 1] = s3 - s1
    q, r = np.linalg.qr(v)
    v = np.matmul(np.linalg.inv(r), q.T)
    v = v.T
    col0 = np.expand_dims(- v[:, 0] - v[:, 1], axis=1)
    return np.concatenate((col0, v), axis=1)


AtA = None
At  = None
Vc  = None
constrain_mask = None
constrain_indices = None


def set_target(verts, faces, cnsts=None, reg=1e-10):
    global AtA
    global At
    global Vc
    global constrain_indices
    global constrain_mask

    if verts.ndim == 1:
        verts = np.reshape(verts, (-1, 3))
    if faces.ndim == 1:
        faces = np.reshape(faces, (-1, 3))

    A = np.zeros((3 * faces.shape[0], verts.shape[0]))
    for i, fids in enumerate(faces):
        v0, v1, v2 = fids
        a = _get_part_A(np.asarray([verts[v0], verts[v1], verts[v2]]))
        A[i*3: i*3+3, v0] = a[:, 0]
        A[i*3: i*3+3, v1] = a[:, 1]
        A[i*3: i*3+3, v2] = a[:, 2]
        # quit()

    # constrains
    Vc, c_mask = None, None
    if cnsts is not None and len(cnsts) > 0:
        c_mask = np.asarray([(i in cnsts) for i in range(A.shape[1])])
        Vc = np.copy(A[:, c_mask])
        A = A[:, ~c_mask]

    # make sparse
    At = A.T
    AtA = np.matmul(At, A)
    reg = np.eye(AtA.shape[0]) * reg
    reg[-1, -1] = 0
    AtA = AtA + reg
    AtA = sparse.csr_matrix(AtA)
    At = sparse.csr_matrix(At)
    Vc = sparse.csr_matrix(Vc)
    constrain_indices = cnsts
    constrain_mask = c_mask


def get_mesh_from_dm(dmat, vert_cnsts=None):
    dmat = dmat.astype(np.float64)
    if dmat.ndim == 1:
        dmat = np.reshape(dmat, (-1, 3))
    c_verts = None
    c_indices = constrain_indices
    if c_indices is not None:
        c_verts = vert_cnsts
        assert Vc is not None
        assert Vc.shape[1] == c_verts.shape[0]
        assert constrain_mask is not None
        assert c_verts is not None
        dmat = dmat - Vc.dot(c_verts)

    # solve
    AtC = At.dot(dmat)
    x = spsolve(AtA, AtC[:, 0])
    y = spsolve(AtA, AtC[:, 1])
    z = spsolve(AtA, AtC[:, 2])
    ret = np.stack([x, y, z], axis=1).astype(np.float32)

    if c_indices is not None:
        verts = np.zeros((ret.shape[0] + c_verts.shape[0], 3), np.float32)
        verts[~constrain_mask] = ret
        verts[constrain_mask] = c_verts.astype(np.float32)
        return verts
    else:
        return ret
