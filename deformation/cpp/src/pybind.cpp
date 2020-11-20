#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include "deform.hpp"
#include "log.hpp"

namespace py = pybind11;
using namespace pybind11::literals;

deformation::TriangleDeformation gDeformManager;


bool SetTarget(
    const py::array_t<float,    py::array::c_style> & verts,
    const py::array_t<uint32_t, py::array::c_style> & faces,
    const py::array_t<uint32_t, py::array::c_style> & cnsts,
    const py::array_t<uint32_t, py::array::c_style> & corrs,
    double reg
) {
    assertion(1 <= verts.ndim() && verts.ndim() <= 2);
    assertion(1 <= faces.ndim() && faces.ndim() <= 2);
    assertion(cnsts.ndim() == 1);
    int n_verts = verts.shape(0); if (verts.ndim() == 1) { n_verts /= 3; }
    int n_tris = faces.shape(0); if (faces.ndim() == 1) { n_tris /= 3; }
    int n_cnsts = cnsts.shape(0);
    return gDeformManager.setStaticTarget(
        verts.data(), n_verts,
        faces.data(), n_tris,
        cnsts.data(), n_cnsts,
        (corrs.size() > 0) ? corrs.data() : nullptr,
        reg
    );
}

/* deformation matrix */

py::array_t<double, py::array::c_style> GetDeformMat(
    const py::array_t<float,    py::array::c_style> & vertsA,
    const py::array_t<float,    py::array::c_style> & vertsB,
    const py::array_t<uint32_t, py::array::c_style> & faces,
    double eps=1e-6
)
{
    assertion(1 <= vertsA.ndim() && vertsA.ndim() <= 2);
    assertion(1 <= vertsB.ndim() && vertsB.ndim() <= 2);
    assertion(1 <= faces.ndim()  && faces.ndim() <= 2);
    assertion(vertsA.size() == vertsB.size());
    int n_verts = vertsA.shape(0); if (vertsA.ndim() == 1) { n_verts /= 3; }
    int n_tris = faces.shape(0);  if (faces.ndim() == 1)  { n_tris /= 3; }
    py::array_t<double, py::array::c_style> ret(n_tris*3*3);
    gDeformManager.getDeformationMatrix(
        ret.mutable_data(),
        vertsA.data(), vertsB.data(), n_verts,
        faces.data(), n_tris,
        eps
    );
    return ret;
}

py::array_t<float, py::array::c_style> GetMeshFromMat(
    const py::array_t<double, py::array::c_style> & dMat,
    const py::array_t<float,  py::array::c_style> & cVerts
)
{
    py::array_t<float, py::array::c_style> verts({gDeformManager.n_tar_verts(), (size_t)3});
    gDeformManager.getMeshFromDeformationMatrix(
        verts.mutable_data(),
        dMat.data(),
        (cVerts.size() == 0) ? nullptr : cVerts.data(),
        (uint32_t *) nullptr,
        (uint32_t *) nullptr
    );
    return verts;
}

/* deformation gradient */

py::array_t<double, py::array::c_style> GetDeformGrad(
    const py::array_t<float,    py::array::c_style> & vertsA,
    const py::array_t<float,    py::array::c_style> & vertsB,
    const py::array_t<uint32_t, py::array::c_style> & faces,
    double eps=1e-6
)
{
    assertion(1 <= vertsA.ndim() && vertsA.ndim() <= 2);
    assertion(1 <= vertsB.ndim() && vertsB.ndim() <= 2);
    assertion(1 <= faces.ndim()  && faces.ndim() <= 2);
    assertion(vertsA.size() == vertsB.size());
    int n_verts = vertsA.shape(0); if (vertsA.ndim() == 1) { n_verts /= 3; }
    int n_tris = faces.shape(0);  if (faces.ndim() == 1)  { n_tris /= 3; }
    py::array_t<double, py::array::c_style> ret(n_tris*3*3);
    gDeformManager.getDeformationGradients(
        ret.mutable_data(),
        vertsA.data(), vertsB.data(), n_verts,
        faces.data(), n_tris,
        eps
    );
    return ret;
}

py::array_t<float, py::array::c_style> GetMeshFromGrad(
    const py::array_t<double,   py::array::c_style> & dGrad,
    const py::array_t<float,    py::array::c_style> & cVerts,
    const py::array_t<uint32_t, py::array::c_style> & countCorrs,
    const py::array_t<uint32_t, py::array::c_style> & triCorrs
)
{
    py::array_t<float, py::array::c_style> verts({gDeformManager.n_tar_verts(), (size_t)3});
    gDeformManager.getMeshFromDeformationGradients(
        verts.mutable_data(),
        dGrad.data(),
        (cVerts.size() == 0) ? nullptr : cVerts.data(),
        (countCorrs.size() == 0) ? nullptr : countCorrs.data(),
        (countCorrs.size() == 0) ? nullptr : triCorrs.data()
    );
    return verts;
}

bool IsSame(int n_verts, int n_tris, int n_cnsts)
{
    return (
        gDeformManager.n_tar_verts() == n_verts &&
        gDeformManager.n_tar_tris() == n_tris &&
        gDeformManager.n_tar_cnsts() == n_cnsts
    );
}


PYBIND11_MODULE(deformation, m)
{
    m.doc() = "pybind11 deformation transfer";
    m.def("set_target",       &SetTarget,       "verts"_a, "faces"_a,
                                                "cnsts"_a = py::array_t<uint32_t, py::array::c_style>(),
                                                "corrs"_a = py::array_t<uint32_t, py::array::c_style>(),
                                                "reg"_a   = 1e-10);
    m.def("is_same",          &IsSame,          "num_verts"_a, "num_faces"_a, "num_cnsts"_a);

    // for matrix
    m.def("get_deform_mat",   &GetDeformMat,    "verts_a"_a, "verts_b"_a, "faces"_a, "eps"_a=1e-6);
    m.def("get_mesh_from_dm", &GetMeshFromMat,  "deform_mat"_a,
                                                "vert_cnsts"_a = py::array_t<float, py::array::c_style>());

    // for gradient
    m.def("get_deform_grad",  &GetDeformGrad,   "verts_a"_a, "verts_b"_a, "faces"_a, "eps"_a=1e-6);
    m.def("get_mesh_from_dg", &GetMeshFromGrad, "deform_grad"_a,
                                                "vert_cnsts"_a = py::array_t<float, py::array::c_style>(),
                                                "corr_count"_a = py::array_t<uint32_t, py::array::c_style>(),
                                                "corr_faces"_a = py::array_t<uint32_t, py::array::c_style>());
    m.def("get_mesh",         &GetMeshFromGrad, "deform_grad"_a,
                                                "vert_cnsts"_a = py::array_t<float, py::array::c_style>(),
                                                "corr_count"_a = py::array_t<uint32_t, py::array::c_style>(),
                                                "corr_faces"_a = py::array_t<uint32_t, py::array::c_style>());
}
