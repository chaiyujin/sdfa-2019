#pragma once
#include <iostream>
#include <Eigen/Eigen>
#include "log.hpp"

namespace deformation {

/**
 * It's the implementation of [sumner etal. 2014](http://people.csail.mit.edu/sumner/research/deftransfer/).
 * TODO: selection of correspondence
 * */
class TriangleDeformation
{
private:
    size_t n_tar_verts_;
    size_t n_tar_tris_;
    size_t n_tar_cnsts_;
    // deformation matrix as {Ti}, input will be filled into this matrix.
    Eigen::MatrixXd mat_deform_;
    Eigen::MatrixXd mat_cnsts_;
    // mapping: indices of vertex <-> indices of column
    std::vector<int> vi_to_col_A_;
    std::vector<int> vi_to_col_Ar_;
    std::vector<int> col_to_vi_A_;
    std::vector<int> col_to_vi_Ar_;
    // matrix and solver
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver_;
    Eigen::SparseMatrix<double> A_, Ar_, At_, AtA_;  // Ar is used for vertex constraints, stands for 'A remained'.

    void _reset() {}
    void _setMatrixCol(Eigen::MatrixXd & _m, const Eigen::Vector3f & _v, int _i_col);
    void _qrFactorize(const Eigen::MatrixXd & _a, Eigen::MatrixXd & _q, Eigen::MatrixXd & _r);
    void _getTransform(Eigen::Matrix3d & _ret, const Eigen::Matrix3d & _a, const Eigen::Matrix3d & _b);
    void _getGradFromMat(std::vector<double> & _grad, const Eigen::Matrix3d & _mat);

public:

    size_t n_tar_verts() const { return n_tar_verts_; }
    size_t n_tar_tris () const { return n_tar_tris_;  }
    size_t n_tar_cnsts() const { return n_tar_cnsts_; }

    template <typename T, typename Int>
    bool setStaticTarget(
        const T      * _verts,                        const size_t _n_verts,
        const Int    * _tris,                         const size_t _n_tris,
        const Int    * _cnst_vert_indices  = nullptr, const size_t _n_cnsts = 0,
        const Int    * _n_corres_each_tri  = nullptr,
        const double   _reg_term           = 1e-10
    );

    template <typename T, typename U, typename Int>
    bool getDeformationGradients(
              T      * _dgrad,
        const U      * _src_verts,
        const U      * _dst_verts, const size_t _n_verts,
        const Int    * _tris,      const size_t _n_tris,
        const double   _eps = 1e-6
    );
    template <typename T, typename U, typename Int>
    bool getMeshFromDeformationGradients(
              U      * _ret_verts,
        const T      * _dgrad,
        const U      * _cnst_verts        = nullptr,
        const Int    * _n_corrs_each_tris = nullptr,
        const Int    * _corrs_tris        = nullptr
    );

    // just for test, should use get deformation gradients!
    template <typename T, typename U, typename Int>
    bool getDeformationMatrix(
              T      * _dmat,
        const U      * _src_verts,
        const U      * _dst_verts,  const size_t _n_verts,
        const Int    * _tris,       const size_t _n_tris,
        const double   _eps = 1e-6
    );
    template <typename T, typename U, typename Int>
    bool getMeshFromDeformationMatrix(
              U      * _ret_verts,
        const T      * _dmat,
        const U      * _cnst_verts        = nullptr,
        const Int    * _n_corrs_each_tris = nullptr,
        const Int    * _corrs_tris        = nullptr
    );
};

}
