#pragma once
#include "deform_triangle.hpp"
#include "rotation/utils_rotation.h"

namespace deformation {

template <typename T, typename Int>
bool TriangleDeformation::setStaticTarget(
    const T      * _verts,             const size_t _n_verts,
    const Int    * _tris,              const size_t _n_tris,
    const Int    * _cnst_vert_indices, const size_t _n_cnsts,
    const Int    * _n_corres_each_tri,
    const double   _reg_term
) {
    _reset();

    // get total face constraints
    size_t n_equations = 0;
    for (int i = 0; i < _n_tris; ++i)
    {
        n_equations += (_n_corres_each_tri) ? std::max((size_t)1, (size_t)_n_corres_each_tri[i]) : (size_t)1;
    }

    n_tar_verts_ = _n_verts;
    n_tar_tris_  = _n_tris;
    n_tar_cnsts_ = _n_cnsts;
    mat_deform_.resize(3 * n_equations, 3);
    mat_cnsts_.resize(std::max(n_tar_cnsts_, (size_t)1), 3);

    A_ .resize(3 * n_equations, n_tar_verts_-n_tar_cnsts_);
    Ar_.resize(3 * n_equations, std::max(n_tar_cnsts_, (size_t)1));
    std::vector<Eigen::Triplet<double>> tripletsA;
    std::vector<Eigen::Triplet<double>> tripletsAr;

    // prepare mapping vert index in matrix
    vi_to_col_A_ .resize(n_tar_verts_);
    vi_to_col_Ar_.resize(n_tar_verts_);
    col_to_vi_A_ .resize(n_tar_verts_-n_tar_cnsts_);
    col_to_vi_Ar_.resize(n_tar_cnsts_);
    std::fill(vi_to_col_A_ .begin(), vi_to_col_A_ .end(), -1);
    std::fill(vi_to_col_Ar_.begin(), vi_to_col_Ar_.end(), -1);
    std::fill(col_to_vi_A_ .begin(), col_to_vi_A_ .end(), -1);
    std::fill(col_to_vi_Ar_.begin(), col_to_vi_Ar_.end(), -1);

    // first, vi -> col
    for (int i = 0; i < n_tar_verts_; ++i) { vi_to_col_A_[i] = i; }
    for (int i = 0; i < n_tar_cnsts_; ++i)
    {
        // ! important: remove the constraint vertex from A into Ar
        auto ci = _cnst_vert_indices[i];
        vi_to_col_A_ [ci] = -1;
        vi_to_col_Ar_[ci] = i;
        for (int j = ci + 1; j < n_tar_verts_; ++j) {
            if (vi_to_col_A_[j] >= 0) vi_to_col_A_[j] -= 1;
        }
    }
    // then, col -> vi
    for (int i = 0; i < n_tar_verts_; ++i)
    {
        assertion((vi_to_col_A_[i] < 0) ^ (vi_to_col_Ar_[i] < 0), "! impossible, vert {} appear in A and Ar!", i);
        if (vi_to_col_A_[i] >= 0)
        {
            auto col = vi_to_col_A_[i];
            assertion(col_to_vi_A_[col] < 0, "A col {} set as {} before!", col, col_to_vi_A_[col]);
            col_to_vi_A_[col] = i;
        }
        else
        {
            auto col = vi_to_col_Ar_[i];
            assertion(col_to_vi_Ar_[col] < 0, "Ar col {} set as {} before!", col, col_to_vi_Ar_[col]);
            col_to_vi_Ar_[col] = i;
        }
    }

    auto _pushTriplet = [&](int r, int vi, double v) -> void
    {
        if (vi_to_col_A_[vi] >= 0) { tripletsA .push_back(Eigen::Triplet<double>(r, vi_to_col_A_ [vi], v)); }
        else                       { tripletsAr.push_back(Eigen::Triplet<double>(r, vi_to_col_Ar_[vi], v)); }
    };

    Eigen::MatrixXd Va(3, 2);
    Eigen::MatrixXd Uj(2, 3);
    Eigen::MatrixXd Q (3, 2);
    Eigen::MatrixXd R (2, 2);
    for (int j = 0, k = 0; j < n_tar_tris_; j++)
    {
        int fi  = j * 3;
        int vi1 = _tris[fi + 0];
        int vi2 = _tris[fi + 1];
        int vi3 = _tris[fi + 2];

        Eigen::Vector3f v1(_verts + vi1 * 3);
        Eigen::Vector3f v2(_verts + vi2 * 3);
        Eigen::Vector3f v3(_verts + vi3 * 3);

        _setMatrixCol(Va, v2-v1, 0);
        _setMatrixCol(Va, v3-v1, 1);
        _qrFactorize(Va, Q, R);

        Uj = R.inverse() * Q.transpose();

        size_t max_step = (_n_corres_each_tri) ? std::max((size_t)1, (size_t)_n_corres_each_tri[j]) : (size_t)1;
        for (size_t step = 0; step < max_step; ++step, ++k)
        {
            int fk = k * 3;
            _pushTriplet(fk + 0, vi1, -Uj(0,0) - Uj(1,0));
            _pushTriplet(fk + 1, vi1, -Uj(0,1) - Uj(1,1));
            _pushTriplet(fk + 2, vi1, -Uj(0,2) - Uj(1,2));

            _pushTriplet(fk + 0, vi2, Uj(0,0));
            _pushTriplet(fk + 1, vi2, Uj(0,1));
            _pushTriplet(fk + 2, vi2, Uj(0,2));

            _pushTriplet(fk + 0, vi3, Uj(1,0));
            _pushTriplet(fk + 1, vi3, Uj(1,1));
            _pushTriplet(fk + 2, vi3, Uj(1,2));
        }
    }

    A_ .setFromTriplets(tripletsA .begin(), tripletsA .end());
    Ar_.setFromTriplets(tripletsAr.begin(), tripletsAr.end());
    At_  = A_.transpose();
    AtA_ = At_ * A_;

    // Regularize matrix
    if (_reg_term !=0)
    {
        for (int i = 0; i < AtA_.rows(); ++i) {
            AtA_.coeffRef(i,i) += _reg_term;
        }
    }

    solver_.compute(AtA_);
    if(solver_.info()!=Eigen::Success) 
    {
        // decomposition failed
        error("solver error: {}", solver_.lastErrorMessage());
        return false;
    }

    return true;
}

template <typename T, typename U, typename Int>
bool TriangleDeformation::getDeformationGradients(
          T      * _dgrad,
    const U      * _src_verts,
    const U      * _dst_verts, const size_t _n_verts,
    const Int    * _tris,      const size_t _n_tris,
    const double   _eps
) {
    auto _getEdge3 = [&](const Eigen::Vector3d & e1, const Eigen::Vector3d & e2, Eigen::Vector3d & e3) -> bool
    {
        e3 = e1.cross(e2);
        auto len1 = std::pow(e1.dot(e1), 0.5);
        auto len2 = std::pow(e2.dot(e2), 0.5);
        auto abs_cos_theta = std::abs(e1.dot(e2) / (len1*len2));
        if (abs_cos_theta > (1.0 - _eps)) return false;
        e3 /= std::max(std::pow(e3.dot(e3), 0.25), _eps);
        return true;
    };

    auto _fillGrad = [=](int row, const std::vector<double> & grad) -> void
    {
        for (size_t i = 0; i < 9; ++i) {
            _dgrad[row + i] = (T)grad[i];
        }
    };

    assertion(_dgrad != nullptr, "mat_deform is not allocated!");
    Eigen::Matrix3d Ti;
    Eigen::Matrix3d mat_a, mat_b;
    Eigen::Vector3d ea3, eb3;
    std::vector<double> grad(9);
    std::vector<double> zero(9, 0);
    for (int j = 0; j < _n_tris; ++j)
    {
        int fi = j * 3;
        int vi1 = _tris[fi + 0] * 3;
        int vi2 = _tris[fi + 1] * 3;
        int vi3 = _tris[fi + 2] * 3;

        Eigen::Vector3d pa1((double)_src_verts[vi1+0], (double)_src_verts[vi1+1], (double)_src_verts[vi1+2]);
        Eigen::Vector3d pa2((double)_src_verts[vi2+0], (double)_src_verts[vi2+1], (double)_src_verts[vi2+2]);
        Eigen::Vector3d pa3((double)_src_verts[vi3+0], (double)_src_verts[vi3+1], (double)_src_verts[vi3+2]);
        auto ea1 = pa2 - pa1;
        auto ea2 = pa3 - pa1;

        Eigen::Vector3d pb1((double)_dst_verts[vi1+0], (double)_dst_verts[vi1+1], (double)_dst_verts[vi1+2]);
        Eigen::Vector3d pb2((double)_dst_verts[vi2+0], (double)_dst_verts[vi2+1], (double)_dst_verts[vi2+2]);
        Eigen::Vector3d pb3((double)_dst_verts[vi3+0], (double)_dst_verts[vi3+1], (double)_dst_verts[vi3+2]);
        auto eb1 = pb2 - pb1;
        auto eb2 = pb3 - pb1;

        auto good_a = _getEdge3(ea1, ea2, ea3);
        auto good_b = _getEdge3(eb1, eb2, eb3);

        if (!(good_a && good_b))
        {
            grad = zero;
        }
        else
        {
            mat_a.col(0) = ea1; mat_a.col(1) = ea2; mat_a.col(2) = ea3;
            mat_b.col(0) = eb1; mat_b.col(1) = eb2; mat_b.col(2) = eb3;
            _getTransform(Ti, mat_a, mat_b);
            _getGradFromMat(grad, Ti);
        }

        _fillGrad(j * 9, grad);
    }
    return true;
}

template <typename T, typename U, typename Int>
bool TriangleDeformation::getMeshFromDeformationGradients(
          U   * _ret_verts,
    const T   * _dgrad,
    const U   * _cnst_verts,
    const Int * _n_corrs_each_tris,
    const Int * _corrs_tris       
) {
    // need to transpose when filling!
    mat_deform_.setZero();

    auto _getTransform = [&](int i) -> Eigen::Matrix3d
    {
        Eigen::Matrix3d mat_log_r;
        Eigen::Matrix3d mat_s;
        Eigen::Matrix3d ret;
        // log r
        mat_log_r <<
            (double) 0,                 (double) _dgrad[9*i+6],    (double)_dgrad[9*i+7],
            (double)-_dgrad[9*i+6],     (double) 0,                (double)_dgrad[9*i+8],
            (double)-_dgrad[9*i+7],     (double)-_dgrad[9*i+8],    (double)0;
        // scaling
        mat_s << 
            (double)_dgrad[9*i+0]+1.0, (double)_dgrad[9*i+1],     (double)_dgrad[9*i+2],
            (double)_dgrad[9*i+1],     (double)_dgrad[9*i+3]+1.0, (double)_dgrad[9*i+4],
            (double)_dgrad[9*i+2],     (double)_dgrad[9*i+4],     (double)_dgrad[9*i+5]+1.0;

        ret = rotation_log_exp::exp(mat_log_r) * mat_s;
        return ret.transpose();
    };

    for (int i = 0, fi = 0; i < n_tar_tris_; ++i)
    {
        // no corres
        if (_n_corrs_each_tris == nullptr)
        {
            mat_deform_.block<3, 3>(i*3, 0) = _getTransform(i);
            ++fi;
        }
        // with corres: this triangle has corres
        else if (_n_corrs_each_tris[i] > 0)
        {
            for (int j = 0; j < _n_corrs_each_tris[i]; ++j)
            {
                mat_deform_.block<3, 3>(fi*3, 0) = _getTransform(_corrs_tris[fi]);
                ++fi;
            }
        }
        // with corres: no corres for this triangle, just identity
        else
        {
            mat_deform_.block<3, 3>(fi*3, 0) = Eigen::Matrix3d::Identity();
            ++fi;
        }
    }

    // constraints
    if (n_tar_cnsts_ > 0)
    {
        assertion(_cnst_verts != nullptr, "cnst_verts is not given, but {} constraints.", n_tar_cnsts_);
        mat_cnsts_.setZero();
        for (int i = 0, ci = 0; i < n_tar_cnsts_; ++i, ci+=3)
        {
            mat_cnsts_(i, 0) = (double)_cnst_verts[ci+0];
            mat_cnsts_(i, 1) = (double)_cnst_verts[ci+1];
            mat_cnsts_(i, 2) = (double)_cnst_verts[ci+2];
        }
        mat_deform_ -= Ar_ * mat_cnsts_;
    }

    // solving
    Eigen::MatrixXd X = solver_.solve(At_ * mat_deform_);
    if(solver_.info()!=Eigen::Success) 
    {
        // solving failed
        std::cout << solver_.lastErrorMessage() << std::endl;
        return false;
    }

    // set return vertices
    for (int i = 0; i < X.rows(); ++i)
    {
        int vi = col_to_vi_A_[i] * 3;
        _ret_verts[vi + 0] = (U)X(i, 0);
        _ret_verts[vi + 1] = (U)X(i, 1);
        _ret_verts[vi + 2] = (U)X(i, 2);
    }
    for (int i = 0; i < n_tar_cnsts_; ++i)
    {
        int vi = col_to_vi_Ar_[i] * 3;
        _ret_verts[vi + 0] = _cnst_verts[i * 3 + 0];
        _ret_verts[vi + 1] = _cnst_verts[i * 3 + 1];
        _ret_verts[vi + 2] = _cnst_verts[i * 3 + 2];
    }
    return true;
}


template <typename T, typename U, typename Int>
bool TriangleDeformation::getDeformationMatrix(
          T      * _dmat,
    const U      * _src_verts,
    const U      * _dst_verts,  const size_t _n_verts,
    const Int    * _tris,       const size_t _n_tris,
    const double   _eps
) {
    auto _getEdge3 = [&](const Eigen::Vector3d & e1, const Eigen::Vector3d & e2, Eigen::Vector3d & e3) -> bool
    {
        e3 = e1.cross(e2);
        auto _len1 = std::pow(e1.dot(e1), 0.5);
        auto _len2 = std::pow(e2.dot(e2), 0.5);
        auto abs_cos_theta = std::abs(e1.dot(e2) / (_len1*_len2));
        if (abs_cos_theta > (1.0 - _eps)) return false;
        e3 /= std::max(std::pow(e3.dot(e3), 0.25), _eps);
        return true;
    };

    auto _fillT = [=](int row, const Eigen::Matrix3d & mat) -> void
    {
        _dmat[(row+0)*3+0] = (T)mat(0, 0); _dmat[(row+0)*3+1] = (T)mat(0, 1); _dmat[(row+0)*3+2] = (T)mat(0, 2);
        _dmat[(row+1)*3+0] = (T)mat(1, 0); _dmat[(row+1)*3+1] = (T)mat(1, 1); _dmat[(row+1)*3+2] = (T)mat(1, 2);
        _dmat[(row+2)*3+0] = (T)mat(2, 0); _dmat[(row+2)*3+1] = (T)mat(2, 1); _dmat[(row+2)*3+2] = (T)mat(2, 2);
    };

    assertion(_dmat != nullptr, "mat_deform is not allocated!");
    Eigen::Matrix3d Ti;
    Eigen::Matrix3d matA, matB;
    Eigen::Vector3d ea3, eb3;
    for (int j = 0; j < _n_tris; ++j)
    {
        int fi = j * 3;
        int vi1 = _tris[fi + 0] * 3;
        int vi2 = _tris[fi + 1] * 3;
        int vi3 = _tris[fi + 2] * 3;
        // debug("face {}/{}: {}, {}, {}", j, n_tris, vi1, vi2, vi3);

        Eigen::Vector3d pa1((double)_src_verts[vi1+0], (double)_src_verts[vi1+1], (double)_src_verts[vi1+2]);
        Eigen::Vector3d pa2((double)_src_verts[vi2+0], (double)_src_verts[vi2+1], (double)_src_verts[vi2+2]);
        Eigen::Vector3d pa3((double)_src_verts[vi3+0], (double)_src_verts[vi3+1], (double)_src_verts[vi3+2]);
        auto ea1 = pa2 - pa1;
        auto ea2 = pa3 - pa1;

        Eigen::Vector3d pb1((double)_dst_verts[vi1+0], (double)_dst_verts[vi1+1], (double)_dst_verts[vi1+2]);
        Eigen::Vector3d pb2((double)_dst_verts[vi2+0], (double)_dst_verts[vi2+1], (double)_dst_verts[vi2+2]);
        Eigen::Vector3d pb3((double)_dst_verts[vi3+0], (double)_dst_verts[vi3+1], (double)_dst_verts[vi3+2]);
        auto eb1 = pb2 - pb1;
        auto eb2 = pb3 - pb1;

        auto good_a = _getEdge3(ea1, ea2, ea3);
        auto good_b = _getEdge3(eb1, eb2, eb3);

        if (!(good_a && good_b))
        {
            Ti = Eigen::Matrix3d::Identity();
        }
        else
        {
            matA.col(0) = ea1; matA.col(1) = ea2; matA.col(2) = ea3;
            matB.col(0) = eb1; matB.col(1) = eb2; matB.col(2) = eb3;
            _getTransform(Ti, matA, matB);
        }

        _fillT(fi, Ti);
    }
    return true;
}

template <typename T, typename U, typename Int>
bool TriangleDeformation::getMeshFromDeformationMatrix(
          U   * _ret_verts,
    const T   * _dmat,
    const U   * _cnst_verts,
    // TODO: corres
    const Int * _n_corrs_each_tris,
    const Int * _corrs_tris
) {
    mat_deform_.setZero();
    for (int i = 0; i < n_tar_tris_; ++i)
    {
        // mat_deform is row major, T treat it as col major, it's auto transpose
        Eigen::Matrix3d mat = Eigen::Matrix<T, 3, 3>(_dmat + i * 9).template cast<double>();
        mat_deform_.block<3, 3>(i*3, 0) = mat;
    }

    // constraints
    if (n_tar_cnsts_ > 0)
    {
        assertion(_cnst_verts != nullptr,
                  "cnst_verts is not given, but {} constraints.",
                  n_tar_cnsts_);
        mat_cnsts_.setZero();
        for (int i = 0, ci = 0; i < n_tar_cnsts_; ++i, ci+=3)
        {
            mat_cnsts_(i, 0) = (double)_cnst_verts[ci+0];
            mat_cnsts_(i, 1) = (double)_cnst_verts[ci+1];
            mat_cnsts_(i, 2) = (double)_cnst_verts[ci+2];
        }
        mat_deform_ -= Ar_ * mat_cnsts_;
    }

    // solving
    Eigen::MatrixXd X = solver_.solve(At_ * mat_deform_);
    if(solver_.info()!=Eigen::Success) 
    {
        // solving failed
        std::cout << solver_.lastErrorMessage() << std::endl;
        return false;
    }

    // set return vertices
    for (int i = 0; i < X.rows(); ++i)
    {
        int vi = col_to_vi_A_[i] * 3;
        _ret_verts[vi + 0] = (U)X(i, 0);
        _ret_verts[vi + 1] = (U)X(i, 1);
        _ret_verts[vi + 2] = (U)X(i, 2);
    }
    for (int i = 0; i < n_tar_cnsts_; ++i)
    {
        int vi = col_to_vi_Ar_[i] * 3;
        _ret_verts[vi + 0] = _cnst_verts[i * 3 + 0];
        _ret_verts[vi + 1] = _cnst_verts[i * 3 + 1];
        _ret_verts[vi + 2] = _cnst_verts[i * 3 + 2];
    }
    return true;
}


inline void TriangleDeformation::_getTransform(Eigen::Matrix3d & _ret, const Eigen::Matrix3d & _mat_a, const Eigen::Matrix3d & _mat_b)
{
    _ret.noalias() = _mat_b * _mat_a.inverse();
}

inline void TriangleDeformation::_getGradFromMat(std::vector<double> & _grad, const Eigen::Matrix3d & _mat)
{
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(_mat, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    Eigen::Matrix3d S(svd.singularValues().asDiagonal());
    Eigen::Matrix3d Temp = Eigen::Matrix3d::Identity();
    Temp(2, 2) = (U * V.transpose()).determinant();
    Eigen::Matrix3d R = U * Temp * V.transpose();
    Eigen::Matrix3d scale = V * Temp * S * V.transpose();
    Eigen::Matrix3d log_r = rotation_log_exp::log(R);

    _grad.resize(9);
    _grad[0] = scale(0, 0) - 1;
    _grad[1] = scale(0, 1);
    _grad[2] = scale(0, 2);
    _grad[3] = scale(1, 1) - 1;
    _grad[4] = scale(1, 2);
    _grad[5] = scale(2, 2) - 1;
    _grad[6] = log_r(0, 1);
    _grad[7] = log_r(0, 2);
    _grad[8] = log_r(1, 2);
}

inline void TriangleDeformation::_setMatrixCol(Eigen::MatrixXd & _m, const Eigen::Vector3f & _v, int _i_col)
{
    _m(0, _i_col) = _v(0, 0);
    _m(1, _i_col) = _v(1, 0);
    _m(2, _i_col) = _v(2, 0);
}

inline void TriangleDeformation::_qrFactorize(const Eigen::MatrixXd & _a, Eigen::MatrixXd & _q, Eigen::MatrixXd & _r)
{
    // constexpr double EPSILON = 0.0001;
    constexpr double EPSILON = 1e-6;

    int i,j, imax, jmax;
    imax = _a.rows();
    jmax = _a.cols();
    _r.setZero();

    for (j=0; j<jmax; j++)
    {
        Eigen::VectorXd v (_a.col(j));
        for (i=0; i<j; i++)
        {
            Eigen::VectorXd qi (_q.col(i));
            _r(i,j)= qi.dot(v);
            v = v - _r(i,j)*qi;
        }
        double vv = v.squaredNorm();
        double vLen = std::sqrt(vv);
        if (vLen < EPSILON)
        {
            _r(j,j) = 1;
            _q.col(j).setZero();
        }
        else
        {
            _r(j,j) = vLen;
            _q.col(j) = v/vLen;
        }
    }
}

}