#include "deform.h"
#include "log.h"
#include "rotation/utils_rotation.h"
#include <fstream>

#define  EPSILON 0.0001

void WriteMatrix(std::string filename, const Eigen::MatrixXd &mat)
{
    std::cout << filename << ": " << mat.rows() << " x " << mat.cols() << std::endl;
    std::ofstream fout(filename);
    fout << mat << std::endl;
    fout.close();

}

bool Deformation::setTarget(
    const float    * verts,           int numVerts,
    const uint32_t * faces,           int numFaces,
    const uint32_t * vertCnstIndices, int numCnsts,
    double           regTerm
)
{
    _reset();
    mNumVerts = numVerts;
    mNumFaces = numFaces;
    mNumCnsts = numCnsts;
    mDeformMatrix.resize(3 * mNumFaces, 3);
    mCnstsMatrix.resize(std::max(mNumCnsts, 1), 3);

    A.resize(3 * mNumFaces, mNumVerts-mNumCnsts);
    Ar.resize(3 * mNumFaces, std::max(mNumCnsts, 1));
    std::vector<Eigen::Triplet<double>> tripletsA;
    std::vector<Eigen::Triplet<double>> tripletsAr;

    // prepare mapping vert index in matrix
    mVi2ACol.resize(mNumVerts);
    mVi2ArCol.resize(mNumVerts);
    mACol2Vi.resize(mNumVerts-mNumCnsts);
    mArCol2Vi.resize(mNumCnsts);
    std::fill(mVi2ACol.begin(),  mVi2ACol.end(),  -1);
    std::fill(mVi2ArCol.begin(), mVi2ArCol.end(), -1);
    std::fill(mACol2Vi.begin(),  mACol2Vi.end(),  -1);
    std::fill(mArCol2Vi.begin(), mArCol2Vi.end(), -1);

    // first, vi -> col
    for (int i = 0; i < mNumVerts; ++i)
    {
        mVi2ACol[i] = i;
    }
    for (int i = 0; i < mNumCnsts; ++i)
    {
        auto ci = vertCnstIndices[i];
        mVi2ACol[ci] = -1; // critical!
        mVi2ArCol[ci] = i;
        for (int j = ci + 1; j < mNumVerts; ++j)
        {
            if (mVi2ACol[j] >= 0) mVi2ACol[j] -= 1;
        }
    }
    // then, col -> vi
    for (int i = 0; i < mNumVerts; ++i)
    {
        assertion((mVi2ACol[i] < 0) ^ (mVi2ArCol[i] < 0), "impossible, vert {} appear in A and Ar!", i);
        if (mVi2ACol[i] >= 0)
        {
            auto col = mVi2ACol[i];
            assertion(mACol2Vi[col] < 0,
                      "A col {} set as {} before!",
                      col, mACol2Vi[col]);  // must be not set before!
            mACol2Vi[col] = i;
        }
        else
        {
            auto col = mVi2ArCol[i];
            assertion(mArCol2Vi[col] < 0,
                      "Ar col {} set as {} before!",
                      col, mArCol2Vi[col]);  // must be not set before!
            mArCol2Vi[col] = i;
        }
    }

    auto _push_triplet = [&](int r, int vi, double v) -> void
    {
        if (mVi2ACol[vi] >= 0) { tripletsA.push_back (Eigen::Triplet<double>(r, mVi2ACol[vi],  v)); }
        else                   { tripletsAr.push_back(Eigen::Triplet<double>(r, mVi2ArCol[vi], v)); }
    };

    Eigen::MatrixXd Va(3,2);
    Eigen::MatrixXd Uj(2,3);
    Eigen::MatrixXd Q(3,2);
    Eigen::MatrixXd R(2,2);
    for (int j = 0; j < mNumFaces; j++)
    {
        int fi  = j * 3;
        int vi1 = faces[fi + 0];
        int vi2 = faces[fi + 1];
        int vi3 = faces[fi + 2];

        Eigen::Vector3f v1(verts + vi1 * 3);
        Eigen::Vector3f v2(verts + vi2 * 3);
        Eigen::Vector3f v3(verts + vi3 * 3);

        _setMatrixCol(Va, v2-v1, 0);
        _setMatrixCol(Va, v3-v1, 1);
        _qrFactorize(Va, Q, R);

        Uj = R.inverse() * Q.transpose();

        _push_triplet(fi + 0, vi1, -Uj(0,0) - Uj(1,0));
        _push_triplet(fi + 1, vi1, -Uj(0,1) - Uj(1,1));
        _push_triplet(fi + 2, vi1, -Uj(0,2) - Uj(1,2));

        _push_triplet(fi + 0, vi2, Uj(0,0));
        _push_triplet(fi + 1, vi2, Uj(0,1));
        _push_triplet(fi + 2, vi2, Uj(0,2));

        _push_triplet(fi + 0, vi3, Uj(1,0));
        _push_triplet(fi + 1, vi3, Uj(1,1));
        _push_triplet(fi + 2, vi3, Uj(1,2));
    }

    A.setFromTriplets(tripletsA.begin(), tripletsA.end());
    Ar.setFromTriplets(tripletsAr.begin(), tripletsAr.end());

    At = A.transpose();
    AtA = At * A;


    // debug("Got A, At, AtA");

    // Regularize matrix
    if (regTerm !=0)
    {
        for (int i = 0; i < AtA.rows(); i++)
        {
            AtA.coeffRef(i,i) += regTerm;
        }
    }

    mSolver.compute(AtA);
    if(mSolver.info()!=Eigen::Success) 
    {
        // decomposition failed
        error("solver error: {}", mSolver.lastErrorMessage());
        return false;
    }

    return true;
}

/* deformation matrix */

bool Deformation::getDeformMat(
          double *   dmat,
    const float *    vertsA,
    const float *    vertsB,
            int      numVerts,
    const uint32_t * faces,
            int      numFaces,
            double   eps)
{
    auto _get_e3 = [=](const Eigen::Vector3d & e1, const Eigen::Vector3d & e2, Eigen::Vector3d & e3) -> bool
    {
        e3 = e1.cross(e2);
        auto _len1 = std::pow(e1.dot(e1), 0.5);
        auto _len2 = std::pow(e2.dot(e2), 0.5);
        auto abs_cos_theta = std::abs(e1.dot(e2) / (_len1*_len2));
        if (abs_cos_theta > (1.0 - eps)) return false;
        e3 /= std::max(std::pow(e3.dot(e3), 0.25), eps);
        return true;
    };

    auto _fill_T = [=](int row, const Eigen::Matrix3d & mat) -> void
    {
        dmat[(row+0)*3+0] = mat(0, 0); dmat[(row+0)*3+1] = mat(0, 1); dmat[(row+0)*3+2] = mat(0, 2);
        dmat[(row+1)*3+0] = mat(1, 0); dmat[(row+1)*3+1] = mat(1, 1); dmat[(row+1)*3+2] = mat(1, 2);
        dmat[(row+2)*3+0] = mat(2, 0); dmat[(row+2)*3+1] = mat(2, 1); dmat[(row+2)*3+2] = mat(2, 2);
    };

    assertion(dmat != nullptr, "deformMat is not allocated!");
    Eigen::Matrix3d T;
    Eigen::Matrix3d matA, matB;
    Eigen::Vector3d ea3, eb3;
    for (int j = 0; j < numFaces; ++j)
    {
        int fi = j * 3;
        int vi1 = faces[fi + 0] * 3;
        int vi2 = faces[fi + 1] * 3;
        int vi3 = faces[fi + 2] * 3;
        // debug("face {}/{}: {}, {}, {}", j, numFaces, vi1, vi2, vi3);

        Eigen::Vector3d pa1((double)vertsA[vi1+0], (double)vertsA[vi1+1], (double)vertsA[vi1+2]);
        Eigen::Vector3d pa2((double)vertsA[vi2+0], (double)vertsA[vi2+1], (double)vertsA[vi2+2]);
        Eigen::Vector3d pa3((double)vertsA[vi3+0], (double)vertsA[vi3+1], (double)vertsA[vi3+2]);
        auto ea1 = pa2 - pa1;
        auto ea2 = pa3 - pa1;

        Eigen::Vector3d pb1((double)vertsB[vi1+0], (double)vertsB[vi1+1], (double)vertsB[vi1+2]);
        Eigen::Vector3d pb2((double)vertsB[vi2+0], (double)vertsB[vi2+1], (double)vertsB[vi2+2]);
        Eigen::Vector3d pb3((double)vertsB[vi3+0], (double)vertsB[vi3+1], (double)vertsB[vi3+2]);
        auto eb1 = pb2 - pb1;
        auto eb2 = pb3 - pb1;

        auto good_a = _get_e3(ea1, ea2, ea3);
        auto good_b = _get_e3(eb1, eb2, eb3);

        if (!(good_a && good_b))
        {
            T = Eigen::Matrix3d::Identity();
        }
        else
        {
            matA.col(0) = ea1; matA.col(1) = ea2; matA.col(2) = ea3;
            matB.col(0) = eb1; matB.col(1) = eb2; matB.col(2) = eb3;
            _getTransform(T, matA, matB);
            T.transposeInPlace();
        }

        _fill_T(fi, T);
    }
    return true;
}

bool Deformation::getMeshFromDeformMat(
    float        * retVerts,
    const double * deformMat,
    const float  * vertCnsts
)
{
    mDeformMatrix.setZero();
    for (int i = 0; i < 3 * mNumFaces; ++i)
    {
        mDeformMatrix.row(i) = Eigen::Vector3d(deformMat + i * 3);
    }

    // constraints
    if (mNumCnsts > 0)
    {
        assertion(vertCnsts != nullptr,
                  "vertCnsts is not given, but {} constraints.",
                  mNumCnsts);
        mCnstsMatrix.setZero();
        for (int i = 0, ci = 0; i < mNumCnsts; ++i, ci+=3)
        {
            mCnstsMatrix(i, 0) = (double)vertCnsts[ci+0];
            mCnstsMatrix(i, 1) = (double)vertCnsts[ci+1];
            mCnstsMatrix(i, 2) = (double)vertCnsts[ci+2];
        }
        mDeformMatrix -= Ar * mCnstsMatrix;
    }

    // solving
    Eigen::MatrixXd X = mSolver.solve(At * mDeformMatrix);
    if(mSolver.info()!=Eigen::Success) 
    {
        // solving failed
        std::cout << mSolver.lastErrorMessage() << std::endl;
        return false;
    }

    // set return vertices
    for (int i = 0; i < X.rows(); ++i)
    {
        int vi = mACol2Vi[i] * 3;
        retVerts[vi + 0] = (float)X(i, 0);
        retVerts[vi + 1] = (float)X(i, 1);
        retVerts[vi + 2] = (float)X(i, 2);
    }
    for (int i = 0; i < mNumCnsts; ++i)
    {
        int vi = mArCol2Vi[i] * 3;
        retVerts[vi + 0] = vertCnsts[i * 3 + 0];
        retVerts[vi + 1] = vertCnsts[i * 3 + 1];
        retVerts[vi + 2] = vertCnsts[i * 3 + 2];
    }
    return true;
}

/* deformation gradient */

bool Deformation::getDeformGrad(
    double * dGrad,
    const float * vertsA, const float * vertsB, int numVerts,
    const uint32_t * faces, int numFaces,
    double eps
)
{
    auto _get_e3 = [=](const Eigen::Vector3d & e1, const Eigen::Vector3d & e2, Eigen::Vector3d & e3) -> bool
    {
        e3 = e1.cross(e2);
        auto _len1 = std::pow(e1.dot(e1), 0.5);
        auto _len2 = std::pow(e2.dot(e2), 0.5);
        auto abs_cos_theta = std::abs(e1.dot(e2) / (_len1*_len2));
        if (abs_cos_theta > (1.0 - eps)) return false;
        e3 /= std::max(std::pow(e3.dot(e3), 0.25), eps);
        return true;
    };

    auto _fill_grad = [=](int row, const std::vector<double> & grad) -> void
    {
        memcpy(dGrad + row, grad.data(), 9 * sizeof(double));
    };

    assertion(dGrad != nullptr, "deformMat is not allocated!");
    Eigen::Matrix3d T;
    Eigen::Matrix3d matA, matB;
    Eigen::Vector3d ea3, eb3;
    std::vector<double> grad(9);
    std::vector<double> Iden(9, 0); Iden[0] = Iden[3] = Iden[5] = 1.0;  // 1 0 0 1 0 1 0 0 0
    for (int j = 0; j < numFaces; ++j)
    {
        int fi = j * 3;
        int vi1 = faces[fi + 0] * 3;
        int vi2 = faces[fi + 1] * 3;
        int vi3 = faces[fi + 2] * 3;
        // debug("face {}/{}: {}, {}, {}", j, numFaces, vi1, vi2, vi3);

        Eigen::Vector3d pa1((double)vertsA[vi1+0], (double)vertsA[vi1+1], (double)vertsA[vi1+2]);
        Eigen::Vector3d pa2((double)vertsA[vi2+0], (double)vertsA[vi2+1], (double)vertsA[vi2+2]);
        Eigen::Vector3d pa3((double)vertsA[vi3+0], (double)vertsA[vi3+1], (double)vertsA[vi3+2]);
        auto ea1 = pa2 - pa1;
        auto ea2 = pa3 - pa1;

        Eigen::Vector3d pb1((double)vertsB[vi1+0], (double)vertsB[vi1+1], (double)vertsB[vi1+2]);
        Eigen::Vector3d pb2((double)vertsB[vi2+0], (double)vertsB[vi2+1], (double)vertsB[vi2+2]);
        Eigen::Vector3d pb3((double)vertsB[vi3+0], (double)vertsB[vi3+1], (double)vertsB[vi3+2]);
        auto eb1 = pb2 - pb1;
        auto eb2 = pb3 - pb1;

        auto good_a = _get_e3(ea1, ea2, ea3);
        auto good_b = _get_e3(eb1, eb2, eb3);

        if (!(good_a && good_b))
        {
            grad = Iden;
        }
        else
        {
            matA.col(0) = ea1; matA.col(1) = ea2; matA.col(2) = ea3;
            matB.col(0) = eb1; matB.col(1) = eb2; matB.col(2) = eb3;
            _getTransform(T, matA, matB);
            _getGradFromMat(grad, T);
        }

        _fill_grad(j * 9, grad);
    }
    return true;
}

bool Deformation::getMeshFromDeformGrad(
    float        * retVerts,
    const double * dGrad,
    const float  * vertCnsts
)
{
    // need to transpose when filling!
    mDeformMatrix.setZero();
    Eigen::Matrix3d logR;
    Eigen::Matrix3d S;
    Eigen::Matrix3d T;
    for (int i = 0; i < mNumFaces; ++i)
    {
        // log r
        logR <<  0,               dGrad[9*i+6],    dGrad[9*i+7],
                -dGrad[9*i+6],    0,               dGrad[9*i+8],
                -dGrad[9*i+7],   -dGrad[9*i+8],    0;
        // scaling
        S    <<  dGrad[9*i+0]+1,  dGrad[9*i+1],    dGrad[9*i+2],
                 dGrad[9*i+1],    dGrad[9*i+3]+1,  dGrad[9*i+4],
                 dGrad[9*i+2],    dGrad[9*i+4],    dGrad[9*i+5]+1;

        T = rotation_log_exp::exp(logR) * S;

        mDeformMatrix.block<3, 3>(i*3, 0) = T.transpose();
    }

    // constraints
    if (mNumCnsts > 0)
    {
        assertion(vertCnsts != nullptr,
                  "vertCnsts is not given, but {} constraints.",
                  mNumCnsts);
        mCnstsMatrix.setZero();
        for (int i = 0, ci = 0; i < mNumCnsts; ++i, ci+=3)
        {
            mCnstsMatrix(i, 0) = (double)vertCnsts[ci+0];
            mCnstsMatrix(i, 1) = (double)vertCnsts[ci+1];
            mCnstsMatrix(i, 2) = (double)vertCnsts[ci+2];
        }
        mDeformMatrix -= Ar * mCnstsMatrix;
    }

    // solving
    Eigen::MatrixXd X = mSolver.solve(At * mDeformMatrix);
    if(mSolver.info()!=Eigen::Success) 
    {
        // solving failed
        std::cout << mSolver.lastErrorMessage() << std::endl;
        return false;
    }

    // set return vertices
    for (int i = 0; i < X.rows(); ++i)
    {
        int vi = mACol2Vi[i] * 3;
        retVerts[vi + 0] = (float)X(i, 0);
        retVerts[vi + 1] = (float)X(i, 1);
        retVerts[vi + 2] = (float)X(i, 2);
    }
    for (int i = 0; i < mNumCnsts; ++i)
    {
        int vi = mArCol2Vi[i] * 3;
        retVerts[vi + 0] = vertCnsts[i * 3 + 0];
        retVerts[vi + 1] = vertCnsts[i * 3 + 1];
        retVerts[vi + 2] = vertCnsts[i * 3 + 2];
    }
    return true;
}

void Deformation::_getTransform(Eigen::Matrix3d & T, const Eigen::Matrix3d & matA, const Eigen::Matrix3d & matB)
{
    T = matB * matA.inverse();
}

void Deformation::_getGradFromMat(std::vector<double> & grad, const Eigen::Matrix3d & T)
{
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(T, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::Matrix3d U,V;
    U=svd.matrixU();
    V=svd.matrixV();
    Eigen::Matrix3d S(svd.singularValues().asDiagonal());
    Eigen::Matrix3d Temp=Eigen::Matrix3d::Identity();
    Temp(2,2) = (U*V.transpose()).determinant();
    Eigen::Matrix3d R=U*Temp*V.transpose();
    Eigen::Matrix3d Scale = V*Temp*S*V.transpose();
    Eigen::Matrix3d logR = rotation_log_exp::log(R);

    grad.resize(9);
    grad[0] = Scale(0, 0) - 1;
    grad[1] = Scale(0, 1);
    grad[2] = Scale(0, 2);
    grad[3] = Scale(1, 1) - 1;
    grad[4] = Scale(1, 2);
    grad[5] = Scale(2, 2) - 1;
    grad[6] = logR(0, 1);
    grad[7] = logR(0, 2);
    grad[8] = logR(1, 2);
}

void Deformation::_reset() {}

void Deformation::_setMatrixCol(Eigen::MatrixXd & m, const Eigen::Vector3f & v, int iCol)
{
    m(0,iCol) = v(0,0);
    m(1,iCol) = v(1,0);
    m(2,iCol) = v(2,0);
}

void Deformation::_setMatrixBlock(Eigen::MatrixXd &mBig, Eigen::MatrixXd &mSmall, int iRow, int iCol)
{
    int r,c, rmax=mSmall.rows(), cmax=mSmall.cols();
    for (r=0; r<rmax; r++)
    {
        for (c=0; c<cmax; c++)
        {
            mBig(iRow+r, iCol+c) = mSmall(r,c);
        }
    }
}

void Deformation::_setSparseMatrixBlock(Eigen::SparseMatrix<double> &mBig, Eigen::MatrixXd &mSmall, int iRow, int iCol)
{
    int r,c, rmax=mSmall.rows(), cmax=mSmall.cols();
    for (r=0; r<rmax; r++)
    {
        for (c=0; c<cmax; c++)
        {
            mBig.coeffRef(iRow+r, iCol+c) = mSmall(r,c);
        }
    }
}

void Deformation::_qrFactorize(const Eigen::MatrixXd &a, Eigen::MatrixXd &q, Eigen::MatrixXd &r)
{
    int i,j, imax, jmax;
    imax = a.rows();
    jmax = a.cols();
    r.setZero();

    for (j=0; j<jmax; j++)
    {
        Eigen::VectorXd v (a.col(j));
        for (i=0; i<j; i++)
        {
            Eigen::VectorXd qi (q.col(i));
            r(i,j)= qi.dot(v);
            v = v - r(i,j)*qi;
        }
        float vv = (float)v.squaredNorm();
        float vLen = sqrtf(vv);
        if (vLen < EPSILON)
        {
            r(j,j) = 1;
            q.col(j).setZero();
        }
        else
        {
            r(j,j) = vLen;
            q.col(j) = v/vLen;
        }
    }
}
