#pragma once
#include <iostream>
#include <Eigen/Eigen>

class Deformation
{
private:
    int mNumVerts, mNumFaces, mNumCnsts;
    Eigen::MatrixXd mDeformMatrix;
    Eigen::MatrixXd mCnstsMatrix;
    Eigen::SparseMatrix<double> A, Ar, At, AtA;
	Eigen::SparseLU<Eigen::SparseMatrix<double>> mSolver;
    std::vector<int> mVi2ACol;
    std::vector<int> mVi2ArCol;
    std::vector<int> mACol2Vi;
    std::vector<int> mArCol2Vi;

    void _reset();
    void _setMatrixCol(Eigen::MatrixXd & m, const Eigen::Vector3f & v, int iCol);
    void _setMatrixBlock(Eigen::MatrixXd & mBig, Eigen::MatrixXd & mSmall, int iRow, int iCol);
    void _setSparseMatrixBlock(Eigen::SparseMatrix<double> & mBig, Eigen::MatrixXd & mSmall, int iRow, int iCol);
    void _qrFactorize(const Eigen::MatrixXd & a, Eigen::MatrixXd & q, Eigen::MatrixXd & r);
    void _getTransform(Eigen::Matrix3d & T, const Eigen::Matrix3d & a, const Eigen::Matrix3d & b);
    void _getGradFromMat(std::vector<double> & grad, const Eigen::Matrix3d & T);

public:
    int numVerts() const { return mNumVerts; }
    int numFaces() const { return mNumFaces; }
    int numCnsts() const { return mNumCnsts; }
    bool setTarget(
        const float    * verts, int numVerts,
        const uint32_t * faces, int numFaces,
        const uint32_t * vertCnstIndices = nullptr,
        int              numCnsts        = 0,
        double           regTerm         = 1e-10
    );

    /* deformation matrix */

    bool getDeformMat(
        double         * deformMat,
        const float    * vertsA,
        const float    * vertsB, int numVerts,
        const uint32_t * faces,  int numFaces,
        double           eps=1e-6
    );
    bool getMeshFromDeformMat(
        float        * retVerts,
        const double * deformMat,
        const float  * vertCnsts = nullptr
    );
    
    /* deformation gradient */

    bool getDeformGrad(
        double         * deformGrad,
        const float    * vertsA,
        const float    * vertsB, int numVerts,
        const uint32_t * faces,  int numFaces,
        double           eps=1e-6
    );
    bool getMeshFromDeformGrad(
        float        * retVerts,
        const double * deformGrad,
        const float  * vertCnsts = nullptr
    );
};
