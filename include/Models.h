#pragma once

#include <string>
#include <vector>
#include <utility>

#include <Utils.h>
#include <Eigen/Sparse>



namespace threeD
{

namespace unconstrained
{
    struct AnchorModel{
        AnchorModel(const std::vector<Eigen::Vector3d> pts_): pts(pts_) {}

        std::pair<Eigen::VectorXd, std::vector<Eigen::Triplet<double>>> compute(std::vector<int> AnchorIdxs, std::vector<int> ArcSegNums,
                                                                                std::vector<std::string> types, 
                                                                                const std::vector<Eigen::Matrix3d>& Us,
                                                                                const std::vector<Eigen::Vector3d>& ms,
                                                                                const std::vector<double>& hs,
                                                                                int m, int row_offset);

        void computeEach(Eigen::VectorXd& res, std::vector<int>& I, std::vector<int>& J, std::vector<double>& V,
                         int AnchorIdx, int ArcSegNum, std::string type, 
                         const Eigen::Matrix3d& U, const Eigen::Vector3d& m, const double& h, int i, int row_offset);
    private:
        const std::vector<Eigen::Vector3d> pts;
    };

    struct MeasurementModel{
        MeasurementModel(const std::vector<Eigen::Vector3d> pts_, const std::vector<Eigen::VectorXd> covs_): pts(pts_), covs(covs_) {}

        std::pair<Eigen::VectorXd, std::vector<Eigen::Triplet<double>>> compute(std::vector<std::pair<int,int>> intvs,
                                                                                const std::vector<Eigen::Matrix3d>& Us,
                                                                                const std::vector<Eigen::Vector3d>& ms,
                                                                                const std::vector<double>& hs,
                                                                                const std::vector<double>& ks,
                                                                                int m, int row_offset);
        

        void computeEach(Eigen::VectorXd& res, std::vector<int>& I, std::vector<int>& J, std::vector<double>& V,
                         int ArcSegNum, const Eigen::Matrix3d& U, const Eigen::Vector3d& m, const double& h, const double& k, int j, int row_offset);

        Eigen::Vector3d getResiduals(const Eigen::Matrix3d& U, const Eigen::Vector3d& m, const double& h, const double& k, int j);

        void getJacobians(Eigen::Matrix3d& m_jac, Eigen::Matrix3d& u_jac, Eigen::Vector3d& h_jac, Eigen::Vector3d& k_jac,
                          const Eigen::Matrix3d& U, const Eigen::Vector3d& m, const double& h, const double& k, int j);

    private:
        const std::vector<Eigen::Vector3d> pts;
        const std::vector<Eigen::VectorXd> covs;
    };

}

namespace constrained
{
    struct G0ContinuityModel
    {
        std::pair<Eigen::VectorXd, std::vector<Eigen::Triplet<double>>> compute(const std::vector<Eigen::Matrix3d>& Us,
                                                                                const std::vector<Eigen::Vector3d>& ms,
                                                                                const std::vector<double>& hs,                                                                                
                                                                                int m, int row_offset, const double& mu);

        void computeEach(Eigen::VectorXd& res, std::vector<int>& I, std::vector<int>& J, std::vector<double>& V, 
                         int frontArcSegNum, int backArcSegNum, 
                         const Eigen::Matrix3d& U1, const Eigen::Matrix3d& U2, const Eigen::Vector3d& m1, const Eigen::Vector3d& m2,
                         const double& h1, const double& h2,  int row_offset, const double& mu);

        Eigen::Vector3d getResiduals(const Eigen::Matrix3d& U1, const Eigen::Matrix3d& U2,
                                     const Eigen::Vector3d& m1, const Eigen::Vector3d& m2, const double& h1, const double& h2);

        void getJacobians(Eigen::Matrix3d& m1_jac, Eigen::Matrix3d& m2_jac, Eigen::Matrix3d& u1_jac, Eigen::Matrix3d& u2_jac,
                          Eigen::Vector3d& h1_jac, Eigen::Vector3d& h2_jac, const Eigen::Matrix3d& U1, const Eigen::Matrix3d& U2,
                          const Eigen::Vector3d& m1, const Eigen::Vector3d& m2, const double& h1, const double& h2);

    };

    struct G1ContinuityModel
    {
        std::pair<Eigen::VectorXd, std::vector<Eigen::Triplet<double>>> compute(const std::vector<Eigen::Matrix3d>& Us,
                                                                                const std::vector<Eigen::Vector3d>& ms,
                                                                                const std::vector<double>& hs,
                                                                                const std::vector<double>& ks,                                                                       
                                                                                int m, int row_offset, const double& mu);

        void computeEach(Eigen::VectorXd& res, std::vector<int>& I, std::vector<int>& J, std::vector<double>& V, 
                         int frontArcSegNum, int backArcSegNum, 
                         const Eigen::Matrix3d& U1, const Eigen::Matrix3d& U2, const Eigen::Vector3d& m1, const Eigen::Vector3d& m2,
                         const double& h1, const double& h2, const double& k1, const double& k2, int row_offset, const double& mu);

        void getResiduals(Eigen::Vector3d& res, double& f, double& g1, double& g2,
                          const Eigen::Matrix3d& U1, const Eigen::Matrix3d& U2, const Eigen::Vector3d& m1, const Eigen::Vector3d& m2,
                          const double& h1, const double& h2, const double& k1, const double& k2);

        Eigen::Matrix<double, 6, 1> getResiduals2(const Eigen::Matrix3d& U1, const Eigen::Matrix3d& U2, const Eigen::Vector3d& m1, const Eigen::Vector3d& m2,
                                                  const double& h1, const double& h2, const double& k1, const double& k2);

        void getJacobians(const double& h1, const double& h2, const double& k1, const double& k2, 
                          double& dfdh1, double& dfdh2, double& dfdk1, double& dfdk2,
                          double& dg1dh1, double& dg1dh2, double& dg1dk1, double& dg1dk2,
                          double& dg2dh1, double& dg2dh2, double& dg2dk1, double& dg2dk2);

        void getJacobians2(Eigen::Matrix<double, 6, 3>& m1_jac, Eigen::Matrix<double, 6, 3>& m2_jac, Eigen::Matrix<double, 6, 3>& u1_jac, Eigen::Matrix<double, 6, 3>& u2_jac,
                           Eigen::Matrix<double, 6, 1>& h1_jac, Eigen::Matrix<double, 6, 1>& h2_jac, Eigen::Matrix<double, 6, 1>& k1_jac, Eigen::Matrix<double, 6, 1>& k2_jac, 
                           const Eigen::Matrix3d& U1, const Eigen::Matrix3d& U2, const Eigen::Vector3d& m1, const Eigen::Vector3d& m2, 
                           const double& h1, const double& h2, const double& k1, const double& k2);
    };

    struct MinimumArcLengthModel
    {
        std::pair<Eigen::VectorXd, std::vector<Eigen::Triplet<double>>> compute(const std::vector<double>& hs,
                                                                                const std::vector<double>& ks,
                                                                                const std::vector<double>& slack,  
                                                                                const double& min_arc_length,                                                                     
                                                                                int m, int row_offset, const double& mu);
        
        void computeEach(Eigen::VectorXd& res, std::vector<int>& I, std::vector<int>& J, std::vector<double>& V,
                         int ArcSegNum, const double& h, const double& k, const double& slack, const double& min_arc_length, int row_offset, const double& mu, int n);

        void getResiduals(double& res, const double& h, const double& k, const double& slack, const double& min_arc_length);

        void getJacobians(const double& h, const double& k, const double& slack, const double& min_arc_length, double& drdh, double& drdk, double& drds);

    };
}

}

