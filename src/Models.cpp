#include <Models.h>

namespace threeD
{

namespace unconstrained
{   
    // Compute Anchor model residual and jacobian 
    std::pair<Eigen::VectorXd, std::vector<Eigen::Triplet<double>>> AnchorModel::compute(std::vector<int> AnchorIdxs, std::vector<int> ArcSegNums, 
                                                                                         std::vector<std::string> types, 
                                                                                         const std::vector<Eigen::Matrix3d>& Us,
                                                                                         const std::vector<Eigen::Vector3d>& ms,
                                                                                         const std::vector<double>& hs,
                                                                                         int m, int row_offset)
    {
        // std::cout << "[Anchor Model]" << std::endl;

        Eigen::VectorXd res(m);
        std::vector<int> I; 
        std::vector<int> J;
        std::vector<double> V;

        for (int i=0; i<AnchorIdxs.size(); i++)
        {
            int seg_num = ArcSegNums[i];
            AnchorModel::computeEach(res, I, J, V, AnchorIdxs[i], seg_num, types[i], Us[seg_num], ms[seg_num], hs[seg_num], i, row_offset);
        }
            

        std::vector<Eigen::Triplet<double>> triplets;
        for (int i=0; i<I.size(); i++)
            triplets.push_back(Eigen::Triplet<double>(I[i], J[i], V[i]));
        
        std::pair<Eigen::VectorXd, std::vector<Eigen::Triplet<double>>> args(res, triplets);
        return args;
    }

    // Compute Anchor model residual and jacobian each.
    void AnchorModel::computeEach(Eigen::VectorXd& res, std::vector<int>& I, std::vector<int>& J, std::vector<double>& V,
                                  int AnchorIdx, int ArcSegNum, std::string type, 
                                  const Eigen::Matrix3d& U, const Eigen::Vector3d& m, const double& h, int i, int row_offset)
    {
        Eigen::Vector3d e1(1,0,0);
        Eigen::Matrix3d cov;
        cov << 5*5, 0, 0,
               0, 5*5, 0,
               0, 0, 5*5;

        std::pair<int,int> rows(3*i,3*i+2);

        // Column boundaries
        std::pair<int,int> m_cols(8*ArcSegNum,8*ArcSegNum+2);
        std::pair<int,int> U_cols(8*ArcSegNum+3,8*ArcSegNum+5);
        std::pair<int,int> h_cols(8*ArcSegNum+6,8*ArcSegNum+6);

        // Add indices to I, J, V
        Eigen::Matrix3d m_jac = math_tools::invMahalanobis(Eigen::Matrix3d::Identity(),cov);
        math_tools::sparseFormat(I, J, V, rows, m_cols, m_jac, row_offset);

        Eigen::Matrix3d u_jac;
        Eigen::Vector3d h_jac;

        if (type == "front")
        {
            Eigen::Vector3d res_ = math_tools::invMahalanobis(m + h * h * U * e1 - pts[AnchorIdx], cov);
            res.segment<3>(3*i) = res_;

            u_jac = -h*h * U * math_tools::skew(e1);
            h_jac = 2*h * U * e1;
            // h_jac << 2*h, 2*h, 2*h;

        }
        else if (type == "back")
        {
            Eigen::Vector3d res_ = math_tools::invMahalanobis(m - h * h * U * e1 - pts[AnchorIdx], cov);
            res.segment<3>(3*i) = res_;

            u_jac = h*h * U * math_tools::skew(e1);
            h_jac = -2*h * U * e1;
            // h_jac << -2*h, -2*h, -2*h;
        }
        Eigen::Matrix3d norm_u_jac = math_tools::invMahalanobis(u_jac,cov);
        Eigen::Vector3d norm_h_jac = math_tools::invMahalanobis(h_jac,cov);
        math_tools::sparseFormat(I, J, V, rows, U_cols, norm_u_jac, row_offset);
        math_tools::sparseFormat(I, J, V, rows, h_cols, norm_h_jac, row_offset);
    }                                  
    
    // Compute Measurement model residual and jacobian 
    std::pair<Eigen::VectorXd, std::vector<Eigen::Triplet<double>>> MeasurementModel::compute(std::vector<std::pair<int,int>> intvs,
                                                                                              const std::vector<Eigen::Matrix3d>& Us,
                                                                                              const std::vector<Eigen::Vector3d>& ms,
                                                                                              const std::vector<double>& hs,
                                                                                              const std::vector<double>& ks,
                                                                                              int m, int row_offset)
    {
        // std::cout << "[Measurement Model]" << std::endl;
        Eigen::VectorXd res(m);
        std::vector<int> I; 
        std::vector<int> J;
        std::vector<double> V;
        for (int i=0;i<intvs.size();i++)
        {
            int lb = intvs[i].first; int ub = intvs[i].second;
            
            if (lb != 0)
                lb += 1; // Exclude repeated measurement computation at each anchors

            for (int j=lb;j<=ub;j++)
                MeasurementModel::computeEach(res, I, J, V, i, Us[i], ms[i], hs[i], ks[i], j, row_offset);
        }

        std::vector<Eigen::Triplet<double>> triplets;
        for (int i=0; i<I.size(); i++)
            triplets.push_back(Eigen::Triplet<double>(I[i], J[i], V[i]));
        
        std::pair<Eigen::VectorXd, std::vector<Eigen::Triplet<double>>> args(res, triplets);
        return args;
    }

    // Compute Measurement model residual and jacobian each
    void MeasurementModel::computeEach(Eigen::VectorXd& res, std::vector<int>& I, std::vector<int>& J, std::vector<double>& V,
                                       int ArcSegNum, const Eigen::Matrix3d& U, const Eigen::Vector3d& m, const double& h, const double& k, int j, int row_offset)
    {
        Eigen::Vector3d e2(0,1,0);
        std::pair<int,int> rows(3*j,3*j+2);

        // Column boundaries
        std::pair<int,int> m_cols(8*ArcSegNum,8*ArcSegNum+2);
        std::pair<int,int> U_cols(8*ArcSegNum+3,8*ArcSegNum+5);
        std::pair<int,int> h_cols(8*ArcSegNum+6,8*ArcSegNum+6);
        std::pair<int,int> k_cols(8*ArcSegNum+7,8*ArcSegNum+7);

        Eigen::Matrix3d m_jac;
        Eigen::Matrix3d u_jac;
        Eigen::Vector3d h_jac;
        Eigen::Vector3d k_jac;

        Eigen::VectorXd cv = covs[j];
        Eigen::Matrix3d cov;
        cov << cv[0], cv[1], cv[2],
               cv[3], cv[4], cv[5],
               cv[6], cv[7], cv[8]; // Need to check for differnent applications.

        // Compute residuals
        // Eigen::Matrix3d S;
        // S << 1, 0, 0,
        //      0, 1, 0,
        //      0, 0, 0;

        // Eigen::Vector3d Xc = m - k * U * e2;
        // Eigen::Vector3d Pj_rel = U.transpose() * (pts[j] - Xc);
        // double den = (S*Pj_rel).norm();
        // Eigen::Vector3d res_ = (sqrt(k*k + h*h*h*h)/den * U * S * U.transpose() - Eigen::Matrix3d::Identity()) * (pts[j] - Xc);
        Eigen::Vector3d res_ = MeasurementModel::getResiduals(U,m,h,k,j);
        res.segment<3>(3*j) = math_tools::invMahalanobis(res_,cov);

        // Compute jacobians
        MeasurementModel::getJacobians(m_jac,u_jac,h_jac,k_jac,U,m,h,k,j);
        math_tools::sparseFormat(I, J, V, rows, m_cols, math_tools::invMahalanobis(m_jac,cov), row_offset);
        math_tools::sparseFormat(I, J, V, rows, U_cols, math_tools::invMahalanobis(u_jac,cov), row_offset);
        math_tools::sparseFormat(I, J, V, rows, h_cols, math_tools::invMahalanobis(h_jac,cov), row_offset);
        math_tools::sparseFormat(I, J, V, rows, k_cols, math_tools::invMahalanobis(k_jac,cov), row_offset);
    }

    Eigen::Vector3d MeasurementModel::getResiduals(const Eigen::Matrix3d& U, const Eigen::Vector3d& m, const double& h, const double& k, int j)
    {
        Eigen::Vector3d e2(0,1,0);
        Eigen::Matrix3d S;
        S << 1, 0, 0,
             0, 1, 0,
             0, 0, 0;

        Eigen::Vector3d Xc = m - k * U * e2;
        Eigen::Vector3d Pj_rel = U.transpose() * (pts[j] - Xc);
        double den = (S*Pj_rel).norm();
        Eigen::Vector3d res = (sqrt(k*k + h*h*h*h)/den * U * S * U.transpose() - Eigen::Matrix3d::Identity()) * (pts[j] - Xc);
        return res;
    }

    void MeasurementModel::getJacobians(Eigen::Matrix3d& m_jac, Eigen::Matrix3d& u_jac, Eigen::Vector3d& h_jac, Eigen::Vector3d& k_jac,
                                        const Eigen::Matrix3d& U, const Eigen::Vector3d& m, const double& h, const double& k, int j)
    {
        // Used symbolic toolbox from MATLAB to compute the jacobians for complex measurement function.
        /*

        % Below is the MATLAB Code for analytic jacobian derivation
        syms h k u1 u2 u3 m1 m2 m3
        u_delta = [u1; u2; u3];
        Pj = sym("pj",[3,1]);
        m = [m1; m2; m3];


        S = [1,0,0;0,1,0;0,0,0];

        Ut_delta = eye(3) - skewSym(u_delta);
        U_delta = eye(3) + skewSym(u_delta);

        U_base = sym("U",[3,3]);
        U = U_base * U_delta;
        Ut = Ut_delta * transposeSym(U_base);

        Xc = m - k * U* [0;1;0];
        Pj_rel = Ut * (Pj - Xc);

        den = normSym(S*Pj_rel);
        res = (sqrt(k^2 + h^4) / den * U * S * Ut - eye(3)) * (Pj - Xc);

        Jm = subs(jacobian(res,m),[u1,u2,u3],[0,0,0]);
        Ju = subs(jacobian(res,u_delta),[u1,u2,u3],[0,0,0]);
        Jh = subs(jacobian(res,h),[u1,u2,u3],[0,0,0]);
        Jk = subs(jacobian(res,k),[u1,u2,u3],[0,0,0]);


        disp([Jm, Ju, Jh, Jk])



        function transSymMat = transposeSym(Mat)
            transSymMat = [Mat(1,1), Mat(2,1), Mat(3,1);
                        Mat(1,2), Mat(2,2), Mat(3,2);
                        Mat(1,3), Mat(2,3), Mat(3,3)];
        end

        function val = normSym(vec)
            val = sqrt(vec(1)^2 + vec(2)^2 + vec(3)^2);
        end

        function mat = skewSym(vec)
            mat = [0 -vec(3) vec(2);
                vec(3) 0 -vec(1);
                -vec(2) vec(1) 0];
        end
        
        */
        
        // unconstrained::getMJac(m_jac, U, m, h, k, pts[i]);
        // unconstrained::getUJac(u_jac, U, m, h, k, pts[i]);
        // unconstrained::getHJac(h_jac, U, m, h, k, pts[i]);
        // unconstrained::getKJac(k_jac, U, m, h, k, pts[i]);
        // double pj1 = pts[j].x(); double pj2 = pts[j].y(); double pj3 = pts[j].z();
        // double m1 = m.x(); double m2 = m.y(); double m3 = m.z();
        // double U11 = U(0,0); double U12 = U(0,1); double U13 = U(0,2);
        // double U21 = U(1,0); double U22 = U(1,1); double U23 = U(2,2);
        // double U31 = U(2,0); double U32 = U(2,1); double U33 = U(2,2);

        // double sig51 = pj1 - m1 + U12 * k;
        // double sig50 = pj2 - m2 + U22 * k;
        // double sig49 = pj3 - m3 + U32 * k;
        // double sig48 = U12 * sig51 + U22 * sig50 + U32 * sig49;
        // double sig47 = U11 * sig51 + U21 * sig50 + U31 * sig49;
        // double sig46 = sig47 * sig47 + sig48 * sig48;
        // double sig45 = sqrt(k*k + h*h*h*h);
        // double sig44 = 2 * pow(sig46, 1.5);
        // double sig43 = 2*sig48*(U11*sig51+U21*sig50+U31*sig49+U11*U12*k+U21*U22*k+U31*U32*k) - 2*sig47*(-k*U11*U11-k*U21*U21-k*U31*U31+U12*sig51+U22*sig50+U32*sig49);
        // double sig42 = 2*U31*sig47 + 2*U32*sig48;
        // double sig41 = 2*U21*sig47 + 2*U22*sig48;
        // double sig40 = 2*U11*sig47 + 2*U12*sig48;
        // double sig39 = U13*sig51 + U23*sig50 + U33*sig49;
        // double sig38 = 2*sig48*(U13*sig51+U23*sig50+U33*sig49+U12*U13*k+U22*U23*k+U32*U33*k) + 2*sig47*(U11*U13*k+U21*U23*k+U31*U33*k);
        // double sig37 = 2*sig47*(U11*U12+U21*U22+U31*U32) + 2*sig48*(U12*U12+U22*U22+U32*U32);
        // double sig36 = U11*U11*sig45/sqrt(sig46);
        // double sig35 = U12*U12*sig45/sqrt(sig46);
        // double sig34 = U21*U21*sig45/sqrt(sig46);
        // double sig33 = U22*U22*sig45/sqrt(sig46);
        // double sig32 = U31*U31*sig45/sqrt(sig46);
        // double sig31 = U32*U32*sig45/sqrt(sig46);
        // double sig30 = U11*U21*sig45/sqrt(sig46);
        // double sig29 = U12*U22*sig45/sqrt(sig46);
        // double sig28 = U11*U31*sig45/sqrt(sig46);
        // double sig27 = U12*U32*sig45/sqrt(sig46);
        // double sig26 = U21*U31*sig45/sqrt(sig46);
        // double sig25 = U22*U32*sig45/sqrt(sig46);
        // double sig24 = U11*U21*sig43*sig45/sig44 + U12*U22*sig43*sig45/sig44;
        // double sig23 = U11*U21*sig42*sig45/sig44 + U12*U22*sig42*sig45/sig44;
        // double sig22 = U11*U21*sig41*sig45/sig44 + U12*U22*sig41*sig45/sig44;
        // double sig21 = U11*U21*sig40*sig45/sig44 + U12*U22*sig40*sig45/sig44;
        // double sig20 = U11*U31*sig43*sig45/sig44 + U12*U32*sig43*sig45/sig44;
        // double sig19 = U11*U31*sig42*sig45/sig44 + U12*U32*sig42*sig45/sig44;
        // double sig18 = U11*U31*sig41*sig45/sig44 + U12*U32*sig41*sig45/sig44;
        // double sig17 = U11*U31*sig40*sig45/sig44 + U12*U32*sig40*sig45/sig44;
        // double sig16 = U21*U31*sig43*sig45/sig44 + U22*U32*sig43*sig45/sig44;
        // double sig15 = U21*U31*sig42*sig45/sig44 + U22*U32*sig42*sig45/sig44;
        // double sig14 = U21*U31*sig41*sig45/sig44 + U22*U32*sig41*sig45/sig44;
        // double sig13 = U21*U31*sig40*sig45/sig44 + U22*U32*sig40*sig45/sig44;
        // double sig12 = 2*U11*U21*h*h*h/(sqrt(sig46)*sig45) + 2*U12*U22*h*h*h/(sqrt(sig46)*sig45);
        // double sig11 = 2*U11*U31*h*h*h/(sqrt(sig46)*sig45) + 2*U12*U32*h*h*h/(sqrt(sig46)*sig45);
        // double sig10 = 2*U21*U31*h*h*h/(sqrt(sig46)*sig45) + 2*U22*U32*h*h*h/(sqrt(sig46)*sig45);
        // double sig9 = U11*U23*sig45/sqrt(sig46) + U13*U21*sig45/sqrt(sig46) - U11*U21*sig45*sig47*sig39/pow(sig46,1.5) - U12*U22*sig45*sig47*sig39/pow(sig46,1.5);
        // double sig8 = U11*U33*sig45/sqrt(sig46) + U13*U31*sig45/sqrt(sig46) - U11*U31*sig45*sig47*sig39/pow(sig46,1.5) - U12*U32*sig45*sig47*sig39/pow(sig46,1.5);
        // double sig7 = U21*U33*sig45/sqrt(sig46) + U23*U31*sig45/sqrt(sig46) - U21*U31*sig45*sig47*sig39/pow(sig46,1.5) - U22*U32*sig45*sig47*sig39/pow(sig46,1.5);
        // double sig6 = U12*U23*sig45/sqrt(sig46) + U13*U22*sig45/sqrt(sig46) - U11*U21*sig38*sig45/sig44 - U12*U22*sig38*sig45/sig44;
        // double sig5 = U12*U33*sig45/sqrt(sig46) + U13*U32*sig45/sqrt(sig46) - U11*U31*sig38*sig45/sig44 - U12*U32*sig38*sig45/sig44;
        // double sig4 = U22*U33*sig45/sqrt(sig46) + U23*U32*sig45/sqrt(sig46) - U21*U31*sig38*sig45/sig44 - U22*U32*sig38*sig45/sig44;
        // double sig3 = U11*U21*k/(sqrt(sig46)*sig45) + U12*U22*k/(sqrt(sig46)*sig45) - U11*U21*sig37*sig45/sig44 - U12*U22*sig37*sig45/sig44;
        // double sig2 = U11*U31*k/(sqrt(sig46)*sig45) + U12*U32*k/(sqrt(sig46)*sig45) - U11*U31*sig37*sig45/sig44 - U12*U32*sig36*sig45/sig44;
        // double sig1 = U21*U31*k/(sqrt(sig46)*sig45) + U22*U32*k/(sqrt(sig46)*sig45) - U21*U31*sig37*sig45/sig44 - U22*U32*sig37*sig45/sig44;

        // m_jac(0,0) = sig21*sig50 + sig17*sig49 + (U11*U11*sig40*sig45 + U12*U12*sig40*sig45)*sig51/sig44 - sig36 - sig35 + 1;
        // m_jac(0,1) = sig22*sig50 + sig18*sig49 + (U11*U11*sig41*sig45 + U12*U12*sig41*sig45)*sig51/sig44 - sig30 - sig29;
        // m_jac(0,2) = sig23*sig50 + sig19*sig49 + (U11*U11*sig42*sig45 + U12*U12*sig42*sig45)*sig51/sig44 - sig28 - sig27;
        // m_jac(1,0) = sig21*sig51 + sig13*sig49 + (U21*U21*sig40*sig45 + U22*U22*sig40*sig45)*sig50/sig44 - sig30 - sig29;
        // m_jac(1,1) = sig22*sig51 + sig14*sig49 + (U21*U21*sig41*sig45 + U22*U22*sig40*sig45)*sig50/sig44 - sig34 - sig33 + 1;
        // m_jac(1,2) = sig23*sig51 + sig15*sig49 + (U21*U21*sig42*sig45 + U22*U22*sig42*sig45)*sig50/sig44 - sig26 - sig25;
        // m_jac(2,0) = sig17*sig51 + sig13*sig50 + (U31*U31*sig40*sig45 + U32*U32*sig40*sig45)*sig49/sig44 - sig28 - sig27;
        // m_jac(2,1) = sig18*sig51 + sig14*sig50 + (U31*U31*sig41*sig45 + U32*U32*sig41*sig45)*sig49/sig44 - sig26 - sig25;
        // m_jac(2,2) = sig19*sig51 + sig15*sig50 + (U31*U31*sig42*sig45 + U32*U32*sig42*sig45)*sig49/sig44 - sig32 - sig31 + 1;

        // u_jac(0,0) = sig50*sig6 - (U11*U11*sig38*sig45/sig44 - 2*U12*U13*sig45/sqrt(sig46) + U12*U12*sig38*sig45/sig44)*sig51 + sig49*sig5 + U13*k*(sig36+sig35-1) + U23*k*(sig30+sig29) + U33*k*(sig28+sig27);
        // u_jac(0,1) = sig51*(U11*U11*sig45*sig47*sig39/pow(sig46,1.5) - 2*U11*U13*sig45/sqrt(sig46) + U12*U12*sig45*sig47*sig39/pow(sig46,1.5)) - sig49*sig8 - sig50*sig9;
        // u_jac(0,2) = sig24*sig50 + sig20*sig49 + (U11*U11*sig43*sig45 + U12*U12*sig43*sig45)*sig51/sig44 - U11*k*(sig36+sig35-1) - U21*k*(sig30+sig29) - U31*k*(sig28*sig27);
        // u_jac(1,0) = sig51*sig6 - (U21*U21*sig38*sig45/sig44 - 2*U22*U23*sig45/sqrt(sig46) + U22*U22*sig38*sig45/sig44)*sig50 + sig49*sig4 + U23*k*(sig34+sig33-1) + U13*k*(sig30+sig29) + U33*k*(sig26+sig25);
        // u_jac(1,1) = sig50*(U21*U21*sig45*sig47*sig39/pow(sig46,1.5) - 2*U21*U23*sig45/sqrt(sig46) + U22*U22*sig45*sig47*sig39/pow(sig46,1.5)) - sig49*sig7 - sig51*sig9;
        // u_jac(1,2) = sig24*sig51 + sig16*sig49 + (U21*U21*sig43*sig45 + U22*U22*sig43*sig45)*sig50/sig44 - U21*k*(sig34+sig33-1) - U11*k*(sig30+sig29) - U31*k*(sig26*sig25);
        // u_jac(2,0) = sig51*sig5 - (U31*U31*sig38*sig45/sig44 - 2*U32*U33*sig45/sqrt(sig46) + U32*U32*sig38*sig45/sig44)*sig49 + sig50*sig4 + U33*k*(sig32+sig31-1) + U13*k*(sig28+sig27) + U23*k*(sig26+sig25);
        // u_jac(2,1) = sig49*(U31*U31*sig45*sig47*sig39/pow(sig46,1.5) - 2*U31*U33*sig45/sqrt(sig46) + U32*U32*sig45*sig47*sig39/pow(sig46,1.5)) - sig50*sig7 - sig51*sig8;
        // u_jac(2,2) = sig20*sig51 + sig16*sig50 + (U31*U31*sig43*sig45 + U32*U32*sig43*sig45)*sig49/sig44 - U31*k*(sig32+sig31-1) - U11*k*(sig28+sig27) - U21*k*(sig26*sig25);

        // h_jac[0] = sig12*sig50 + sig11*sig49 + (2*U11*U11 + 2*U12*U12)*sig51*h*h*h/(sqrt(sig46)*sig45);
        // h_jac[1] = sig12*sig51 + sig10*sig49 + (2*U21*U21 + 2*U22*U22)*sig50*h*h*h/(sqrt(sig46)*sig45);
        // h_jac[2] = sig11*sig51 + sig10*sig50 + (2*U31*U31 + 2*U32*U32)*sig49*h*h*h/(sqrt(sig46)*sig45);

        // k_jac[0] = sig50*sig3 + sig49*sig2 + sig51*(U11*U11*k/(sqrt(sig46)*sig45) + U12*U12*k/(sqrt(sig46)*sig45) - U11*U11*sig37*sig45/sig44 - U12*U12*sig37*sig45/sig44) + U22*(sig30+sig29) + U32*(sig28+sig27) + U12*(sig36+sig35-1);
        // k_jac[1] = sig51*sig3 + sig49*sig1 + sig50*(U21*U21*k/(sqrt(sig46)*sig45) + U22*U22*k/(sqrt(sig46)*sig45) - U21*U21*sig37*sig45/sig44 - U22*U22*sig37*sig45/sig44) + U12*(sig30+sig29) + U32*(sig26+sig25) + U22*(sig34+sig33-1);
        // k_jac[2] = sig51*sig2 + sig50*sig1 + sig49*(U31*U31*k/(sqrt(sig46)*sig45) + U32*U32*k/(sqrt(sig46)*sig45) - U31*U31*sig37*sig45/sig44 - U32*U32*sig37*sig45/sig44) + U12*(sig28+sig27) + U22*(sig26+sig25) + U32*(sig32+sig31-1);

        double epsilon = 1e-6;
        std::vector<Eigen::Vector3d> eps;
        Eigen::Vector3d eps1(1e-6,0,0);
        Eigen::Vector3d eps2(0,1e-6,0);
        Eigen::Vector3d eps3(0,0,1e-6);
        eps.push_back(eps1); eps.push_back(eps2); eps.push_back(eps3);

        Eigen::Vector3d jac;

        //m
        for (int i=0;i<3;i++)
        {
            Eigen::Vector3d m_minus = m - eps[i];
            Eigen::Vector3d m_plus = m + eps[i];
            Eigen::Vector3d res_minus = MeasurementModel::getResiduals(U,m_minus,h,k,j);
            Eigen::Vector3d res_plus = MeasurementModel::getResiduals(U,m_plus,h,k,j);
            jac = (res_plus - res_minus)/(2*epsilon);
            m_jac.col(i) << jac(0), jac(1) , jac(2);
        }

        //U
        for (int i=0;i<3;i++)
        {
            Eigen::Matrix3d U_minus = U * math_tools::Exp_map(-eps[i]);
            Eigen::Matrix3d U_plus = U * math_tools::Exp_map(eps[i]);
            jac = (MeasurementModel::getResiduals(U_plus,m,h,k,j) - MeasurementModel::getResiduals(U_minus,m,h,k,j))/(2*epsilon);
            u_jac.col(i) << jac(0), jac(1) , jac(2);
        }

        // h
        
        double h_minus = h - epsilon;
        double h_plus = h + epsilon;
        jac = (MeasurementModel::getResiduals(U,m,h_plus,k,j) - MeasurementModel::getResiduals(U,m,h_minus,k,j))/(2*epsilon);
        h_jac = jac;
        

        //k
        double k_minus = k - epsilon;
        double k_plus = k + epsilon;
        jac = (MeasurementModel::getResiduals(U,m,h,k_plus,j) - MeasurementModel::getResiduals(U,m,h,k_minus,j))/(2*epsilon);
        k_jac = jac;
    }

}

namespace constrained
{
    // Compute G0 Continuity model residual and jacobian 
    std::pair<Eigen::VectorXd, std::vector<Eigen::Triplet<double>>> G0ContinuityModel::compute(const std::vector<Eigen::Matrix3d>& Us,
                                                                                               const std::vector<Eigen::Vector3d>& ms,
                                                                                               const std::vector<double>& hs,
                                                                                               int m, int row_offset, const double& mu)
    {
        // std::cout << "[G0 Continuity Model]" << std::endl;
        int n = static_cast<int>(ms.size());
        Eigen::VectorXd res(m);
        std::vector<int> I; 
        std::vector<int> J;
        std::vector<double> V;

        for (int i=0;i<n-1;i++)
            G0ContinuityModel::computeEach(res, I, J, V, i, i+1, Us[i], Us[i+1], ms[i], ms[i+1], hs[i], hs[i+1], row_offset, mu);
        
        // std::cout << res << std::endl << std::endl;

        std::vector<Eigen::Triplet<double>> triplets;
        for (int i=0;i<I.size();i++)
            triplets.push_back(Eigen::Triplet<double>(I[i], J[i], V[i]));

        std::pair<Eigen::VectorXd, std::vector<Eigen::Triplet<double>>> args(res, triplets);
        return args;
    }

    // Compute G0 Continuity model residual and jacobian each
    void G0ContinuityModel::computeEach(Eigen::VectorXd& res, std::vector<int>& I, std::vector<int>& J, std::vector<double>& V, 
                                        int frontArcSegNum, int backArcSegNum, 
                                        const Eigen::Matrix3d& U1, const Eigen::Matrix3d& U2, const Eigen::Vector3d& m1, const Eigen::Vector3d& m2,
                                        const double& h1, const double& h2, int row_offset, const double& mu)
    {
        Eigen::Vector3d e1(1,0,0);
        std::pair<int,int> rows(3*frontArcSegNum,3*frontArcSegNum+2);

        // Column Boundaries
        std::pair<int,int> m1_cols(8*frontArcSegNum,8*frontArcSegNum+2);
        std::pair<int,int> U1_cols(8*frontArcSegNum+3,8*frontArcSegNum+5);
        std::pair<int,int> h1_cols(8*frontArcSegNum+6,8*frontArcSegNum+6);

        std::pair<int,int> m2_cols(8*backArcSegNum,8*backArcSegNum+2);
        std::pair<int,int> U2_cols(8*backArcSegNum+3,8*backArcSegNum+5);
        std::pair<int,int> h2_cols(8*backArcSegNum+6,8*backArcSegNum+6);

        Eigen::Vector3d res_ = G0ContinuityModel::getResiduals(U1,U2,m1,m2,h1,h2);
        res.segment<3>(3*frontArcSegNum) = sqrt(mu) * res_;

        Eigen::Matrix3d m1_jac; Eigen::Matrix3d m2_jac;
        Eigen::Matrix3d u1_jac; Eigen::Matrix3d u2_jac;
        Eigen::Vector3d h1_jac; Eigen::Vector3d h2_jac;
        // G0ContinuityModel::getJacobians(m1_jac,m2_jac,u1_jac,u2_jac,h1_jac,h2_jac,U1,U2,m1,m2,h1,h2);
        h1_jac = -2*h1*U1*e1;
        h2_jac = -2*h2*U2*e1;

        math_tools::sparseFormat(I, J, V, rows, m1_cols, sqrt(mu)*Eigen::Matrix3d::Identity(), row_offset);
        math_tools::sparseFormat(I, J, V, rows, U1_cols, sqrt(mu)*h1*h1*U1*math_tools::skew(e1), row_offset);
        math_tools::sparseFormat(I, J, V, rows, h1_cols, sqrt(mu)*h1_jac, row_offset);
        math_tools::sparseFormat(I, J, V, rows, m2_cols, -sqrt(mu)*Eigen::Matrix3d::Identity(), row_offset);
        math_tools::sparseFormat(I, J, V, rows, U2_cols, sqrt(mu)*h2*h2*U2*math_tools::skew(e1), row_offset);
        math_tools::sparseFormat(I, J, V, rows, h2_cols, sqrt(mu)*h2_jac, row_offset);
        // math_tools::sparseFormat(I, J, V, rows, m1_cols, sqrt(mu)*m1_jac, row_offset);
        // math_tools::sparseFormat(I, J, V, rows, U1_cols, sqrt(mu)*u1_jac, row_offset);
        // math_tools::sparseFormat(I, J, V, rows, h1_cols, sqrt(mu)*h1_jac, row_offset);
        // math_tools::sparseFormat(I, J, V, rows, m2_cols, sqrt(mu)*m2_jac, row_offset);
        // math_tools::sparseFormat(I, J, V, rows, U2_cols, sqrt(mu)*u2_jac, row_offset);
        // math_tools::sparseFormat(I, J, V, rows, h2_cols, sqrt(mu)*h2_jac, row_offset);
    }                                                                      

    Eigen::Vector3d G0ContinuityModel::getResiduals(const Eigen::Matrix3d& U1, const Eigen::Matrix3d& U2,
                                                    const Eigen::Vector3d& m1, const Eigen::Vector3d& m2, const double& h1, const double& h2)
    {
        Eigen::Vector3d e1(1,0,0);
        Eigen::Vector3d res;
        res = m1 - h1 * h1 * U1 * e1 - m2 - h2 * h2 * U2 * e1;
        return res;
    }

    void G0ContinuityModel::getJacobians(Eigen::Matrix3d& m1_jac, Eigen::Matrix3d& m2_jac, Eigen::Matrix3d& u1_jac, Eigen::Matrix3d& u2_jac,
                                         Eigen::Vector3d& h1_jac, Eigen::Vector3d& h2_jac, const Eigen::Matrix3d& U1, const Eigen::Matrix3d& U2,
                                         const Eigen::Vector3d& m1, const Eigen::Vector3d& m2, const double& h1, const double& h2)
    {
        double epsilon = 1e-6;
        std::vector<Eigen::Vector3d> eps;
        Eigen::Vector3d eps1(1e-6,0,0);
        Eigen::Vector3d eps2(0,1e-6,0);
        Eigen::Vector3d eps3(0,0,1e-6);
        eps.push_back(eps1); eps.push_back(eps2); eps.push_back(eps3);

        Eigen::Vector3d jac;

        //m1
        for (int i=0;i<3;i++)
        {
            Eigen::Vector3d m1_minus = m1 - eps[i];
            Eigen::Vector3d m1_plus = m1 + eps[i];
            Eigen::Vector3d res_minus = G0ContinuityModel::getResiduals(U1,U2,m1_minus,m2,h1,h2);
            Eigen::Vector3d res_plus = G0ContinuityModel::getResiduals(U1,U2,m1_plus,m2,h1,h2);
            jac = (res_plus - res_minus)/(2*epsilon);
            m1_jac.col(i) << jac(0), jac(1) , jac(2);
        }

        //m2
        for (int i=0;i<3;i++)
        {
            Eigen::Vector3d m2_minus = m2 - eps[i];
            Eigen::Vector3d m2_plus = m2 + eps[i];
            Eigen::Vector3d res_minus = G0ContinuityModel::getResiduals(U1,U2,m1,m2_minus,h1,h2);
            Eigen::Vector3d res_plus = G0ContinuityModel::getResiduals(U1,U2,m1,m2_plus,h1,h2);
            jac = (res_plus - res_minus)/(2*epsilon);
            m2_jac.col(i) << jac(0), jac(1) , jac(2);
        }

        //U1
        for (int i=0;i<3;i++)
        {
            Eigen::Matrix3d U1_minus = U1 * math_tools::Exp_map(-eps[i]);
            Eigen::Matrix3d U1_plus = U1 * math_tools::Exp_map(eps[i]);
            Eigen::Vector3d res_minus = G0ContinuityModel::getResiduals(U1_minus,U2,m1,m2,h1,h2);
            Eigen::Vector3d res_plus = G0ContinuityModel::getResiduals(U1_plus,U2,m1,m2,h1,h2);

            jac = (res_plus - res_minus)/(2*epsilon);
            u1_jac.col(i) << jac(0), jac(1) , jac(2);
        }

        //U2
        for (int i=0;i<3;i++)
        {
            Eigen::Matrix3d U2_minus = U2 * math_tools::Exp_map(-eps[i]);
            Eigen::Matrix3d U2_plus = U2 * math_tools::Exp_map(eps[i]);
            Eigen::Vector3d res_minus = G0ContinuityModel::getResiduals(U1,U2_minus,m1,m2,h1,h2);
            Eigen::Vector3d res_plus = G0ContinuityModel::getResiduals(U1,U2_plus,m1,m2,h1,h2);

            jac = (res_plus - res_minus)/(2*epsilon);
            u2_jac.col(i) << jac(0), jac(1) , jac(2);
        }

        // h1
        double h1_minus = h1 - epsilon;
        double h1_plus = h1 + epsilon;
        Eigen::Vector3d res_minus = G0ContinuityModel::getResiduals(U1,U2,m1,m2,h1_minus,h2);
        Eigen::Vector3d res_plus = G0ContinuityModel::getResiduals(U1,U2,m1,m2,h1_plus,h2);
        jac = (res_plus - res_minus)/(2*epsilon);
        h1_jac = jac;

        // h2
        double h2_minus = h2 - epsilon;
        double h2_plus = h2 + epsilon;
        res_minus = G0ContinuityModel::getResiduals(U1,U2,m1,m2,h1,h2_minus);
        res_plus = G0ContinuityModel::getResiduals(U1,U2,m1,m2,h1,h2_plus);
        jac = (res_plus - res_minus)/(2*epsilon);
        h2_jac = jac;
    }


    // Compute G1 Continuity model residual and jacobian 
    std::pair<Eigen::VectorXd, std::vector<Eigen::Triplet<double>>> G1ContinuityModel::compute(const std::vector<Eigen::Matrix3d>& Us,
                                                                                               const std::vector<Eigen::Vector3d>& ms,
                                                                                               const std::vector<double>& hs,
                                                                                               const std::vector<double>& ks,                                                                       
                                                                                               int m, int row_offset, const double& mu)
    {
        // std::cout << "[G1 Continuity Model]" << std::endl;
        int n = static_cast<int>(ms.size());
        Eigen::VectorXd res(m);
        std::vector<int> I; 
        std::vector<int> J;
        std::vector<double> V;

        for (int i=0;i<n-1;i++)
            G1ContinuityModel::computeEach(res, I, J, V, i, i+1, Us[i], Us[i+1], ms[i], ms[i+1], hs[i], hs[i+1], ks[i], ks[i+1], row_offset, mu);

        // std::cout << res << std::endl << std::endl;

        std::vector<Eigen::Triplet<double>> triplets;
        for (int i=0;i<I.size();i++)
            triplets.push_back(Eigen::Triplet<double>(I[i], J[i], V[i]));

        std::pair<Eigen::VectorXd, std::vector<Eigen::Triplet<double>>> args(res, triplets);
        return args;
    }

    // Compute G1 Continuity model residual and jacobian each
    void G1ContinuityModel::computeEach(Eigen::VectorXd& res, std::vector<int>& I, std::vector<int>& J, std::vector<double>& V, 
                                        int frontArcSegNum, int backArcSegNum, 
                                        const Eigen::Matrix3d& U1, const Eigen::Matrix3d& U2, const Eigen::Vector3d& m1, const Eigen::Vector3d& m2,
                                        const double& h1, const double& h2, const double& k1, const double& k2, int row_offset, const double& mu)
    {
        Eigen::Vector3d e1(1,0,0);
        Eigen::Vector3d e2(0,1,0);
        std::pair<int,int> rows(3*frontArcSegNum,3*frontArcSegNum+2);

        // Column Boundaries
        std::pair<int,int> m1_cols(8*frontArcSegNum,8*frontArcSegNum+2);
        std::pair<int,int> U1_cols(8*frontArcSegNum+3,8*frontArcSegNum+5);
        std::pair<int,int> h1_cols(8*frontArcSegNum+6,8*frontArcSegNum+6);
        std::pair<int,int> k1_cols(8*frontArcSegNum+7,8*frontArcSegNum+7);

        std::pair<int,int> m2_cols(8*backArcSegNum,8*backArcSegNum+2);
        std::pair<int,int> U2_cols(8*backArcSegNum+3,8*backArcSegNum+5);
        std::pair<int,int> h2_cols(8*backArcSegNum+6,8*backArcSegNum+6);
        std::pair<int,int> k2_cols(8*backArcSegNum+7,8*backArcSegNum+7);

        // Eigen::Matrix<double, 6, 3> m1_jac; Eigen::Matrix<double, 6, 3> m2_jac;
        // Eigen::Matrix<double, 6, 3> U1_jac; Eigen::Matrix<double, 6, 3> U2_jac;
        // Eigen::Matrix<double, 6, 1> h1_jac; Eigen::Matrix<double, 6, 1> h2_jac;
        // Eigen::Matrix<double, 6, 1> k1_jac; Eigen::Matrix<double, 6, 1> k2_jac;
        Eigen::Matrix3d m1_jac; Eigen::Matrix3d m2_jac;
        Eigen::Matrix3d U1_jac; Eigen::Matrix3d U2_jac;
        Eigen::Vector3d h1_jac; Eigen::Vector3d h2_jac;
        Eigen::Vector3d k1_jac; Eigen::Vector3d k2_jac;
        
        // G1ContinuityModel::getJacobians2(m1_jac,m2_jac,U1_jac,U2_jac,h1_jac,h2_jac,k1_jac,k2_jac,
        //                                  U1,U2,m1,m2,h1,h2,k1,k2);
        double dfdh1; double dfdh2;
        double dfdk1; double dfdk2;
        double dg1dh1; double dg1dh2;
        double dg1dk1; double dg1dk2;
        double dg2dh1; double dg2dh2;
        double dg2dk1; double dg2dk2;

        G1ContinuityModel::getJacobians(h1,h2,k1,k2,dfdh1,dfdh2,dfdk1,dfdk2,
                                        dg1dh1,dg1dh2,dg1dk1,dg1dk2,
                                        dg2dh1,dg2dh2,dg2dk1,dg2dk2);

        double f; double g1; double g2; 
        // Eigen::Matrix<double, 6, 1> res_ = G1ContinuityModel::getResiduals2(U1,U2,m1,m2,h1,h2,k1,k2);
        // res.segment<6>(6*frontArcSegNum) = sqrt(mu) * res_;

        Eigen::Vector3d res_;
        G1ContinuityModel::getResiduals(res_,f,g1,g2,U1,U2,m1,m2,h1,h2,k1,k2);
        res.segment<3>(3*frontArcSegNum) = sqrt(mu) * res_;
        // Eigen::Matrix3d m1_jac; Eigen::Matrix3d m2_jac;
        // Eigen::Matrix3d U1_jac; Eigen::Matrix3d U2_jac;
        // Eigen::Vector3d h1_jac; Eigen::Vector3d h2_jac;
        // Eigen::Vector3d k1_jac; Eigen::Vector3d k2_jac;

        m1_jac = (f - 0.5) * Eigen::Matrix3d::Identity();
        m2_jac = (0.5 - f) * Eigen::Matrix3d::Identity();
        U1_jac = -g1*U1*math_tools::skew(e2) - 0.5*h1*h1*U1*math_tools::skew(e1);
        U2_jac = -g2*U2*math_tools::skew(e2) + 0.5*h2*h2*U2*math_tools::skew(e1);
        h1_jac = (m1 - m2)*dfdh1 + h1*U1*e1 + (dg1dh1*U1 + dg2dh1*U2)*e2; 
        h2_jac = (m1 - m2)*dfdh2 - h2*U2*e1 + (dg1dh2*U1 + dg2dh2*U2)*e2;
        k1_jac = (m1 - m2)*dfdk1 + (dg1dk1*U1 + dg2dk1*U2)*e2;
        k2_jac = (m1 - m2)*dfdk2 + (dg1dk2*U1 + dg2dk2*U2)*e2; 
        // std::cout << "G1 Continuity" << std::endl;
        math_tools::sparseFormat(I, J, V, rows, m1_cols, sqrt(mu)*m1_jac, row_offset);
        math_tools::sparseFormat(I, J, V, rows, U1_cols, sqrt(mu)*U1_jac, row_offset);
        math_tools::sparseFormat(I, J, V, rows, h1_cols, sqrt(mu)*h1_jac, row_offset);
        math_tools::sparseFormat(I, J, V, rows, k1_cols, sqrt(mu)*k1_jac, row_offset);
        math_tools::sparseFormat(I, J, V, rows, m2_cols, sqrt(mu)*m2_jac, row_offset);
        math_tools::sparseFormat(I, J, V, rows, U2_cols, sqrt(mu)*U2_jac, row_offset);
        math_tools::sparseFormat(I, J, V, rows, h2_cols, sqrt(mu)*h2_jac, row_offset);
        math_tools::sparseFormat(I, J, V, rows, k2_cols, sqrt(mu)*k2_jac, row_offset);
    }

    void G1ContinuityModel::getResiduals(Eigen::Vector3d& res, double& f, double& g1, double& g2,
                                         const Eigen::Matrix3d& U1, const Eigen::Matrix3d& U2, const Eigen::Vector3d& m1, const Eigen::Vector3d& m2,
                                         const double& h1, const double& h2, const double& k1, const double& k2)
    {
        double num1 = pow(h1,2) * sqrt(pow(h1,4)/pow(k1,2) + 1);
        double num2 = pow(h2,2) * sqrt(pow(h2,4)/pow(k2,2) + 1);

        f = num2/(num1+num2);
        g1 = pow(h1,4)/k1 * f;
        g2 = pow(h2,4)/k2 * (1.0 - f);

        Eigen::Vector3d e1(1,0,0);
        Eigen::Vector3d e2(0,1,0);

        res = (f - 0.5)*m1 + g1*U1*e2 + 0.5*h1*h1*U1*e1 + (0.5 - f)*m2 + g2*U2*e2 - 0.5*h2*h2*U2*e1;
        // Eigen::Vector3d res1 = (f - 1)*m1 + g1*U1*e2 + pow(h1,2)*U1*e1 + (1 - f)*m2 + g2*U2*e2;
        // Eigen::Vector3d res2 = f*m1 + g1*U2*e2 - f*m2 + g2*U2*e2 - pow(h2,2)*U2*e1;
    }

    Eigen::Matrix<double, 6, 1> G1ContinuityModel::getResiduals2(const Eigen::Matrix3d& U1, const Eigen::Matrix3d& U2, const Eigen::Vector3d& m1, const Eigen::Vector3d& m2,
                                                                 const double& h1, const double& h2, const double& k1, const double& k2)
    {
        double num1 = pow(h1,2) * sqrt(pow(h1,4)/pow(k1,2) + 1);
        double num2 = pow(h2,2) * sqrt(pow(h2,4)/pow(k2,2) + 1);

        double f = num2/(num1+num2);
        double g1 = pow(h1,4)/k1 * f;
        double g2 = pow(h2,4)/k2 * (1.0 - f);

        Eigen::Vector3d e1(1,0,0);
        Eigen::Vector3d e2(0,1,0);

        Eigen::Vector3d res1 = (f - 1)*m1 + g1*U1*e2 + pow(h1,2)*U1*e1 + (1 - f)*m2 + g2*U2*e2;
        Eigen::Vector3d res2 = f*m1 + g1*U2*e2 - f*m2 + g2*U2*e2 - pow(h2,2)*U2*e1;

        Eigen::Matrix<double, 6, 1> res;
        res << res1, res2;

        // res = (f - 0.5)*m1 + g1*U1*e2 + 0.5*h1*h1*U1*e1 + (0.5 - f)*m2 + g2*U2*e2 - 0.5*h2*h2*U2*e1;
        return res;
    }

    void G1ContinuityModel::getJacobians(const double& h1, const double& h2, const double& k1, const double& k2, 
                                         double& dfdh1, double& dfdh2, double& dfdk1, double& dfdk2,
                                         double& dg1dh1, double& dg1dh2, double& dg1dk1, double& dg1dk2,
                                         double& dg2dh1, double& dg2dh2, double& dg2dk1, double& dg2dk2)
    {
        // Used symbolic toolbox from MATLAB to compute the jacobians for complex constraint function.
        /*
        syms h1 h2 k1 k2

        num1 = h1^2 * sqrt(h1^4/k1^2 + 1);
        num2 = h2^2 * sqrt(h2^4/k2^2 + 1);

        f = num2/(num1+num2);

        dfdh1 = jacobian(f,h1);
        dfdh2 = jacobian(f,h2);
        dfdk1 = jacobian(f,k1);
        dfdk2 = jacobian(f,k2);


        g1 = h1^4/k1 * f;
        g2 = h2^4/k2 * (1-f);

        dg1dh1 = jacobian(g1,h1);
        dg1dh2 = jacobian(g1,h2);
        dg1dk1 = jacobian(g1,k1);
        dg1dk2 = jacobian(g1,k2);

        dg2dh1 = jacobian(g2,h1);
        dg2dh2 = jacobian(g2,h2);
        dg2dk1 = jacobian(g2,k1);
        dg2dk2 = jacobian(g2,k2);

        d = [dfdh1; dfdh2; dfdk1; dfdk2; dg1dh1; dg1dh2; dg1dk1; dg1dk2; dg2dh1; dg2dh2; dg2dk1; dg2dk2];
        disp(d)
        */
        double sig8 = sqrt(pow(h1,4)/pow(k1,2) + 1);
        double sig7 = sqrt(pow(h2,4)/pow(k2,2) + 1);
        double sig6 = pow(h1,2) * sig8 + pow(h2,2) * sig7;
        double sig5 = 2*h2*sig7 + 2*pow(h2,5)/(pow(k2,2)*sig7);
        double sig4 = pow(h2,2) * sig7 / sig6 - 1;
        double sig3 = pow(h2,8)/(pow(k2,3)*pow(sig6,2)) - pow(h2,6)/(pow(k2,3)*sig6*sig7);
        double sig2 = 2*h1*sig8 + 2*pow(h1,5)/(pow(k1,2)*sig8);
        double sig1 = 2*h2*sig7/sig6 + 2*pow(h2,5)/(pow(k2,2)*sig6*sig7) - pow(h2,2)*sig7*sig5/pow(sig6,2);

        dfdh1 = -pow(h2,2)*sig7*sig2/pow(sig6,2);
        dfdh2 = sig1;
        dfdk1 = pow(h1,6)*pow(h2,2)*sig7/(pow(k1,3)*pow(sig6,2)*sig8);
        dfdk2 = sig3;
        dg1dh1 = 4*pow(h1,3)*pow(h2,2)*sig7/(k1*sig6) - pow(h1,4)*pow(h2,2)*sig7*sig2/(k1*pow(sig6,2));
        dg1dh2 = 2*pow(h1,4)*h2*sig7/(k1*sig6) - pow(h1,4)*pow(h2,2)*sig7*sig5/(k1*pow(sig6,2)) + 2*pow(h1,4)*pow(h2,5)/(k1*pow(k2,2)*sig6*sig7);
        dg1dk1 = pow(h1,10)*pow(h2,2)*sig7/(pow(k1,4)*pow(sig6,2)*sig8) - pow(h1,4)*pow(h2,2)*sig7/(pow(k1,2)*sig6);
        dg1dk2 = pow(h1,4)*pow(h2,8)/(k1*pow(k2,3)*pow(sig6,2)) - pow(h1,4)*pow(h2,6)/(k1*pow(k2,3)*sig6*sig7);
        dg2dh1 = pow(h2,6)*sig7*sig2/(k2*pow(sig6,2));
        dg2dh2 = -pow(h2,4)*sig1/k2 - 4*pow(h2,3)*sig4/k2;
        dg2dk1 = -pow(h1,6)*pow(h2,6)*sig7/(pow(k1,3)*k2*pow(sig6,2)*sig8);
        dg2dk2 = pow(h2,4)*sig4/pow(k2,2) - pow(h2,4)*sig3/k2;
    }

    void G1ContinuityModel::getJacobians2(Eigen::Matrix<double, 6, 3>& m1_jac, Eigen::Matrix<double, 6, 3>& m2_jac, Eigen::Matrix<double, 6, 3>& u1_jac, Eigen::Matrix<double, 6, 3>& u2_jac,
                                          Eigen::Matrix<double, 6, 1>& h1_jac, Eigen::Matrix<double, 6, 1>& h2_jac, Eigen::Matrix<double, 6, 1>& k1_jac, Eigen::Matrix<double, 6, 1>& k2_jac, 
                                          const Eigen::Matrix3d& U1, const Eigen::Matrix3d& U2, const Eigen::Vector3d& m1, const Eigen::Vector3d& m2, 
                                          const double& h1, const double& h2, const double& k1, const double& k2)
    {
        double epsilon = 1e-6;
        std::vector<Eigen::Vector3d> eps;
        Eigen::Vector3d eps1(1e-6,0,0);
        Eigen::Vector3d eps2(0,1e-6,0);
        Eigen::Vector3d eps3(0,0,1e-6);
        eps.push_back(eps1); eps.push_back(eps2); eps.push_back(eps3);

        Eigen::Matrix<double, 6, 1> jac;

        //m1
        for (int i=0;i<3;i++)
        {
            Eigen::Vector3d m1_minus = m1 - eps[i];
            Eigen::Vector3d m1_plus = m1 + eps[i];
            Eigen::Matrix<double, 6, 1> res_minus = G1ContinuityModel::getResiduals2(U1,U2,m1_minus,m2,h1,h2,k1,k2);
            Eigen::Matrix<double, 6, 1> res_plus = G1ContinuityModel::getResiduals2(U1,U2,m1_plus,m2,h1,h2,k1,k2);
            jac = (res_plus - res_minus)/(2*epsilon);
            m1_jac.col(i) << jac;
        }

        //m2
        for (int i=0;i<3;i++)
        {
            Eigen::Vector3d m2_minus = m2 - eps[i];
            Eigen::Vector3d m2_plus = m2 + eps[i];
            Eigen::Matrix<double, 6, 1> res_minus = G1ContinuityModel::getResiduals2(U1,U2,m1,m2_minus,h1,h2,k1,k2);
            Eigen::Matrix<double, 6, 1> res_plus = G1ContinuityModel::getResiduals2(U1,U2,m1,m2_plus,h1,h2,k1,k2);
            jac = (res_plus - res_minus)/(2*epsilon);
            m2_jac.col(i) << jac;
        }

        //U1
        for (int i=0;i<3;i++)
        {
            Eigen::Matrix3d U1_minus = U1 * math_tools::Exp_map(-eps[i]);
            Eigen::Matrix3d U1_plus = U1 * math_tools::Exp_map(eps[i]);
            Eigen::Matrix<double, 6, 1> res_minus = G1ContinuityModel::getResiduals2(U1_minus,U2,m1,m2,h1,h2,k1,k2);
            Eigen::Matrix<double, 6, 1> res_plus = G1ContinuityModel::getResiduals2(U1_plus,U2,m1,m2,h1,h2,k1,k2);

            jac = (res_plus - res_minus)/(2*epsilon);
            u1_jac.col(i) << jac;
        }

        //U2
        for (int i=0;i<3;i++)
        {
            Eigen::Matrix3d U2_minus = U2 * math_tools::Exp_map(-eps[i]);
            Eigen::Matrix3d U2_plus = U2 * math_tools::Exp_map(eps[i]);
            Eigen::Matrix<double, 6, 1> res_minus = G1ContinuityModel::getResiduals2(U1,U2_minus,m1,m2,h1,h2,k1,k2);
            Eigen::Matrix<double, 6, 1> res_plus = G1ContinuityModel::getResiduals2(U1,U2_plus,m1,m2,h1,h2,k1,k2);

            jac = (res_plus - res_minus)/(2*epsilon);
            u2_jac.col(i) << jac;
        }

        // h1
        double h1_minus = h1 - epsilon;
        double h1_plus = h1 + epsilon;
        Eigen::Matrix<double, 6, 1> res_minus = G1ContinuityModel::getResiduals2(U1,U2,m1,m2,h1_minus,h2,k1,k2);
        Eigen::Matrix<double, 6, 1> res_plus = G1ContinuityModel::getResiduals2(U1,U2,m1,m2,h1_plus,h2,k1,k2);
        jac = (res_plus - res_minus)/(2*epsilon);
        h1_jac = jac;

        // h2
        double h2_minus = h2 - epsilon;
        double h2_plus = h2 + epsilon;
        res_minus = G1ContinuityModel::getResiduals2(U1,U2,m1,m2,h1,h2_minus,k1,k2);
        res_plus = G1ContinuityModel::getResiduals2(U1,U2,m1,m2,h1,h2_plus,k1,k2);
        jac = (res_plus - res_minus)/(2*epsilon);
        h2_jac = jac;

        // k1
        double k1_minus = k1 - epsilon;
        double k1_plus = k1 + epsilon;
        res_minus = G1ContinuityModel::getResiduals2(U1,U2,m1,m2,h1,h2,k1_minus,k2);
        res_plus = G1ContinuityModel::getResiduals2(U1,U2,m1,m2,h1,h2,k1_plus,k2);
        jac = (res_plus - res_minus)/(2*epsilon);
        k1_jac = jac;

        // k2
        double k2_minus = k2 - epsilon;
        double k2_plus = k2 + epsilon;
        res_minus = G1ContinuityModel::getResiduals2(U1,U2,m1,m2,h1,h2,k1,k2_minus);
        res_plus = G1ContinuityModel::getResiduals2(U1,U2,m1,m2,h1,h2,k1,k2_plus);
        jac = (res_plus - res_minus)/(2*epsilon);
        k2_jac = jac;
    }

    // Compute Minimum Arc Length model residual and jacobian 
    std::pair<Eigen::VectorXd, std::vector<Eigen::Triplet<double>>> MinimumArcLengthModel::compute(const std::vector<double>& hs,
                                                                                                   const std::vector<double>& ks,
                                                                                                   const std::vector<double>& slack,  
                                                                                                   const double& min_arc_length,                                                                     
                                                                                                   int m, int row_offset, const double& mu)
    {
        // std::cout << "[MinimumArcLength Model]" << std::endl;
        int n = static_cast<int>(hs.size());
        Eigen::VectorXd res(m);
        std::vector<int> I; 
        std::vector<int> J;
        std::vector<double> V;

        for (int i=0;i<n;i++)
            MinimumArcLengthModel::computeEach(res, I, J, V, i, hs[i], ks[i], slack[i], min_arc_length, row_offset, mu, n);
        
        std::vector<Eigen::Triplet<double>> triplets;
        for (int i=0;i<I.size();i++)
            triplets.push_back(Eigen::Triplet<double>(I[i], J[i], V[i]));

        std::pair<Eigen::VectorXd, std::vector<Eigen::Triplet<double>>> args(res, triplets);
        return args;
    }
    
    // Compute Minimum Arc Length model residual and jacobian each
    void MinimumArcLengthModel::computeEach(Eigen::VectorXd& res, std::vector<int>& I, std::vector<int>& J, std::vector<double>& V,
                                            int ArcSegNum, const double& h, const double& k, const double& slack, const double& min_arc_length, int row_offset, const double& mu, int n)
    {
        std::pair<int,int> rows(ArcSegNum,ArcSegNum);
        std::pair<int,int> h_cols(8*ArcSegNum+6,8*ArcSegNum+6);
        std::pair<int,int> k_cols(8*ArcSegNum+7,8*ArcSegNum+7);
        std::pair<int,int> s_cols(8*n+ArcSegNum,8*n+ArcSegNum);

        double drdh; double drdk; double drds;
        MinimumArcLengthModel::getJacobians(h,k,slack,min_arc_length,drdh,drdk,drds);

        double res_;
        MinimumArcLengthModel::getResiduals(res_,h,k,slack,min_arc_length);
        res[ArcSegNum] = sqrt(mu) * res_;
        I.push_back(ArcSegNum+row_offset); J.push_back(8*ArcSegNum+6); V.push_back(sqrt(mu)*drdh);
        I.push_back(ArcSegNum+row_offset); J.push_back(8*ArcSegNum+7); V.push_back(sqrt(mu)*drdk);
        I.push_back(ArcSegNum+row_offset); J.push_back(8*n+ArcSegNum); V.push_back(sqrt(mu)*drds);
        // math_tools::sparseFormat(I, J, V, rows, h_cols, sqrt(mu)*drdh, row_offset);
        // math_tools::sparseFormat(I, J, V, rows, k_cols, sqrt(mu)*drdk, row_offset);
        // math_tools::sparseFormat(I, J, V, rows, s_cols, sqrt(mu)*drds, row_offset);
    }

    void MinimumArcLengthModel::getResiduals(double& res, const double& h, const double& k, const double& slack, const double& min_arc_length)
    {
        double r = sqrt(pow(h,4)+pow(k,2));
        double L = 2*sqrt(2)*sqrt(r/(r+abs(k)))*pow(h,2);
        res = 1 - L/min_arc_length + pow(slack,2);
    }

    void MinimumArcLengthModel::getJacobians(const double& h, const double& k, const double& slack, const double& min_arc_length, double& drdh, double& drdk, double& drds)
    {
    // Used symbolic toolbox from MATLAB to compute the jacobians for complex constraint function.
    /*
    clear; clc;
    syms h k s L_min
    r = sqrt(h^4+k^2);
    L = 2*sqrt(2)*sqrt(r/(r+abs(k))) * h^2;
    res = 1 - L/L_min + s^2;
    dLdh = jacobian(res,h);
    dLdk = jacobian(res,k);
    dLds = jacobian(res,s);
    disp([dLdh; dLdk; dLds])
    */
        double sig2 = sqrt(pow(h,4)+pow(k,2));
        drds = 2.0*slack;

        if (k > 0)
        {
            double sig1 = sqrt(sig2/(sig2+k));
            drdh = sqrt(2.0)*pow(h,2)*(2.0*pow(h,3)/pow(k+sig2,2) - 2.0*pow(h,3)/((k+sig2)*sig2)) / (min_arc_length*sig1) - 4.0*sqrt(2.0)*h*sig1/min_arc_length;
            drdk = -sqrt(2)*pow(h,2)*(k/((k+sig2)*sig2) - 1.0/(k+sig2)) / (min_arc_length*sig1);
        }
        else
        {
            double sig1 = sqrt(sig2/(sig2-k));
            drdh = sqrt(2.0)*pow(h,2)*(2.0*pow(h,3)/pow(k-sig2,2) + 2.0*pow(h,3)/((k-sig2)*sig2)) / (min_arc_length*sig1) - 4.0*sqrt(2.0)*h*sig1/min_arc_length;
            drdk = sqrt(2)*pow(h,2)*(k/((k-sig2)*sig2) + 1.0/(k-sig2)) / (min_arc_length*sig1);
        }   

        // double epsilon = 1e-6;
        // double jac;

        // // h
        // double h_minus = h - epsilon;
        // double h_plus = h + epsilon;
        // double res_minus; double res_plus;
        // MinimumArcLengthModel::getResiduals(res_minus,h_minus,k,slack,min_arc_length);
        // MinimumArcLengthModel::getResiduals(res_plus,h_plus,k,slack,min_arc_length);
        // jac = (res_plus - res_minus)/(2*epsilon);
        // drdh = jac;

        // // k
        // double k_minus = k - epsilon;
        // double k_plus = k + epsilon;
        // MinimumArcLengthModel::getResiduals(res_minus,h,k_minus,slack,min_arc_length);
        // MinimumArcLengthModel::getResiduals(res_plus,h,k_plus,slack,min_arc_length);
        // jac = (res_plus - res_minus)/(2*epsilon);
        // drdk = jac;

        // // slack
        // double slack_minus = slack - epsilon;
        // double slack_plus = slack + epsilon;
        // MinimumArcLengthModel::getResiduals(res_minus,h,k,slack_minus,min_arc_length);
        // MinimumArcLengthModel::getResiduals(res_plus,h,k,slack_plus,min_arc_length);
        // jac = (res_plus - res_minus)/(2*epsilon);
        // drds = jac;
    }

}

}

