#include <Utils.h>

namespace math_tools
{
    // Inverse Mahalanobis (Normalization)
    Eigen::MatrixXd invMahalanobis(Eigen::MatrixXd res, Eigen::MatrixXd cov)
    {
        int v_size = res.rows();
        int rows = cov.rows(); int cols = cov.cols();
        Eigen::MatrixXd ret;
        try
        {
            if (v_size != rows)
                throw std::runtime_error("Size of residual and covariance matrix does not match.");
            else if(rows != cols)
                throw std::runtime_error("Covariance matrix should be square matrix.");
            else
            {
                int n = rows;
                Eigen::LLT<Eigen::MatrixXd> chol_cov(cov);
                Eigen::MatrixXd L = chol_cov.matrixL();
                Eigen::MatrixXd invL = L.inverse();
                ret = invL.transpose() * res;
            }
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << '\n';
        }
        return ret;
    }

    // linspace function (similar to MATLAB)
    std::vector<double> linspace(double start, double end, int num_samples) 
    {
        std::vector<double> result;
        if (num_samples <= 1) {
            if (num_samples == 1) {
                result.push_back(start);
            }
            return result;
        }
        result.reserve(num_samples);
        double step = (end - start) / (num_samples - 1);
        for (int i = 0; i < num_samples; ++i) {
            result.push_back(start + step * i);
        }
        return result;
    }

    // Creates Skew-Symmetric Matrix from 3d vector
    Eigen::Matrix3d skew(const Eigen::Vector3d& v) 
    {
        Eigen::Matrix3d skew_symmetric_matrix;
        skew_symmetric_matrix <<  0, -v[2],  v[1],
                                  v[2],  0, -v[0],
                                 -v[1],  v[0],  0;
        return skew_symmetric_matrix;
    }

    // Exponential Mapping from 3d vector to SO(3)
    Eigen::Matrix3d Exp_map(Eigen::Vector3d vec)
    {
        double mag = vec.norm();
        Eigen::Matrix3d S = math_tools::skew(vec);
        if (mag < 1e-6)
            return Eigen::Matrix3d::Identity() + S;
        else
        {
            double one_minus_cos = 2 * sin(mag/2) * sin(mag/2);
            return Eigen::Matrix3d::Identity() + sin(mag)/mag * S + one_minus_cos/std::pow(mag,2) * S*S;
        }
    }

    // adds triplets to given I, J, V
    void sparseFormat(std::vector<int>& I, std::vector<int>& J, std::vector<double>& V, 
                      std::pair<int,int> row, std::pair<int,int> col, Eigen::MatrixXd value, int row_offset)
    {
        try
        {
            int row_lb = row.first; int row_ub = row.second;
            int col_lb = col.first; int col_ub = col.second;

            if (row_ub - row_lb + 1 !=  value.rows() || col_ub - col_lb + 1 != value.cols())
                throw std::runtime_error("Number of total elements are not matched for input row / col boundaries and jacobian matrix");

            else
            {
                for (int i=row_lb; i<= row_ub; i++)
                {
                    for (int j=col_lb; j<=col_ub; j++)
                    {
                        I.push_back(i + row_offset); 
                        J.push_back(j); 
                        V.push_back(value(i-row_lb,j-col_lb));
                    }
                }
            }

        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << '\n';
        }
        
    }

    Eigen::VectorXd computeDogLeg(Eigen::VectorXd h_gn, Eigen::VectorXd h_gd, double& tr_rad)
    {

        double GN_step = h_gn.norm();
        double GD_step = h_gd.norm();
        
        if(GN_step <= tr_rad)
            return h_gn;
        else if (GD_step >= tr_rad)
            return (tr_rad/GD_step) * h_gd;
        else
        {
            Eigen::VectorXd v = h_gn - h_gd;
            double hgdv = h_gd.transpose() * v;
            double vsq = v.transpose() * v;
            double beta = (-hgdv + sqrt(hgdv*hgdv + (tr_rad*tr_rad - GD_step*GD_step) * vsq)) / vsq;
            return h_gd + beta * v;
        }

    }

    void updateTRrad(double rho, double& tr_rad, double eta1, double eta2, double gamma1, double gamma2)
    {
        if (rho >= eta2)
            tr_rad *= gamma2;
        else if (rho < eta1)
            tr_rad *= gamma1;
    }

    bool chi_squared_test(Eigen::Vector3d x_t, Eigen::Vector3d x, Eigen::Matrix3d cov, double degreesOfFreedom, double confidenceLevel)
    {
        Eigen::Vector3d residual_vec = x_t - x;
        
        // std::cout << "Covariance matrix: " << std::endl;
        // std::cout << cov << std::endl;
        // std::cout << std::endl;
        // std::cout << "Actual Residual: " << std::endl;
        // std::cout << residual_vec << std::endl;

        double chi_squared_dist = residual_vec.transpose() * cov.inverse() * residual_vec;
        double chi_squared_thres = boost::math::quantile(boost::math::chi_squared(degreesOfFreedom), confidenceLevel);
        // std::cout << "Chisq Dist: " << chi_squared_dist << ", Chisq Thres: " << chi_squared_thres << std::endl;

        if (chi_squared_thres < chi_squared_dist)
            return false;
        else
            return true;
    }
    
}