#pragma once

#include <iostream>
#include <stdexcept>
#include <vector>
#include <Eigen/Dense>
#include <boost/math/distributions/chi_squared.hpp>

namespace math_tools
{
    // Inverse Mahalanobis (Normalization)
    Eigen::MatrixXd invMahalanobis(Eigen::MatrixXd res, Eigen::MatrixXd cov);
    
    // linspace function (similar to MATLAB)
    std::vector<double> linspace(double start, double end, int num_samples); 

    // Creates Skew-Symmetric Matrix from 3d vector
    Eigen::Matrix3d skew(const Eigen::Vector3d& v); 
    
    // Exponential Mapping from 3d vector to SO(3)
    Eigen::Matrix3d Exp_map(Eigen::Vector3d vec);

    // adds triplets to given I, J, V
    void sparseFormat(std::vector<int>& I, std::vector<int>& J, std::vector<double>& V, 
                      std::pair<int,int> row, std::pair<int,int> col, Eigen::MatrixXd value, int row_offset);

    // Compute the dog-leg step vector from give GaussNewton step and GradientDescent step
    Eigen::VectorXd computeDogLeg(Eigen::VectorXd h_gn, Eigen::VectorXd h_gd, double& tr_rad);

    // Update the trust-region radius
    void updateTRrad(double rho, double& tr_rad, double eta1, double eta2, double gamma1, double gamma2);

    bool chi_squared_test(Eigen::Vector3d x_t, Eigen::Vector3d x, Eigen::Matrix3d cov, double degreesOfFreedom, double confidenceLevel);
    
}


namespace options
{
    struct optim_options
    {
        int max_iter = 50;
        double cost_thres = 1e-2;
        double step_thres = 1e-6;
    };

    struct trust_region_options
    {
        double eta1 = 0.5;
        double eta2 = 0.9;
        double gamma1 = 0.1;  
        double gamma2 = 2;
        double thres = 1e-6; // Trust Region radius threshold
        double init = 1e5;
    };

}
