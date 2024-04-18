#pragma once

#include <matplot/matplot.h>
#include <Models.h>

namespace twoD
{

namespace unconstrained
{

class SingleArcFit
{
public:
    struct parameters
    {
        Eigen::Vector2d m;
        double theta;
        double h;
        double k;
    };

    const std::vector<Eigen::Vector2d> pts;
    const std::vector<Eigen::VectorXd> covs;
    options::optim_options basic_options;
    options::trust_region_options tr_options;

    SingleArcFit(std::vector<Eigen::Vector2d> pts_, std::vector<Eigen::VectorXd> covs_);
    // void fit(const std::string& verbose = "none");
    void setBasicOptions(int max_iter_, double cost_thres_, double step_thres_);
    void setTROptions(double eta1_, double eta2_, double gamma1_, double gamma2_, double thres_, double init_);
    virtual void visualize(void);
    SingleArcFit::parameters getParams(void);

private:
    parameters params;

    virtual void initialize(void);
    virtual int getParamNum(void);
    // virtual std::pair<Eigen::VectorXd, Eigen::SparseMatrix<double>> cost_func(Eigen::VectorXd dx);
    // virtual void retract(Eigen::VectorXd dx);

};

}

}

namespace threeD
{

namespace unconstrained
{

class SingleArcFit
{
public:
    struct parameters
    {
        Eigen::Vector3d m;
        Eigen::Matrix3d U; // Defines rotational matrix
        double h;
        double k;
    };

    const std::vector<Eigen::Vector3d> pts;
    const std::vector<Eigen::VectorXd> covs;
    options::optim_options basic_options;
    options::trust_region_options tr_options;

    SingleArcFit(std::vector<Eigen::Vector3d> pts_, std::vector<Eigen::VectorXd> covs_);
    void fit(const std::string& verbose = "none");
    void setBasicOptions(int max_iter_, double cost_thres_, double step_thres_);
    void setTROptions(double eta1_, double eta2_, double gamma1_, double gamma2_, double thres_, double init_);
    virtual void visualize(void);
    SingleArcFit::parameters getParams(void);
    
private:

    parameters params;

    virtual void initialize(void);
    virtual int getParamNum(void);
    virtual std::pair<Eigen::VectorXd, Eigen::SparseMatrix<double>> cost_func(Eigen::VectorXd dx);
    virtual void retract(Eigen::VectorXd dx);

};

}

}


// namespace constrained
// {

// class SingleArcFit
// {
// private:
//     Eigen::MatrixXd pts;
//     Eigen::MatrixXd covs;
//     struct params
//     {
//         std::vector<Eigen::Vector3d> arcNodes;
//         Eigen::Vector3d midNode;
//     };
// public:
//     SingleArcFit(Eigen::MatrixXd pts_, Eigen::MatrixXd covs_): pts(pts_), covs(covs_) { }


// };

// }
