#pragma once

#include <algorithm>
#include <SingleArcFit.h>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/geometries/linestring.hpp>
#include <boost/geometry/index/rtree.hpp>

namespace bg = boost::geometry;
// namespace bgi = boost::geometry::index;
namespace threeD
{

typedef bg::model::point<double,3, bg::cs::cartesian> bgPoint;
typedef bg::model::linestring<bgPoint> bgLineString;
// typedef std::pair<bgPoint,unsigned> value;

namespace constrained
{

class MultipleArcFit : public unconstrained::SingleArcFit
{
public:
    struct parameters
    {
        std::vector<Eigen::Vector3d> ms;
        std::vector<Eigen::Matrix3d> Us;
        std::vector<double> hs;
        std::vector<double> ks;
        std::vector<double> slack;
        std::vector<std::pair<int,int>> intvs;
    };

    MultipleArcFit(std::vector<Eigen::Vector3d> pts_, std::vector<Eigen::VectorXd> covs_);
    void fit(std::pair<std::string,int> option, const std::string& verbose = "none");
    void visualize(void) override;
    MultipleArcFit::parameters getParams(void);

private:
    parameters params;
    Eigen::VectorXd z; // lagrange multiplier
    double mu; // penalty parameter
    double min_arc_length = 10.0; // minimum arc length to prevent singularity

    void initialize(void) override;
    int getParamNum(void) override;
    std::vector<Eigen::Vector3d> getNodes(void);
    void constrained_fit(const std::string& verbose);
    std::pair<Eigen::VectorXd, Eigen::SparseMatrix<double>> cost_func(Eigen::VectorXd dx) override;
    void retract(Eigen::VectorXd dx) override;
    void associate(void);
    void getConstraintVec(Eigen::VectorXd& constraint_res);
    std::vector<int> checkApproxValidity(std::vector<std::vector<int>> intvs, std::vector<Eigen::Vector3d> ms_, std::vector<Eigen::Matrix3d> Us_, 
                                         std::vector<double> hs_, std::vector<double> ks_, const std::string verbose);
    std::vector<int> getLinearIntvs(std::pair<int,int> intv, int depth);

};

}

}

