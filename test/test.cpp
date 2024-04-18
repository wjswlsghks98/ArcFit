#include <random>

#include <Utils.h>
#include <SingleArcFit.h>
#include <MultipleArcFit.h>

std::vector<Eigen::Vector2d> get2dCircPoints(double rad, double noise_std, double start_ang, double end_ang, int N);
std::vector<Eigen::Vector3d> get3dCircPoints(double rad, double noise_std, double start_ang, double end_ang, int N);

int main(void)
{
    std::cout << "Running arcfit tests..." << std::endl;
    
    // Generate Random Points
    double std = 0.1;
    std::vector<Eigen::Vector3d> pts = get3dCircPoints(100.0, std, 0.0, 2.0/3 * M_PI, 100);
    // std::vector<Eigen::Vector2d> pts = get2dCircPoints(10.0, std, 0.0, 2.0/3 * M_PI, 100);

    std = 0.1;
    Eigen::VectorXd cov(9);
    cov << std*std, 0, 0, 
           0, std*std, 0,
           0, 0, std*std;

    // Eigen::VectorXd cov(4);
    // cov << std*std, 0,
    //        0, std*std;

    std::vector<Eigen::VectorXd> covs;
    for (int i=0;i<pts.size();i++)
        covs.push_back(cov);

    // threeD::unconstrained::SingleArcFit saf(pts,covs); 
    // // saf.setBasicOptions(100,1e-3,1e-5); // set options if you want to change the default options
    // // saf.setTROptions(0.6,0.85,0.1,2,1e-4,1e8); 
    // saf.fit("iter-detailed");

    // saf.visualize();

    threeD::constrained::MultipleArcFit maf(pts,covs);
    std::pair<std::string,int> option("min",-1);
    maf.fit(option,"iter-detailed");

    // twoD::unconstrained::SingleArcFit saf(pts,covs);
    // saf.visualize();

    return 0;
}

std::vector<Eigen::Vector2d> get2dCircPoints(double rad, double noise_std, double start_ang, double end_ang, int N)
{
    using namespace math_tools;
    std::vector<double> angles = math_tools::linspace(start_ang,end_ang,N);
    std::vector<Eigen::Vector2d> pts;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> distribution(0.0, noise_std);
    double noiseX;
    double noiseY;

    for (int i=0; i<angles.size(); i++)
    {
        noiseX = distribution(gen);
        noiseY = distribution(gen);
        pts.push_back(Eigen::Vector2d(rad*cos(angles[i]) + noiseX, rad*sin(angles[i]) + noiseY));
    }
    return pts;
}

std::vector<Eigen::Vector3d> get3dCircPoints(double rad, double noise_std, double start_ang, double end_ang, int N)
{
    using namespace math_tools;
    std::vector<double> angles = math_tools::linspace(start_ang,end_ang,N);
    std::vector<Eigen::Vector3d> pts;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> distribution(0.0, noise_std);
    double noiseX;
    double noiseY;
    double noiseZ;

    for (int i=0; i<angles.size(); i++)
    {
        noiseX = distribution(gen);
        noiseY = distribution(gen);
        noiseZ = distribution(gen);
        pts.push_back(Eigen::Vector3d(rad*cos(angles[i]) + noiseX, rad*sin(angles[i]) + noiseY, noiseZ));
    }
    return pts;
}

