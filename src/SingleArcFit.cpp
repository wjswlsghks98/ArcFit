#include <iomanip>
#include <iostream>
#include <sstream>
#include <SingleArcFit.h>

namespace twoD
{
namespace unconstrained
{

// Constructor + initializer
SingleArcFit::SingleArcFit(std::vector<Eigen::Vector2d> pts_, std::vector<Eigen::VectorXd> covs_): pts(pts_), covs(covs_) { initialize(); }

void SingleArcFit::initialize(void)
{
    params.m = 0.5 * (pts.front() + pts.back());
    double hsq = 0.5 * (pts.front() - pts.back()).norm();

    Eigen::Vector2d r1 = (pts.front() - params.m)/hsq;
    params.theta = atan2(r1[1],r1[0]);
    
    params.h = sqrt(hsq);
    double dist = (pts[pts.size()/2] - params.m).norm();
    params.k = (pow(hsq,2)-pow(dist,2))/(2*dist);
}

// Single Arc Approximation
// void SingleArcFit::fit(const std::string& verbose)
// {
//     double tr_rad = tr_options.init;

//     Eigen::VectorXd dx = Eigen::VectorXd::Zero(getParamNum());
//     std::pair<Eigen::VectorXd, Eigen::SparseMatrix<double>> out = cost_func(dx);
//     Eigen::VectorXd res = out.first;
//     Eigen::SparseMatrix<double> jac = out.second;
//     double cost = res.transpose() * res;
//     double prev_cost = cost;

//     if (verbose == "iter-detailed")
//     {
//         std::cout <<"   Iteration       Cost           Step         TR radius       Acceptance" << std::endl;
//         std::printf("     %3d         %.2e       %.2e        %.2e          %s\n",0,cost,dx.norm(),tr_rad,"Init");
//     }

//     Eigen::SparseMatrix<double> A;
//     Eigen::VectorXd b;
//     Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
//     Eigen::VectorXd h_gn;
//     Eigen::VectorXd h_gd;
//     double alpha;
//     double den; double num;
//     double ared; double pred;
//     double bx0; double x0Ax0;
//     double rho;

//     Eigen::VectorXd res_;
//     Eigen::SparseMatrix<double> jac_;
//     std::string validity;
//     bool flag;
//     std::ostringstream oss;

//     for (int i=1;i<=basic_options.max_iter;i++)
//     {
//         A = jac.transpose() * jac;
//         b = -jac.transpose() * res;

//         solver.compute(A);
//         if (solver.info() != Eigen::Success) {
//             std::cerr << "Decomposition failed" << std::endl;
//         }

//         h_gn = solver.solve(b); // Gauss-Newton Step
//         if (solver.info() != Eigen::Success) {
//             std::cerr << "Solving failed" << std::endl;
//         }
//         num = b.transpose() * b;
//         den = b.transpose() * A  * b;
//         alpha = num / den;
//         h_gd = alpha * b; // Gradient Descent Step

//         dx = math_tools::computeDogLeg(h_gn,h_gd,tr_rad);

//         parameters dummy_params = params;

//         out = cost_func(dx);
//         res_ = out.first;
//         jac_ = out.second;
//         cost = res_.transpose() * res_;

//         ared = prev_cost - cost;
//         bx0 = b.transpose() * dx;
//         x0Ax0 = dx.transpose() * A * dx;
//         rho = ared/(bx0 - 0.5 * x0Ax0);

//         if (rho >= tr_options.eta1) // Current step accepted. Update loop variables
//         {
//             res = res_;
//             jac = jac_;
//             prev_cost = cost;
//             validity = "Accepted";
//             flag = true;
//         }
//         else
//         {
//             params = dummy_params;
//             validity = "Rejected";
//             flag = false;
//         }


//         if (verbose == "iter-detailed")
//             std::printf("     %3d         %.2e       %.2e        %.2e        %s\n",i,cost,dx.norm(),tr_rad,validity.c_str());

//         math_tools::updateTRrad(rho,tr_rad,tr_options.eta1,tr_options.eta2,tr_options.gamma1,tr_options.gamma2);

//         if (flag) // Check ending criterion for accepted steps
//         {
//             if (abs(ared) < basic_options.cost_thres)
//             {
//                 retract(dx);
//                 oss << "Current cost difference " << abs(ared) << " is below threshold: " << basic_options.cost_thres << std::endl;
//                 break;
//             }
//             else if (dx.norm() < basic_options.step_thres)
//             {
//                 retract(dx);
//                 oss << "Current step size " << dx.norm() << " is below threshold: " << basic_options.step_thres << std::endl;
//                 break;
//             }
//             else if (i == basic_options.step_thres)
//             {
//                 retract(dx);
//                 oss << "Current iteration number " << i+1 << " is above threshold: " << basic_options.max_iter << std::endl;
//                 break;
//             }
//             else if (tr_rad < tr_options.thres)
//             {
//                 retract(dx);
//                 oss << "Current trust region radius " << tr_rad << " is below threshold: " << tr_options.thres << std::endl;
//                 break;
//             }
//         }
//         else if (tr_rad < tr_options.thres)
//         {
//             retract(dx);
//             oss << "Current trust region radius " << tr_rad << " is below threshold: " << tr_options.thres << std::endl;
//             break;
//         }
//     }
//     if (verbose == "iter-detailed")
//     {
//         std::string output = oss.str();
//         std::cout << output;
//     }
// }

// Set basic options (Use if default options are not preferred)
void SingleArcFit::setBasicOptions(int max_iter_, double cost_thres_, double step_thres_)
{
    basic_options.max_iter = max_iter_;
    basic_options.cost_thres = cost_thres_;
    basic_options.step_thres = step_thres_;
}

// Set trust region options (Use if default options are not preferred)
void SingleArcFit::setTROptions(double eta1_, double eta2_, double gamma1_, double gamma2_, double thres_, double init_)
{
    tr_options.eta1 = eta1_;
    tr_options.eta2 = eta2_;
    tr_options.gamma1 = gamma1_;
    tr_options.gamma2 = gamma2_;
    tr_options.thres = thres_;
    tr_options.init = init_;
}

// Visualize Fitting Results
void SingleArcFit::visualize(void)
{
    using namespace matplot;
    std::vector<double> pt_x;
    std::vector<double> pt_y;

    for (int i=0;i<pts.size();i++)
    {
        pt_x.push_back(pts[i].x());
        pt_y.push_back(pts[i].y());
    }

    plot(pt_x,pt_y,"o");
    hold(on);

    std::vector<double> u = math_tools::linspace(0.0, 1.0, 1e3);

    Eigen::Vector2d m = params.m;
    Eigen::Matrix2d R;
    R << cos(params.theta), -sin(params.theta),
         sin(params.theta), cos(params.theta);

    double h = params.h; double k = params.k;
    double r = sqrt(h*h*h*h + k*k);
    double w = k/r;
    Eigen::Vector2d e1(1,0);
    Eigen::Vector2d e2(0,1);

    Eigen::Vector2d A = m + h*h * R * e1;
    Eigen::Vector2d B = m - h*h * R * e1;
    Eigen::Vector2d C = m + h*h*h*h/k * R * e2;
    Eigen::Vector2d Xc = m - k * R * e2;

    std::vector<double> xc_x; xc_x.push_back(Xc.x()); xc_x.push_back(pts[pts.size()/2].x());
    std::vector<double> xc_y; xc_y.push_back(Xc.y()); xc_y.push_back(pts[pts.size()/2].y());
    plot(xc_x,xc_y,"sg");

    std::vector<double> approx_x;
    std::vector<double> approx_y;
    
    Eigen::Vector2d approx;

    for (int i=0;i<u.size();i++)
    {
        approx = (pow(1-u[i],2) * A + 2*u[i]*(1-u[i])*w * C + pow(u[i],2) * B) / (pow(1-u[i],2) + 2*u[i]*(1-u[i])*w + pow(u[i],2));
        approx_x.push_back(approx.x());
        approx_y.push_back(approx.y());
    }
    plot(approx_x,approx_y,"-.r");
    xlabel("X"); ylabel("Y");
    axis(equal);
    hold(off);
    show();
    
}

// get Params
SingleArcFit::parameters SingleArcFit::getParams(void)
{
    return params;
}

// Obtain the number of parameters before optimization
int SingleArcFit::getParamNum(void)
{
    int a = 5;
    return a;
}



}

}

namespace threeD
{

namespace unconstrained
{

// Constructor + initializer
SingleArcFit::SingleArcFit(const std::vector<Eigen::Vector3d> pts_, const std::vector<Eigen::VectorXd> covs_): pts(pts_), covs(covs_) { initialize(); }

// Initialize optimizer
void SingleArcFit::initialize(void)
{
    params.m = 0.5 * (pts.front() + pts.back());
    double hsq = 0.5 * (pts.front() - pts.back()).norm();

    Eigen::Vector3d e1 = (pts.front() - params.m)/hsq;
    Eigen::Vector3d e2 = (pts[pts.size()/2] - params.m).normalized();
    Eigen::Vector3d e3 = e1.cross(e2);
    params.U << e1, e2, e3;
    params.h = sqrt(hsq);
    double dist = (pts[pts.size()/2] - params.m).norm();
    params.k = (pow(hsq,2)-pow(dist,2))/(2*dist);
}

// Single Arc Approximation
void SingleArcFit::fit(const std::string& verbose)
{
    double tr_rad = tr_options.init;

    Eigen::VectorXd dx = Eigen::VectorXd::Zero(getParamNum());
    std::pair<Eigen::VectorXd, Eigen::SparseMatrix<double>> out = cost_func(dx);
    Eigen::VectorXd res = out.first;
    Eigen::SparseMatrix<double> jac = out.second;
    double cost = res.transpose() * res;
    double prev_cost = cost;

    if (verbose == "iter-detailed")
    {
        std::cout <<"   Iteration       Cost           Step         TR radius       Acceptance" << std::endl;
        std::printf("     %3d         %.2e       %.2e        %.2e          %s\n",0,cost,dx.norm(),tr_rad,"Init");
    }

    Eigen::SparseMatrix<double> A;
    Eigen::VectorXd b;
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    Eigen::VectorXd h_gn;
    Eigen::VectorXd h_gd;
    double alpha;
    double den; double num;
    double ared; double pred;
    double bx0; double x0Ax0;
    double rho;

    Eigen::VectorXd res_;
    Eigen::SparseMatrix<double> jac_;
    std::string validity;
    bool flag;
    std::ostringstream oss;

    for (int i=1;i<=basic_options.max_iter;i++)
    {
        A = jac.transpose() * jac;
        b = -jac.transpose() * res;

        solver.compute(A);
        if (solver.info() != Eigen::Success) {
            std::cerr << "Decomposition failed" << std::endl;
        }

        h_gn = solver.solve(b); // Gauss-Newton Step
        if (solver.info() != Eigen::Success) {
            std::cerr << "Solving failed" << std::endl;
        }
        num = b.transpose() * b;
        den = b.transpose() * A  * b;
        alpha = num / den;
        h_gd = alpha * b; // Gradient Descent Step

        dx = math_tools::computeDogLeg(h_gn,h_gd,tr_rad);

        parameters dummy_params = params;

        out = cost_func(dx);
        res_ = out.first;
        jac_ = out.second;
        cost = res_.transpose() * res_;

        ared = prev_cost - cost;
        bx0 = b.transpose() * dx;
        x0Ax0 = dx.transpose() * A * dx;
        rho = ared/(bx0 - 0.5 * x0Ax0);

        if (rho >= tr_options.eta1) // Current step accepted. Update loop variables
        {
            res = res_;
            jac = jac_;
            prev_cost = cost;
            validity = "Accepted";
            flag = true;
        }
        else
        {
            params = dummy_params;
            validity = "Rejected";
            flag = false;
        }


        if (verbose == "iter-detailed")
            std::printf("     %3d         %.2e       %.2e        %.2e        %s\n",i,cost,dx.norm(),tr_rad,validity.c_str());

        math_tools::updateTRrad(rho,tr_rad,tr_options.eta1,tr_options.eta2,tr_options.gamma1,tr_options.gamma2);

        if (flag) // Check ending criterion for accepted steps
        {
            if (abs(ared) < basic_options.cost_thres)
            {
                retract(dx);
                oss << "Current cost difference " << abs(ared) << " is below threshold: " << basic_options.cost_thres << std::endl;
                break;
            }
            else if (dx.norm() < basic_options.step_thres)
            {
                retract(dx);
                oss << "Current step size " << dx.norm() << " is below threshold: " << basic_options.step_thres << std::endl;
                break;
            }
            else if (i == basic_options.step_thres)
            {
                retract(dx);
                oss << "Current iteration number " << i+1 << " is above threshold: " << basic_options.max_iter << std::endl;
                break;
            }
            else if (tr_rad < tr_options.thres)
            {
                retract(dx);
                oss << "Current trust region radius " << tr_rad << " is below threshold: " << tr_options.thres << std::endl;
                break;
            }
        }
        else if (tr_rad < tr_options.thres)
        {
            if (flag)
                retract(dx);
            oss << "Current trust region radius " << tr_rad << " is below threshold: " << tr_options.thres << std::endl;
            break;
        }
    }
    if (verbose == "iter-detailed")
    {
        std::string output = oss.str();
        std::cout << output;
    }
}

// Set basic options (Use if default options are not preferred)
void SingleArcFit::setBasicOptions(int max_iter_, double cost_thres_, double step_thres_)
{
    basic_options.max_iter = max_iter_;
    basic_options.cost_thres = cost_thres_;
    basic_options.step_thres = step_thres_;
}

// Set trust region options (Use if default options are not preferred)
void SingleArcFit::setTROptions(double eta1_, double eta2_, double gamma1_, double gamma2_, double thres_, double init_)
{
    tr_options.eta1 = eta1_;
    tr_options.eta2 = eta2_;
    tr_options.gamma1 = gamma1_;
    tr_options.gamma2 = gamma2_;
    tr_options.thres = thres_;
    tr_options.init = init_;
}

// Visualize Fitting Results
void SingleArcFit::visualize(void)
{
    using namespace matplot;
    std::vector<double> pt_x;
    std::vector<double> pt_y;
    std::vector<double> pt_z;

    for (int i=0;i<pts.size();i++)
    {
        pt_x.push_back(pts[i].x());
        pt_y.push_back(pts[i].y());
        pt_z.push_back(pts[i].z());
    }

    plot3(pt_x,pt_y,pt_z,"o");
    hold(on);

    std::vector<double> u = math_tools::linspace(0.0, 1.0, 1e3);

    Eigen::Vector3d m = params.m;
    Eigen::Matrix3d U = params.U;
    double h = params.h; double k = params.k;
    double r = sqrt(h*h*h*h + k*k);
    double w = k/r;
    Eigen::Vector3d e1(1,0,0);
    Eigen::Vector3d e2(0,1,0);

    Eigen::Vector3d A = m + h*h * U * e1;
    Eigen::Vector3d B = m - h*h * U * e1;
    Eigen::Vector3d C = m + h*h*h*h/k * U * e2;

    std::vector<double> approx_x;
    std::vector<double> approx_y;
    std::vector<double> approx_z;
    
    Eigen::Vector3d approx;

    for (int i=0;i<u.size();i++)
    {
        approx = (pow(1-u[i],2) * A + 2*u[i]*(1-u[i])*w * C + pow(u[i],2) * B) / (pow(1-u[i],2) + 2*u[i]*(1-u[i])*w + pow(u[i],2));
        approx_x.push_back(approx.x());
        approx_y.push_back(approx.y());
        approx_z.push_back(approx.z());
    }
    plot3(approx_x,approx_y,approx_z,"-.r");
    xlabel("X"); ylabel("Y"); zlabel("Z");
    axis(equal);
    hold(off);
    show();
    
}

// get Params
SingleArcFit::parameters SingleArcFit::getParams(void)
{
    return params;
}

// Obtain the number of parameters before optimization
int SingleArcFit::getParamNum(void)
{
    int a = 8;
    return a;
}

// Compute residual vector and jacobian matrix
std::pair<Eigen::VectorXd, Eigen::SparseMatrix<double>> SingleArcFit::cost_func(Eigen::VectorXd dx)
{
    retract(dx);
    std::vector<Eigen::Matrix3d> Us; Us.push_back(params.U);
    std::vector<Eigen::Vector3d> ms; ms.push_back(params.m);
    std::vector<double> hs; hs.push_back(params.h);
    std::vector<double> ks; ks.push_back(params.k);

    // Anchor Model
    std::vector<int> AnchorIdxs = {0, static_cast<int>(pts.size())-1};
    std::vector<int> ArcSegNums = {0, 0};
    std::vector<std::string> types = {"front", "back"};

    int row_offset = 0;
    AnchorModel ac_model(pts);
    std::pair<Eigen::VectorXd, std::vector<Eigen::Triplet<double>>> anchor_ = ac_model.compute(AnchorIdxs,ArcSegNums,types,
                                                                                               Us,ms,hs,3*2,row_offset);

    std::vector<Eigen::Triplet<double>> triplets = anchor_.second;

    row_offset += 3*2; 

    // Measurement Model
    std::vector<std::pair<int,int>> intvs;
    std::pair<int,int> intv(0,static_cast<int>(pts.size())-1);
    intvs.push_back(intv);

    MeasurementModel me_model(pts, covs);
    std::pair<Eigen::VectorXd, std::vector<Eigen::Triplet<double>>> measurement_ = me_model.compute(intvs,Us,ms,hs,ks,
                                                                                                    3*pts.size(),row_offset);
    
    // Concatenate triplets for sparse matrix generation
    triplets.insert(triplets.end(), measurement_.second.begin(), measurement_.second.end());

    // Vertically concatenate residual vector
    Eigen::VectorXd cost_res(anchor_.first.size() + measurement_.first.size());
    cost_res << anchor_.first, measurement_.first;

    // Generate sparse matrix with concatenated triplets
    Eigen::SparseMatrix<double> cost_jac(cost_res.size(),3+3+1+1);
    cost_jac.setFromTriplets(triplets.begin(), triplets.end());

    std::pair<Eigen::VectorXd, Eigen::SparseMatrix<double>> args(cost_res, cost_jac);

    return args;
}

// Retract optimization variables on manifold
void SingleArcFit::retract(Eigen::VectorXd dx)
{
    Eigen::Vector3d m_delta = dx.segment(0,3);
    Eigen::Vector3d U_delta = dx.segment(3,3);
    double h_delta = dx[6];
    double k_delta = dx[7];

    params.m += m_delta;
    params.U *= math_tools::Exp_map(U_delta);
    params.h += h_delta;
    params.k += k_delta;
}

}
}
