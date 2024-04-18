#include <MultipleArcFit.h>

namespace threeD
{

namespace constrained
{

MultipleArcFit::MultipleArcFit(std::vector<Eigen::Vector3d> pts_, std::vector<Eigen::VectorXd> covs_): unconstrained::SingleArcFit(pts_, covs_) { initialize(); }

void MultipleArcFit::initialize(void)
{    
    // Find the initial number of arc segments for efficient approximation
    threeD::unconstrained::SingleArcFit oneArc(pts, covs);
    // oneArc.visualize();
    oneArc.fit("iter-detailed");
    oneArc.visualize();
    //  
    int max_invalidN = 20;

    threeD::unconstrained::SingleArcFit::parameters params_ = oneArc.getParams();
    std::vector<Eigen::Vector3d> ms_; ms_.push_back(params_.m);
    std::vector<Eigen::Matrix3d> Us_; Us_.push_back(params_.U);
    std::vector<double> hs_; hs_.push_back(params_.h);
    std::vector<double> ks_; ks_.push_back(params_.k);

    std::pair<int,int> intv(0,static_cast<int>(pts.size())-1);
    std::vector<std::vector<int>> intvs = {{0,static_cast<int>(pts.size())-1}};
    std::vector<int> invalidN = checkApproxValidity(intvs,ms_,Us_,hs_,ks_,"none");

    if (invalidN[0] < max_invalidN) // Maximum number of invalid approximations not exceeded: valid with one arc
    {
        params.ms = ms_; params.Us = Us_; params.hs = hs_; params.ks = ks_;
    }
    else // Need more than one arc for approximation
    {   
        std::vector<int> intv_ = getLinearIntvs(intv,0);
        auto it = std::unique(intv_.begin(),intv_.end());
        intv_.erase(it, intv_.end());

        int lb = 0; int ub = 1;
        std::vector<std::vector<int>> circle_intvs;

        while (ub <= static_cast<int>(intv_.size())-1)
        {
            
            int lb_idx = intv_[lb];
            int ub_idx = intv_[ub];
            // std::cout << "[" << lb_idx << ", " << ub_idx << "]" << std::endl;

            if (ub_idx - lb_idx + 1 <= 3)
            {
                ub += 1;
                ub_idx = intv_[ub];
            }

            std::vector<Eigen::Vector3d> sampled_pts;
            std::vector<Eigen::VectorXd> sampled_covs;

            for (int i=lb_idx;i<=ub_idx;i++)
            {
                sampled_pts.push_back(pts[i]);
                sampled_covs.push_back(covs[i]);
            }
            threeD::unconstrained::SingleArcFit sample(sampled_pts,sampled_covs);
            sample.fit("none");

            threeD::unconstrained::SingleArcFit::parameters params_ = sample.getParams();
            std::vector<Eigen::Vector3d> ms_; ms_.push_back(params_.m);
            std::vector<Eigen::Matrix3d> Us_; Us_.push_back(params_.U);
            std::vector<double> hs_; hs_.push_back(params_.h);
            std::vector<double> ks_; ks_.push_back(params_.k);
            std::vector<std::vector<int>> intvs_ = {{0,static_cast<int>(sampled_pts.size())-1}};
            std::vector<int> invalidN_ = checkApproxValidity(intvs_,ms_,Us_,hs_,ks_,"none");

            if (invalidN_[0] < max_invalidN) // Current arc approximation is valid. --> Move upper bound.
            {
                ub += 1;
                if (ub == static_cast<int>(intv_.size()-1))
                {
                    ub_idx = intv_[ub];
                    circle_intvs.push_back({lb_idx, ub_idx});
                    break;
                }
            }
            else
            {
                if (lb!= ub-1)
                {
                    circle_intvs.push_back({lb_idx, intv_[ub-1]});
                    lb = ub-1;
                }
                else
                {
                    // If one linear interval cannot be represented by a single arc, 
                    // just assume we can, just for initialization step.
                    circle_intvs.push_back({lb_idx, intv_[ub]});
                    lb = ub; ub = lb + 1;
                }
            }
        }
        // After initial segmentation, save the parameters.
        for (int i=0;i<circle_intvs.size();i++)
        {
            std::pair<int,int> seg_intv(circle_intvs[i].front(),circle_intvs[i].back());
            params.intvs.push_back(seg_intv);

            std::vector<Eigen::Vector3d> sampled_pts;
            std::vector<Eigen::VectorXd> sampled_covs;

            for (int j=circle_intvs[i].front();j<=circle_intvs[i].back();j++)
            {
                sampled_pts.push_back(pts[j]);
                sampled_covs.push_back(covs[j]);
            }
            threeD::unconstrained::SingleArcFit sample(sampled_pts,sampled_covs);
            sample.fit("none");
            threeD::unconstrained::SingleArcFit::parameters sampled_params = sample.getParams();
            
            params.ms.push_back(sampled_params.m);
            params.Us.push_back(sampled_params.U);
            params.hs.push_back(sampled_params.h);
            params.ks.push_back(sampled_params.k);
        }
    }



}

void MultipleArcFit::fit(std::pair<std::string,int> option, const std::string& verbose)
{
    /*
    This is a overloaded function of 'fit' method from the SingleArcFit class. 
    'fit' method inherited from the base will be called multiple times for performing constrained optimization.

    There are three modes supported in the input 'option'

    1. min
        Perform multiple arc approximation from a single arc segment. This may be useful when trying to obtain the least number of arc segments.
    But may fall into unwanted local minima, or even diverge for extreme cases. Also this method is not recommended for approximation of large 
    dataset (it may take too long time). 

    2. max 
        Perform multiple arc approximation with segment number upper bound. Although if the validity conditions are not met, this method ignores 
    and forces finish in the approximation process. If the number of minimum segments computed during the initialization process exceeds the 
    upper bound, we perform approximation in 'min' mode. If not, we start approximation with the pre-processed initial number of arc segments.

    3. default
        Performs multiple arc approximation with no limitation in the number of arc segments. The initial data points are processed to obtain the
    initial number of arc segments. This enhances the overall speed of the approximation process. 
    
    ** Optimization Scheme
    Augmented Lagragian is implemented for performing constrained nonlinear least squares optimization.

    Perhaps need to implement methods to set options of augmented lagrangian.
    
    */
    int max_num;

    if (option.first == "max")
    {
        // If max_num < number of initialized arc segments, need to re-initialize the parameters.
        max_num = option.second;
        if (params.intvs.size() > max_num)
            option.first = "min";
    }

    if (option.first == "min")
    {
        // Re-initialize the intervals
        std::pair<int,int> intv(0,static_cast<int>(pts.size())-1);
        std::vector<std::pair<int,int>> intvs; intvs.push_back(intv);
        params.intvs = intvs;

        unconstrained::SingleArcFit oneArc(pts,covs);
        unconstrained::SingleArcFit::parameters params_ = oneArc.getParams();
        std::vector<Eigen::Vector3d> ms_; ms_.push_back(params_.m);
        std::vector<Eigen::Matrix3d> Us_; Us_.push_back(params_.U);
        std::vector<double> hs_; hs_.push_back(params_.h);
        std::vector<double> ks_; ks_.push_back(params_.k);
        std::vector<double> slack_; slack_.push_back(sqrt(min_arc_length));

        params.ms = ms_; params.Us = Us_; params.hs = hs_; params.ks = ks_;
        params.slack = slack_;
    }
    else // Any other input string will be recognized as default mode.
    {
        parameters params_;

        // Initialize the parameters with the computed intervals
        for (std::pair<int,int> intv : params.intvs)
        {
            int lb = intv.first; int ub = intv.second;
            std::cout << "lb: " << lb << ", ub: " << ub << std::endl;
            std::vector<Eigen::Vector3d> sampled_pts;
            std::vector<Eigen::VectorXd> sampled_covs;

            for (int i=lb;i<=ub;i++)
            {
                sampled_pts.push_back(pts[i]);
                sampled_covs.push_back(covs[i]);
            }
            threeD::unconstrained::SingleArcFit sample(sampled_pts,sampled_covs);
            sample.fit("none");
            // sample.visualize();
            threeD::unconstrained::SingleArcFit::parameters sampled_params = sample.getParams();
            params_.ms.push_back(sampled_params.m); params_.Us.push_back(sampled_params.U);
            params_.hs.push_back(sampled_params.h); params_.ks.push_back(sampled_params.k);
            params_.slack.push_back(1e-2);
        }
        params.ms = params_.ms; params.Us = params_.Us; params.hs = params_.hs; params.ks = params_.ks;
        params.slack = params_.slack;
    }
    
    int n;
    // for loop if maximum exists
    n = static_cast<int>(params.ms.size());

    mu = 1.0;
    z = Eigen::VectorXd::Zero(3*(n-1) + 3*(n-1) + n);

    // Repeat fitting by increasing the number of arc segment.
    constrained_fit("none");
    
    /*
    Currently Testing Constrained Fitting. will be changed to full loops
    
    */
    // Need to check if cost func is called properly.

    int i=0;
    Eigen::Vector3d e1(1,0,0);
    Eigen::Vector3d e2(0,1,0);
    Eigen::Vector3d m = params.ms[i];
    Eigen::Matrix3d U = params.Us[i];
    double h = params.hs[i]; double k = params.ks[i];
    double r = sqrt(h*h*h*h + k*k);
    double w = k/r;
    
    Eigen::Vector3d A = m + h*h * U * e1;
    Eigen::Vector3d B = m - h*h * U * e1;
    Eigen::Vector3d C = m + h*h*h*h/k * U * e2;
    Eigen::Vector3d Xc = m - k * U * e2;
    double u = 0.5;
    Eigen::Vector3d midNode = (pow(1-u,2) * A + 2*u*(1-u)*w * C + pow(u,2) * B) / (pow(1-u,2) + 2*u*(1-u)*w + pow(u,2));

    parameters params_; params_.intvs = params.intvs;
    Eigen::Vector3d m1 = 0.5 * (A + midNode);
    Eigen::Vector3d m2 = 0.5 * (midNode + B);

    params_.ms.push_back(m1);
    params_.ms.push_back(m2);
    
    double hsq1 = 0.5 * (A - midNode).norm();
    double hsq2 = 0.5 * (midNode - B).norm();

    params_.hs.push_back(sqrt(hsq1));
    params_.hs.push_back(sqrt(hsq2));

    Eigen::Vector3d e1_1 = (A - params_.ms[0])/hsq1;
    Eigen::Vector3d e2_1 = (params_.ms[0] - Xc).normalized();
    Eigen::Vector3d e3_1 = e1_1.cross(e2_1);
    Eigen::Matrix3d U1;
    U1 << e1_1, e2_1, e3_1;

    Eigen::Vector3d e1_2 = (midNode - params_.ms[1])/hsq1;
    Eigen::Vector3d e2_2 = (params_.ms[1] - Xc).normalized();
    Eigen::Vector3d e3_2 = e1_2.cross(e2_2);
    Eigen::Matrix3d U2;
    U2 << e1_2, e2_2, e3_2;

    params_.Us.push_back(U1);
    params_.Us.push_back(U2);

    double k1 = (params_.ms[0] - Xc).norm();
    double k2 = (params_.ms[1] - Xc).norm();
    params_.ks.push_back(k1);
    params_.ks.push_back(k2);

    params_.slack.push_back(1e-2);
    params_.slack.push_back(1e-2);

    params = params_;
    std::pair<int,int> intv1(pts.size()/2,pts.size()-1);
    params.intvs.front().second = pts.size()/2;
    params.intvs.push_back(intv1);

    n = static_cast<int>(params.ms.size());
    mu = 1.0;
    z = Eigen::VectorXd::Zero(3*(n-1) + 3*(n-1) + n);
    constrained_fit("none");
    visualize();
}

void MultipleArcFit::visualize(void)
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

    int n = params.ms.size();
    std::vector<double> u = math_tools::linspace(0.0, 1.0, 1e3);
    Eigen::Vector3d e1(1,0,0);
    Eigen::Vector3d e2(0,1,0);
    std::vector<double> arcNodesX;
    std::vector<double> arcNodesY;
    std::vector<double> arcNodesZ;

    std::vector<std::vector<double>> approx_x;
    std::vector<std::vector<double>> approx_y;
    std::vector<std::vector<double>> approx_z;
    approx_x.reserve(n);
    approx_y.reserve(n);
    approx_z.reserve(n);

    for (int i=0;i<n;i++)
    {
        Eigen::Vector3d m = params.ms[i];
        Eigen::Matrix3d U = params.Us[i];
        double h = params.hs[i]; double k = params.ks[i];
        double r = sqrt(h*h*h*h + k*k);
        double w = k/r;
        
        Eigen::Vector3d A = m + h*h * U * e1;
        Eigen::Vector3d B = m - h*h * U * e1;
        Eigen::Vector3d C = m + h*h*h*h/k * U * e2;
        arcNodesX.push_back(A.x()); arcNodesX.push_back(B.x());
        arcNodesY.push_back(A.y()); arcNodesY.push_back(B.y());
        arcNodesZ.push_back(A.z()); arcNodesZ.push_back(B.z());

        std::vector<double> approx_x_;
        std::vector<double> approx_y_;
        std::vector<double> approx_z_;
        
        Eigen::Vector3d approx;

        for (int j=0;j<u.size();j++)
        {
            approx = (pow(1-u[j],2) * A + 2*u[j]*(1-u[j])*w * C + pow(u[j],2) * B) / (pow(1-u[j],2) + 2*u[j]*(1-u[j])*w + pow(u[j],2));
            approx_x_.push_back(approx.x());
            approx_y_.push_back(approx.y());
            approx_z_.push_back(approx.z());
        }
        approx_x.push_back(approx_x_);
        approx_y.push_back(approx_y_);
        approx_z.push_back(approx_z_);        
    }

    for (int i=0;i<approx_x.size();i++)
        plot3(approx_x[i],approx_y[i],approx_z[i],"-.r");
    
    plot3(arcNodesX,arcNodesY,arcNodesZ,"ob");
    hold(off);
    axis(equal);
    xlabel("X"); ylabel("Y"); zlabel("Z");
    show();
    
}

MultipleArcFit::parameters MultipleArcFit::getParams(void)
{
    return params;
}

int MultipleArcFit::getParamNum(void)
{
    int n = params.ms.size();
    // 8*n : Arc segment parameters
    // n : Minimum arc length parameter slack variables.
    return 8*n + n;
}

// Obtain the arc nodes and the middle node
std::vector<Eigen::Vector3d> MultipleArcFit::getNodes(void)
{
    std::vector<Eigen::Vector3d> ms = params.ms;
    std::vector<Eigen::Matrix3d> Us = params.Us;
    std::vector<double> hs = params.hs;
    std::vector<double> ks = params.ks;

    std::vector<Eigen::Vector3d> nodes;

    Eigen::Vector3d e1(1,0,0);
    Eigen::Vector3d e2(0,1,0);

    for (int i=0;i<ms.size();i++)
    {
        Eigen::Vector3d m = ms[i];
        Eigen::Matrix3d U = Us[i];
        double h = hs[i]; double k = ks[i];
        Eigen::Vector3d A = m + h*h * U * e1;
        Eigen::Vector3d B = m - h*h * U * e1;
        Eigen::Vector3d C = m + h*h*h*h/k * U * e2;

        double r = sqrt(h*h*h*h + k*k);
        double w = k/r;

        Eigen::Vector3d midNode = (0.25*A + 0.5*w*C + 0.25*B)/(0.5+0.5*w);

        if (i == 0)
            nodes.push_back(A);
        
        nodes.push_back(midNode);
        nodes.push_back(B);
    }
    return nodes;
}

// Perform Constrained Optimization for a fixed number of arc segments
void MultipleArcFit::constrained_fit(const std::string& verbose)
{
    int n = static_cast<int>(params.ms.size());
    std::cout << "Number of Arc Segments: " << n << std::endl;
    Eigen::VectorXd g(6*(n-1)+n);
    getConstraintVec(g);
    double prev_g_norm = g.norm();
    double curr_g_norm = prev_g_norm;

    // visualize();
    std::cout << "Constraint Norm: " << curr_g_norm << std::endl;
    
    while (curr_g_norm > 1e-2)
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
        

        getConstraintVec(g);
        curr_g_norm = g.norm();
        std::cout << "Constraint Norm: " << curr_g_norm << "  Mu: " << mu << std::endl;
        // std::cout << g << std::endl;

        z += 2*mu*g;
        if (curr_g_norm > 0.5 * prev_g_norm)
            mu *= 5;
        prev_g_norm = curr_g_norm;
    }
} 

std::pair<Eigen::VectorXd, Eigen::SparseMatrix<double>> MultipleArcFit::cost_func(Eigen::VectorXd dx)
{
    retract(dx);

    // Perform data association between data points and arc segments.
    associate();

    int n = static_cast<int>(params.ms.size());

    // Anchor Model
    // 1. Test with to middle anchors.
    // 2. Middle anchors with loose covariance
    std::vector<int> AnchorIdxs = {0, static_cast<int>(pts.size())-1};
    std::vector<int> ArcSegNums = {0, n-1};
    std::vector<std::string> types = {"front", "back"};

    int row_offset = 0;
    unconstrained::AnchorModel ac_model(pts);

    std::pair<Eigen::VectorXd, std::vector<Eigen::Triplet<double>>> anchor_ = ac_model.compute(AnchorIdxs,ArcSegNums,types,
                                                                                               params.Us,params.ms,params.hs,
                                                                                               3*2,row_offset);
    
    // std::cout << "Anchor Model" << std::endl;
    // std::cout << anchor_.first << std::endl << std::endl;
    // 3 * (n+1) for option 2
    std::vector<Eigen::Triplet<double>> triplets = anchor_.second;
    row_offset += 3*2;

    // Measurement Model
    // std::cout << "intvervals: " << std::endl;
    // for (std::pair<int,int> intv : params.intvs)
    //     std::cout << intv.first << ", " << intv.second << std::endl;
    // std::cout << std::endl;
    unconstrained::MeasurementModel me_model(pts, covs);
    std::pair<Eigen::VectorXd, std::vector<Eigen::Triplet<double>>> measurement_ = me_model.compute(params.intvs,params.Us,params.ms,params.hs,params.ks,
                                                                                                    3*pts.size(),row_offset);
    triplets.insert(triplets.end(), measurement_.second.begin(), measurement_.second.end());
    row_offset += 3*pts.size();

    // std::cout << "Measurement Model" << std::endl;
    // std::cout << measurement_.first << std::endl << std::endl;
    // Constraint Model 1: G0 Continuity
    constrained::G0ContinuityModel g0_model;
    std::pair<Eigen::VectorXd, std::vector<Eigen::Triplet<double>>> g0_ = g0_model.compute(params.Us,params.ms,params.hs,
                                                                                           3*(n-1),row_offset,mu);

    triplets.insert(triplets.end(), g0_.second.begin(), g0_.second.end());
    row_offset += 3 * (n-1);
    
    // std::cout << "G0 Model" << std::endl;
    // std::cout << g0_.first << std::endl << std::endl;

    // Constraint Model 2: G1 Continuity
    constrained::G1ContinuityModel g1_model;
    std::pair<Eigen::VectorXd, std::vector<Eigen::Triplet<double>>> g1_ = g1_model.compute(params.Us,params.ms,params.hs,params.ks,
                                                                                           3*(n-1),row_offset,mu);

    triplets.insert(triplets.end(), g1_.second.begin(), g1_.second.end());
    row_offset += 3 * (n-1);

    // std::cout << "G1 Model" << std::endl;
    // std::cout << g1_.first << std::endl << std::endl;

    // Constraint Model 3: Minimum Arc Length
    constrained::MinimumArcLengthModel mal_model;
    std::pair<Eigen::VectorXd, std::vector<Eigen::Triplet<double>>> mal_ = mal_model.compute(params.hs,params.ks,params.slack,
                                                                                             min_arc_length,n,row_offset,mu);

    triplets.insert(triplets.end(), mal_.second.begin(), mal_.second.end());
    row_offset += n;

    // std::cout << "MAL Model" << std::endl;
    // std::cout << mal_.first << std::endl << std::endl;

    /*
    Augment cost function residual and jacobian
    
    */
    Eigen::VectorXd unconstrained_cost_res(anchor_.first.size() + measurement_.first.size());
    unconstrained_cost_res << anchor_.first, measurement_.first;

    Eigen::VectorXd constrained_cost_res(g0_.first.size() + g1_.first.size() + mal_.first.size());
    constrained_cost_res << g0_.first, g1_.first, mal_.first;
    constrained_cost_res += z/(2*sqrt(mu)); // This vector addition should be valid (in terms of size)

    Eigen::VectorXd cost_res(unconstrained_cost_res.size() + constrained_cost_res.size());
    cost_res << unconstrained_cost_res, constrained_cost_res;

    Eigen::SparseMatrix<double> cost_jac(cost_res.size(),getParamNum());
    cost_jac.setFromTriplets(triplets.begin(), triplets.end());
    std::pair<Eigen::VectorXd, Eigen::SparseMatrix<double>> args(cost_res, cost_jac);

    return args;
}

void MultipleArcFit::retract(Eigen::VectorXd dx)
{
    int n = params.ms.size();
    for (int i=0;i<n;i++)
    {
        Eigen::Vector3d m_delta = dx.segment(8*i,3);
        Eigen::Vector3d U_delta = dx.segment(8*i+3,3);
        double h_delta = dx[8*i+6];
        double k_delta = dx[8*i+7];
        double slack_delta = dx[8*n+i];

        params.ms[i] += m_delta;
        params.Us[i] *= math_tools::Exp_map(U_delta);
        params.hs[i] += h_delta;
        params.ks[i] += k_delta;
        params.slack[i] += slack_delta;
    }

    // Eigen::Vector3d z_delta = dx.segment(8*n,6*(n-1));
    // z += z_delta;

    // std::cout << "Parameters: " << std::endl;
    // std::cout << "m: " << std::endl;
    // for (Eigen::Vector3d m : params.ms)
    //     std::cout << m << std::endl << std::endl;

    // std::cout << "U: " << std::endl;
    // for (Eigen::Matrix3d U : params.Us)
    //     std::cout << U << std::endl << std::endl;

    // std::cout << "h: " << std::endl;
    // for (double h : params.hs)
    //     std::cout << h << std::endl << std::endl;

    // std::cout << "k: " << std::endl;
    // for (double k : params.ks)
    //     std::cout << k << std::endl << std::endl;

    // std::cout << "slack: " << std::endl;
    // for (double s : params.slack)
    //     std::cout << s << std::endl << std::endl;
}

void MultipleArcFit::associate(void)
{
    // Association is performed each iteration of the cost function. 
    // * Note that the number of arc segments is fixed during the association.
    // This may not work really well, but just as a trial.
    // [Current Method]: Locate the closest point to an arc node and set it as the anchor point for that specific iteration.
    // [Possible Alternatives]: Skip association for in-middle arc segments? --> may lead to singularity if arc collapses

    Eigen::Vector3d e1(1,0,0);
    int sample_lb = 0;
    for (int i=0;i<params.ms.size()-1;i++)
    {
        // Compute the second anchor for each arc segment.
        Eigen::Vector3d m = params.ms[i]; Eigen::Matrix3d U = params.Us[i];
        double h = params.hs[i]; double k = params.ks[i];
        Eigen::Vector3d anchor = m - h * h * U * e1;
        
        // Sample points based on the previous match to the closest arc node.
        std::vector<Eigen::Vector3d> sampled_pts(pts.begin()+sample_lb,pts.end());
        std::vector<double> dist(sampled_pts.size());
        
        for (int j=0;j<sampled_pts.size();j++)
            dist[j] = (anchor - sampled_pts[j]).norm();
        
        auto min_it = std::min_element(dist.begin(),dist.end());
        int idx = std::distance(dist.begin(),min_it) + sample_lb; // Choosing the closest point as anchor index. 

        // Update the anchor indices
        params.intvs[i].second = idx; 
        params.intvs[i+1].first = idx;
        sample_lb = idx;
        // std::cout << sample_lb << std::endl;
    }

    // std::cout << "Intvs: " << std::endl;
    // for (std::pair<int,int> intv : params.intvs)
    //     std::cout << intv.first << ", " << intv.second << std::endl;
    // std::cout << std::endl;
}

// Compute the constraint residual(without multiplier parameter) using the current parameters
void MultipleArcFit::getConstraintVec(Eigen::VectorXd& constraint_res)
{
    // Size of constraint_res is already defined before passed.
    constrained::G0ContinuityModel g0_model;
    constrained::G1ContinuityModel g1_model;
    constrained::MinimumArcLengthModel mal_model;

    int n = static_cast<int>(params.ms.size());
    Eigen::Vector3d res_;
    Eigen::Matrix<double, 6, 1> res_g1;
    double f; double g1; double g2;
    double res__;
    
    for (int i=0;i<n-1;i++)
    {
        res_ = g0_model.getResiduals(params.Us[i],params.Us[i+1],params.ms[i],params.ms[i+1],params.hs[i],params.hs[i+1]);
        constraint_res.segment<3>(3*i) = res_;

        // res_g1 = g1_model.getResiduals2(params.Us[i],params.Us[i+1],params.ms[i],params.ms[i+1],params.hs[i],params.hs[i+1],params.ks[i],params.ks[i+1]);
        // constraint_res.segment<6>(3*(n-1)+6*i) = res_g1;
        g1_model.getResiduals(res_,f,g1,g2,params.Us[i],params.Us[i+1],params.ms[i],params.ms[i+1],params.hs[i],params.hs[i+1],params.ks[i],params.ks[i+1]);
        constraint_res.segment<3>(3*(n-1)+3*i) = res_;

        mal_model.getResiduals(res__,params.hs[i],params.ks[i],params.slack[i],min_arc_length);
        constraint_res[6*(n-1)+i] = res__;
    }
    mal_model.getResiduals(res__,params.hs[n-1],params.ks[n-1],params.slack[n-1],min_arc_length);
    constraint_res[6*(n-1)+n-1] = res__;
}

std::vector<int> MultipleArcFit::checkApproxValidity(std::vector<std::vector<int>> intvs, std::vector<Eigen::Vector3d> ms_, std::vector<Eigen::Matrix3d> Us_, 
                                                     std::vector<double> hs_, std::vector<double> ks_, const std::string verbose)
{
    std::vector<int> invalidN(intvs.size(),0);

    Eigen::Vector3d e1(1,0,0);
    Eigen::Vector3d e2(0,1,0);
    Eigen::Matrix3d S;
    S << 1, 0, 0,
         0, 1, 0,
         0, 0, 0;

    for (int i=0;i<intvs.size();i++)
    {
        int lb = intvs[i].front(); int ub = intvs[i].back();
        Eigen::Vector3d m = ms_[i];
        Eigen::Matrix3d U = Us_[i];
        double h = hs_[i]; double k = ks_[i];

        double r = sqrt(k*k + h*h*h*h);
        Eigen::Vector3d Xc = m - k * U * e2;

        for (int j=lb;j<=ub;j++)
        {   
            Eigen::VectorXd cv = covs[j];
            Eigen::Matrix3d cov;
            cov << cv[0], cv[1], cv[2],
                   cv[3], cv[4], cv[5],
                   cv[6], cv[7], cv[8];

            Eigen::Vector3d Pj_rel = S * U.transpose() * (pts[j] - Xc);
            double den = Pj_rel.norm();
            Eigen::Vector3d p_t = Xc + r * U * Pj_rel / den;

            if (!math_tools::chi_squared_test(p_t, pts[j], cov, 3.0, 0.99))
                invalidN[i] += 1;
        }
    }

    if (verbose == "full")
    {
        std::cout << "[Number of invalid point approximation for each arc segment]" << std::endl;
        std::cout << "Arc segments No. ";
        for (int i=0;i<invalidN.size();i++)
            std::cout << i << " ";
        std::cout << std::endl;

        std::cout << "Invalid Approx # ";
        for (int num : invalidN)
            std::cout << num << " ";
        std::cout << std::endl;
    }

    return invalidN;
}

std::vector<int> MultipleArcFit::getLinearIntvs(std::pair<int,int> intv, int depth)
{
    /*
    (1) Connect first and end intervals with a line.
    (2) Find the data point that is furthest from the line in (1).
    (3) Divide the input data points into two, w.r.t the point found in (2).
    (4) Repeat (1)~(3) for every divided segments.

    If maximum error computed at (2) is lower than a certain threshold,
    stop dividing the current segment further and perform backward propagation.

    Originally implemented in MATLAB, 2022 for Master's Thesis, and transfered to C++ in 2024 April.
    */

    double line_acc_thres = 0.2;
    
    Eigen::Vector3d init_point = pts[intv.first];
    Eigen::Vector3d last_point = pts[intv.second];
    bgLineString line;
    bg::append(line, bgPoint(init_point.x(),init_point.y(),init_point.z()));
    bg::append(line, bgPoint(last_point.x(),last_point.y(),last_point.z()));

    std::vector<double> dist;
    for (int i=intv.first;i<=intv.second;i++)
    {
        Eigen::Vector3d target_point = pts[i];
        bgPoint point(target_point.x(),target_point.y(),target_point.z());
        dist.push_back(bg::distance(point,line));
    }

    auto max_it = std::max_element(dist.begin(), dist.end());
    int max_idx = std::distance(dist.begin(), max_it);
    // std::cout << "Maximum error occurred at index: " << max_idx << std::endl;

    if (dist[max_idx] < line_acc_thres)
    {
        std::vector<int> ret;
        ret.push_back(intv.first); ret.push_back(intv.second);
        return ret;
    }
    else
    {
        std::pair<int,int> intv1_(intv.first,intv.first + max_idx);
        std::pair<int,int> intv2_(intv.first + max_idx,intv.second);
        std::vector<int> intv1 = getLinearIntvs(intv1_,depth+1);
        std::vector<int> intv2 = getLinearIntvs(intv2_,depth+1);

        for (int el : intv2)
            intv1.push_back(el);
        return intv1;
    }

}

}


}

