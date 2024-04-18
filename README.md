# ArcFit 

Multiple arc spline approximation for data points with covariances. 

Approximation is done based on solving Constrained Nonlinear Least Squares via Augmented Lagrangian (which is in fact not very efficient)

Needs more development for general options in MultipleArcFit class, but test works quite well on simple cases.

Currently stopped development due to the limits of optimizer. Will be using MATLAB's lsqnonlin function instead.

If you want to use this for your research, cite 

```
@misc{jeon2024reliabilitybased,
      title={Reliability-based G1 Continuous Arc Spline Approximation}, 
      author={Jinhwan Jeon and Yoonjin Hwang and Seibum B. Choi},
      year={2024},
      eprint={2401.09770},
      archivePrefix={arXiv},
      primaryClass={cs.CG}
}
```

# Installation
```
1. Install dependencies: Eigen, Matplotplusplus (for visualization), Boost

2. Clone this repository.

3. Create build directory and 'cd' inside.

4. Build (Run cmake .. and make)
```

Now you are ready!

To run, take a look at the 'test' folder for usage. 
