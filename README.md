# Bayesian-group-shrinkage-spatial-autoregressive-models
This repocitory contains implementation of the estimation algorithms proposed in the paper "Bayesian group shrinkage for spatial autoregressive model with convex combination of spatial weights matrices" with replication of main simulation result.

## Repository structure
```
├── src/
│   ├── libs/           # Helper and utility functions
│   ├── mcmc/           # Core algorithm implementation
│   └── simulations/    # Simulation scripts
```

### Summary
#### `libs/`
Utility and helper functions that support the core algorithm and simulation
files. These are not part of the algorithm itself but are required for running the algorithm and simulation.

#### `mcmc/`
Core implementation of the Bayesian group shrinkage algorithms for the spatial autoregressive model with convex combination of spatial weights matrices. Each file corresponds to a distinct prior imposed on the convex coefficients.

| File | Prior | 
|---|---|
| `mcmc_bagl_sa.m`[^1] | Bayesian adaptive group lasso | 
| `mcmc_bal.m` | Bayesian adaptive lasso | 
| `mcmc_basgl_sa_b.m`[^2] |  Bayesian adaptive sparse group lasso | 
|`mcmc_gigg_sa_b.m` [^2]| Group inverse Gamma-Gamma prior |
|`mcmc_gigg.m`|Group inverse Gamma-Gamma prior|
|`mcmc_up.m`|Uniform prior|

[^1]: Stochastic approximation for hyperparameter tuning
[^2]: Stochastic approximation for hyperparameter tuning on group-wise tuning parameters

#### `simulations/`
Scripts for producing the main experiment and table reported in the paper.
Each file is self-contained and can be run independently given that `libs` and `mcmc` directories are included the path.

## Acknowledgements
`libs` directory contains some third-party functions provided by other scholars/programmers. Detailed attribution for these functions can be assessed by directly viewing the respective source files. Here we provide a brief summary.

### Third-party functions
| Function | Original Author | Source |
|---|---|---|
| `gigrnd` | Jan Patrick Hartkopf | [Link](https://www.mathworks.com/matlabcentral/fileexchange/78805-gigrnd)| 
| `make_neighborsw` | James P. LeSage | [Link](https://www.spatial-econometrics.com) | 

This work builds upon these third-party functions. We thank the authors for making their work publicly available.

## License
All source code in this repository, except for those mentioned in the third-party functions, is freely available for use, modification, and distribution without restriction. No reference is required to use the work in this repository.
