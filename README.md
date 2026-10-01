# Monte Carlo Pricing of Asian Options Under the CEV Model

**Variance Reduction and Sensitivity Analysis**

This repository contains the code and final seminar paper for a project completed in the MSc Economics programme at the **University of Copenhagen** as part of the course *Monte Carlo Methods in Finance and Econometrics*.

The project studies the pricing of an **arithmetic Asian call option** under the **Constant Elasticity of Variance (CEV) model**. Since the option does not have a closed-form pricing solution under the CEV model, we use simulation-based methods and compare their accuracy and computational efficiency.

The main comparison is between **Crude Monte Carlo (CMC)** and **Randomized Quasi-Monte Carlo (RQMC)** using scrambled Sobol' sequences. We also investigate Brownian bridge construction, control variates, antithetic variates, Euler-Maruyama and Milstein discretization, and finite-difference estimation of Delta and Vega.

**Grade:** 10/12 on the Danish 7-point grading scale.

## Main Findings

- **RQMC combined with Brownian bridge construction substantially improves precision** relative to standard Monte Carlo and plain RQMC in the numerical experiments.
- **Milstein is more accurate than Euler-Maruyama** for simulating the CEV process and is therefore used in the main analysis.
- **Control variates improve CMC**, but provide smaller gains when added to RQMC.
- **Antithetic variates are useful for CMC**, but generally provide little or no benefit when combined with RQMC and Brownian bridge construction.
- Estimating **Delta and Vega** is more noise-sensitive than estimating the option price. RQMC nevertheless provides substantial variance reductions for the Greek estimates.

Overall, the results show that RQMC with Brownian bridge construction is particularly effective for pricing path-dependent options under the CEV model, while the usefulness of additional variance-reduction techniques depends on the simulation method and computational budget.

## Repository Structure

| File | Description |
|---|---|
| `lib.py` | Core implementation of the CEV simulation, Monte Carlo methods, variance-reduction techniques and Greek estimators. |
| `Base.ipynb` | Baseline implementation and initial simulation experiments. |
| `Eul_Mil_comp.ipynb` | Comparison of Euler-Maruyama and Milstein discretization. |
| `RQMC_BB.ipynb` | Analysis of the effect of Brownian bridge construction on RQMC. |
| `CMC_RQMC_comp.ipynb` | Comparison of CMC and RQMC across different sample sizes. |
| `Variance_reduction.ipynb` | Experiments with control variates and antithetic variates. |
| `Comp.ipynb` | Combined comparison of the pricing methods in terms of accuracy and runtime. |
| `Sensitivity.ipynb` | Delta and Vega estimation and finite-difference sensitivity analysis. |
| `Seminarpaper_MC_Final.pdf` | Final seminar paper containing the theory, methodology, numerical results and discussion. |

## Paper

The full seminar paper is available in [`Seminarpaper_MC_Final.pdf`](Seminarpaper_MC_Final.pdf).

**Authors:** Mikkel Foss Engelsted, Mikkel Rath Tornerup and Nicklas Busk Jensen.
