
#ifndef _SteepDescentALLS_H_
#define __SteepDescentALLS_H_

#include "ComputeSD.h"

//void SD_grad_LS(GradientOptimizerContext &, double, double, double *, double *);
//bool FitCompare(GradientOptimizerContext &, double, double, double *, double *);
//void steepDES_LS(GradientOptimizerContext &rf, int maxIter, double rho, double *lambda, double *mu);
//void auglagSD_LS(GradientOptimizerContext &, double, double *, double *);
void auglag_minimize_SD_LS(GradientOptimizerContext &);

#endif
