#include "ComputeSD_AL.h"

/*
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define ABS(a) ((a) >= 0 ? (a) : (-a))
*/
double auglagSD(GradientOptimizerContext &rf, double rho, double *lambda, double *mu)
{
    rf.fc->copyParamToModel();
    ComputeFit("auglag", rf.fitMatrix, FF_COMPUTE_FIT, rf.fc);

    double augFit = rf.fc->fit;
    rf.solEqBFun();
    rf.myineqFun();

    for (size_t i = 0; i < rf.equality.size(); ++i)
    {
        augFit += 0.5 * rho * (rf.equality[i] + lambda[i] / rho) * (rf.equality[i] + lambda[i] / rho);
    }

    for (size_t i = 0; i < rf.inequality.size(); ++i)
    {
        augFit += 0.5 * rho * std::max(0.0,(rf.inequality[i] + mu[i] / rho)) * std::max(0.0,(rf.inequality[i] + mu[i] / rho));
    }
    return augFit;
}

void SD_grad(GradientOptimizerContext &rf, double eps, double rho, double *lambda, double *mu)
{
    const double refFit = auglagSD(rf, rho, lambda, mu);

    Eigen::VectorXd p1(rf.fc->numParam), p2(rf.fc->numParam), grad(rf.fc->numParam);

    memcpy(p1.data(), rf.fc->est, (rf.fc->numParam) * sizeof(double));

    for (int px = 0; px < int(rf.fc->numParam); px++) {
        memcpy(p2.data(), rf.fc->est, (rf.fc->numParam) * sizeof(double));
        p2[px] += eps;
        memcpy(rf.fc->est, p2.data(), (rf.fc->numParam) * sizeof(double));

        double newFit = auglagSD(rf, rho, lambda, mu);

        grad[px] = (newFit - refFit) / eps;
        memcpy(rf.fc->est, p1.data(), (rf.fc->numParam) * sizeof(double));

        rf.fc->copyParamToModel();
    }
    ComputeFit("auglag", rf.fitMatrix, FF_COMPUTE_FIT, rf.fc);
    rf.fc->grad = grad;
}


bool FitCompare(GradientOptimizerContext &rf, double speed, double rho, double *lambda, double *mu)
{
    Eigen::Map< Eigen::VectorXd > currEst(rf.fc->est, rf.fc->numParam);
    Eigen::VectorXd prevEst = currEst;

    rf.fc->fit = auglagSD(rf, rho, lambda, mu);

    if (isnan(rf.fc->fit))
    {
        rf.informOut = INFORM_STARTING_VALUES_INFEASIBLE;
        return FALSE;
    }
    double refFit = rf.fc->fit;

    Eigen::VectorXd searchDir = rf.fc->grad;

    currEst = prevEst - speed * searchDir / searchDir.norm();
    currEst = currEst.cwiseMax(rf.solLB).cwiseMin(rf.solUB);
    if(rf.verbose >= 2){
        for(int index = 0; index < int(rf.fc->numParam); index++)
        {
            if(currEst[index] == rf.solLB[index])
                mxLog("paramter %i hit lower bound %f", index, rf.solLB[index]);
            if(currEst[index] == rf.solUB[index])
                mxLog("paramter %i hit upper bound %f", index, rf.solUB[index]);
        }
    }

    rf.fc->copyParamToModel();
    double newFit = auglagSD(rf, rho, lambda, mu);

    if(newFit < refFit) return newFit < refFit;
    currEst = prevEst;
    rf.fc->copyParamToModel();
    return newFit < refFit;
}

void steepDES(GradientOptimizerContext &rf, int maxIter, double rho, double *lambda, double *mu)
{
	int iter = 0;
	double priorSpeed = 1;
    rf.setupSimpleBounds();
    rf.informOut = INFORM_UNINITIALIZED;
    rf.fc->copyParamToModel();

    rf.fc->fit = auglagSD(rf, rho, lambda, mu);

    SD_grad(rf, 1e-12, rho, lambda, mu);

	while(iter < maxIter && !isErrorRaised())
	{
        if(rf.fc->grad.norm() == 0)
        {
            rf.informOut = INFORM_CONVERGED_OPTIMUM;
            mxLog("after %i iterations, gradient achieves zero!", iter);
            break;
        }
        bool findit = FitCompare(rf, priorSpeed, rho, lambda, mu);

        int retries = 200;
        double speed = priorSpeed;
        while (--retries > 0 && !findit && !isErrorRaised()){
            speed *= 0.5;
            findit = FitCompare(rf, speed, rho, lambda, mu);
        }
        if(findit){
            //priorSpeed = speed * 1.1;
            iter++;
            if(iter == maxIter){
                rf.informOut = INFORM_ITERATION_LIMIT;
                mxLog("Maximum iteration achieved!");
                break;
            }
        }
        else{
            switch (iter)
            {
                case 0:
                    if(rf.informOut == INFORM_STARTING_VALUES_INFEASIBLE)
                    {
                        mxLog("Infeasbile starting values!");
                        break;
                    }
                    rf.informOut = INFORM_CONVERGED_OPTIMUM;
                    mxLog("after %i iterations, cannot find better estimation along the gradient direction", iter);
                    break;
                case 599999:
                    rf.informOut = INFORM_ITERATION_LIMIT;
                    mxLog("Maximum iteration achieved!");
                    break;
                default:
                    rf.informOut = INFORM_CONVERGED_OPTIMUM;
                    mxLog("after %i iterations, cannot find better estimation along the gradient direction", iter);
            }
            break;
        }
    }
    mxLog("status code : %i", rf.informOut);
    return;
}


void auglag_minimize_SD(GradientOptimizerContext &rf)
{
    double ICM = HUGE_VAL;

    /* magic parameters from Birgin & Martinez */
    const double tau = 0.5, gam = 10;
    const double lam_min = -1e20, lam_max = 1e20, mu_max = 1e20;

    rf.fc->copyParamToModel();
    ComputeFit("auglag", rf.fitMatrix, FF_COMPUTE_FIT, rf.fc);
    rf.solEqBFun();
    rf.myineqFun();

    size_t ineq_size = rf.inequality.size(), eq_size = rf.equality.size();

    // initialize penalty parameter rho and the Lagrange multipliers lambda and mu
    double eq_norm = 0, ineq_norm = 0;

    for(size_t i = 0; i < eq_size; i++)
    {
      eq_norm += rf.equality[i] * rf.equality[i];
    }

    for(size_t i = 0; i < ineq_size; i++)
    {
      ineq_norm += std::max(0.0, rf.inequality[i]) * std::max(0.0, rf.inequality[i]);
    }

    double rho = std::max(1e-6, std::min(10.0, (2 * std::abs(rf.fc->fit) / (eq_norm + ineq_norm))));
    double lambda[eq_size], mu[ineq_size];  // not double *lambda[eq_size], *mu[ineq_size]
    for(size_t i = 0; i < eq_size; i++){
        lambda[i] = 0;
    }
    for(size_t i = 0; i < ineq_size; i++){
        mu[i] = 0;
    }


    int iter = 0;
    Eigen::VectorXd V(ineq_size);

    do{
        iter++;
        double prev_ICM = ICM;
        ICM = 0;

        steepDES(rf, 600000, rho, lambda, mu);

//        if(rf.informOut == INFORM_STARTING_VALUES_INFEASIBLE) return;

        rf.fc->copyParamToModel();
        rf.solEqBFun();
        rf.myineqFun();

        for(size_t i = 0; i < eq_size; i++){
            lambda[i] = std::min(std::max(lam_min, (lambda[i] + rho * rf.equality[i])), lam_max);
            ICM = std::max(ICM, std::abs(rf.equality[i]));
        }

        for(size_t i = 0; i < ineq_size; i++){
            mu[i] = std::min(std::max(0.0, (mu[i] + rho * rf.inequality[i])),mu_max);
        }

        for(size_t i = 0; i < ineq_size; i++){
            V[i] = std::max(rf.inequality[i], (-mu[i] / rho));
            ICM = std::max(ICM, std::abs(V[i]));
        }

        if(!(iter == 1 || ICM <= tau * prev_ICM))
        {
            rho *= gam;
        }
        mxLog("rho: %f", rho);
        mxLog("original fit: %f", rf.fc->fit);
        mxLog("aug fit: %f", auglagSD(rf, rho, lambda, mu));
        mxLog("ICM: %f", ICM);
        if((ICM - 0)<1e-8)
        {
            mxLog("augmented lagrangian coverges!");
            return;
        }
    } while (1);

}















