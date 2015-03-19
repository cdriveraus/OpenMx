#include "ComputeSD_AL_LS.h"


template <typename T1, typename T2>
static void auglagSD_LS(GradientOptimizerContext &rf, double rho,
			Eigen::MatrixBase<T1> &lambda, Eigen::MatrixBase<T2> &mu)
{
    rf.fc->copyParamToModel();
    ComputeFit("auglag", rf.fitMatrix, FF_COMPUTE_FIT, rf.fc);

    rf.solEqBFun();
    rf.myineqFun();

    for (size_t i = 0; i < rf.equality.size(); ++i)
    {
        rf.fc->fit += 0.5 * rho * (rf.equality[i] + lambda[i] / rho) * (rf.equality[i] + lambda[i] / rho);
    }

    for (size_t i = 0; i < rf.inequality.size(); ++i)
    {
        rf.fc->fit += 0.5 * rho * std::max(0.0,(rf.inequality[i] + mu[i] / rho)) * std::max(0.0,(rf.inequality[i] + mu[i] / rho));
    }
    return;
}

template <typename T1, typename T2>
static void SD_grad_LS(GradientOptimizerContext &rf, double eps, double rho,
		       Eigen::MatrixBase<T1> &lambda, Eigen::MatrixBase<T2> &mu)
{
    auglagSD_LS(rf, rho, lambda, mu);
    const double refFit = rf.fc->fit;

    Eigen::VectorXd p1(rf.fc->numParam), p2(rf.fc->numParam), grad(rf.fc->numParam);

    memcpy(p1.data(), rf.fc->est, (rf.fc->numParam) * sizeof(double));

    for (int px = 0; px < int(rf.fc->numParam); px++) {
        memcpy(p2.data(), rf.fc->est, (rf.fc->numParam) * sizeof(double));
        p2[px] += eps;
        memcpy(rf.fc->est, p2.data(), (rf.fc->numParam) * sizeof(double));

        auglagSD_LS(rf, rho, lambda, mu);
        grad[px] = (rf.fc->fit - refFit) / eps;

        memcpy(rf.fc->est, p1.data(), (rf.fc->numParam) * sizeof(double));
        rf.fc->copyParamToModel();
    }
    rf.fc->fit = refFit;
    rf.fc->grad = grad;
}

//Armijo's Rule
template <typename T1, typename T2>
static void steepDES_LS(GradientOptimizerContext &rf, int maxIter, double rho,
			Eigen::MatrixBase<T1> &lambda, Eigen::MatrixBase<T2> &mu, double &alpha)
{
	int iter = 0;
	double epsilon = 0.2, eta = 2;
    rf.setupSimpleBounds();
    rf.informOut = INFORM_UNINITIALIZED;
    rf.fc->copyParamToModel();

    SD_grad_LS(rf, 1e-12, rho, lambda, mu);

	while(iter < maxIter && !isErrorRaised())
	{
        alpha = std::min(1.0, alpha);
        iter++;
        SD_grad_LS(rf, 1e-12, rho, lambda, mu);
        Eigen::VectorXd searchDir = rf.fc->grad;   // memory issue bug???
        if (isnan(rf.fc->fit))
        {
            rf.informOut = INFORM_STARTING_VALUES_INFEASIBLE;
            break;
        }

        if(rf.fc->grad.norm() < 1e-4)
        {
            rf.informOut = INFORM_CONVERGED_OPTIMUM;
            mxLog("after %i iterations, gradient achieves zero!", iter);
            mxLog("alpha: %f", alpha);
            break;
        }

        Eigen::Map< Eigen::VectorXd > currEst(rf.fc->est, rf.fc->numParam);
        Eigen::VectorXd prevEst = currEst;


        currEst = prevEst - alpha * searchDir / searchDir.norm();
        // ensure the estimation is in the box
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

        auglagSD_LS(rf, rho, lambda, mu);
        double phi = rf.fc->fit;

        currEst = prevEst;
        auglagSD_LS(rf, rho, lambda, mu);
        double dashline = rf.fc->fit - epsilon * alpha * searchDir.norm();

        if(phi <= dashline)
        {
            do{
                alpha *= eta;
                currEst = prevEst;
                auglagSD_LS(rf, rho, lambda, mu);
                dashline = rf.fc->fit - epsilon * alpha * searchDir.norm();

                currEst = prevEst - alpha * searchDir / searchDir.norm();
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

                auglagSD_LS(rf, rho, lambda, mu);
                phi = rf.fc->fit;
            }while(phi <= dashline);
            currEst = prevEst - alpha / eta * searchDir / searchDir.norm();
            currEst = currEst.cwiseMax(rf.solLB).cwiseMin(rf.solUB);
            auglagSD_LS(rf, rho, lambda, mu);
        }
        else
        {
            do{
                alpha /= eta;
                currEst = prevEst;
                auglagSD_LS(rf, rho, lambda, mu);
                dashline = rf.fc->fit - epsilon * alpha * searchDir.norm();

                currEst = prevEst - alpha * searchDir / searchDir.norm();
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

                auglagSD_LS(rf, rho, lambda, mu);
                phi = rf.fc->fit;
            }while(phi > dashline);
      }


        mxLog("iteration: %i", iter);

        if (iter == maxIter - 1)
        {
            rf.informOut = INFORM_ITERATION_LIMIT;
            mxLog("Maximum iteration achieved!");
            break;
        }

    }
    mxLog("status code : %i", rf.informOut);
    return;
}


void auglag_minimize_SD_LS(GradientOptimizerContext &rf)
{
    double ICM = HUGE_VAL;

    /* magic parameters from Birgin & Martinez */
    const double tau = 0.5, gam = 10;
    const double lam_min = -1e20, lam_max = 1e20, mu_max = 1e20;
    double alpha = 1;

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

    Eigen::VectorXd lambda(eq_size);
    lambda.setZero();
    Eigen::VectorXd mu(ineq_size);
    mu.setZero();

    int iter = 0;
    Eigen::VectorXd V(ineq_size);

    do{
        iter++;
        double prev_ICM = ICM;
        ICM = 0;

        steepDES_LS(rf, 500, rho, lambda, mu, alpha);

        if(rf.informOut == INFORM_STARTING_VALUES_INFEASIBLE) return;

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

        mxLog("ICM: %f", ICM);
        if((ICM - 0)<1e-4)
        {
            mxLog("augmented lagrangian coverges!");
            return;
        }
    } while (1);

}






