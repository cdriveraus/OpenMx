/* Steepest descent optimizer test */

#include "ComputeSD.h"

void SD_grad(GradientOptimizerContext &rf)
{
    ComputeFit("steep_fd", rf.fitMatrix, FF_COMPUTE_FIT, rf.fc);

    const double refFit = rf.fc->fit;
    const double eps = 1e-5;
    Eigen::VectorXd p1(rf.fc->numParam), p2(rf.fc->numParam), grad(rf.fc->numParam);

    memcpy(p1.data(), rf.fc->est, (rf.fc->numParam) * sizeof(double));

    for (int px = 0; px < rf.fc->numParam; px++) {
        memcpy(p2.data(), rf.fc->est, (rf.fc->numParam) * sizeof(double));
        p2[px] += eps;
        memcpy(rf.fc->est, p2.data(), (rf.fc->numParam) * sizeof(double));
        rf.fc->copyParamToModel();
        ComputeFit("steep_fd", rf.fitMatrix, FF_COMPUTE_FIT, rf.fc);
        grad[px] = (rf.fc->fit - refFit) / eps;
        memcpy(rf.fc->est, p1.data(), (rf.fc->numParam) * sizeof(double));
        rf.fc->copyParamToModel();
    }
    rf.fc->grad = grad;
}

bool FitCompare(GradientOptimizerContext &rf, double speed)
{
    Eigen::VectorXd prevEst(rf.fc->numParam);
    double refFit, newFit;

    memcpy(prevEst.data(), rf.fc->est, (rf.fc->numParam) * sizeof(double));
    ComputeFit("steep", rf.fitMatrix, FF_COMPUTE_FIT, rf.fc);
    refFit = rf.fc->fit;

    SD_grad(rf);
    Eigen::VectorXd searchDir = rf.fc->grad;
    Eigen::Map< Eigen::VectorXd > currEst(rf.fc->est, rf.fc->numParam);
    currEst = prevEst - speed / searchDir.norm() * searchDir;
    rf.fc->copyParamToModel();
    ComputeFit("steep", rf.fitMatrix, FF_COMPUTE_FIT, rf.fc);
    newFit = rf.fc->fit;

    if(newFit < refFit) return newFit < refFit;
    currEst = prevEst;
    rf.fc->copyParamToModel();
    return newFit < refFit;
}



void steepDES(GradientOptimizerContext &rf, int maxIter)
{
	int iter = 0;
	double priorSpeed = 1.0;

	while(iter < maxIter)
	{
        bool findit;
        findit = FitCompare(rf, priorSpeed);
        if (findit)
        {
            priorSpeed *=1.1;
            findit = FitCompare(rf, priorSpeed);
        }

        int retries = 15;
        double speed = priorSpeed;
        while (--retries > 0 && !findit){
            speed *= 0.2;
            findit = FitCompare(rf, speed);
        }
        if(findit){
            iter++;
            SD_grad(rf);
        }
        else{
            mxLog("after %i iterations, cannot find better estimation along the gradient direction", iter);
            return;
        }
    }
    return;
}
