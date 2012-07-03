/*
 *  Copyright 2007-2012 The OpenMx Project
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include "omxObjective.h"
#include "omxBLAS.h"
#include "omxFIMLObjective.h"
#include "omxMLObjective.h"
#include "omxRAMObjective.h"

void calculateRAMGradientComponents(omxObjective* oo, omxMatrix**, omxMatrix**, int*);
void sliceCrossUpdate(omxMatrix* A, omxMatrix* B, int row, int col, omxMatrix* result);
void fastRAMGradientML(omxObjective* oo, double* result);

// Speedup Helper
void ADB(omxMatrix** A, omxMatrix** B, int numArgs, omxMatrix** D, int matNum, omxFreeVar* varList, 
        int* pNums, int nParam, omxMatrix*** result) {
    // Computes Matrix %*% params %*% Matrix in O(K^2) time.  Based on von Oertzen & Brick, in prep.
    // Also populates the matrices called D if it appears.
    // Minimal error checking.
    if(OMX_DEBUG_ALGEBRA) Rprintf("Beginning ADB.\n"); //:::DEBUG:::

    omxFreeVar var;
    int paramNo;
    omxMatrix *thisD;

    for(int param = 0; param < nParam; param++) {
        if(pNums != NULL) {
            paramNo = pNums[param];
        } else {
            paramNo = param;
        }
        var = varList[paramNo];

        // Set D
        if(D != NULL) {
            thisD = D[param];
            memset(thisD->data, 0, sizeof(double) * thisD->cols * thisD->rows);
            // Honestly, this should be calculated only once.
            for(int varLoc = 0; varLoc < var.numLocations; varLoc++) {
                if(~var.matrices[varLoc] == matNum) {
                    omxSetMatrixElement(thisD, var.row[varLoc], var.col[varLoc], 1.0);
                }
            }
        }
        for(int eqn = 0; eqn < numArgs; eqn++) {
            omxMatrix* thisResult = result[eqn][param];
            memset(thisResult->data, 0, sizeof(double) * thisResult->cols * thisResult->rows);
            for(int varLoc = 0; varLoc < var.numLocations; varLoc++) {
                if(~var.matrices[varLoc] == matNum) {
                    sliceCrossUpdate(A[eqn], B[eqn], var.row[varLoc], var.col[varLoc], thisResult);
                }
            }
        }
    }
}

void sliceCrossUpdate(omxMatrix* A, omxMatrix* B, int row, int col, omxMatrix* result) {
    // Performs outer multiply of column col of A by row row of B.
    // Adds product to beta * result.
    int nrow = A->rows;
    int ncol = B->cols;
    int resultrows = result->rows;
    int brows = B->rows;

    if (A->colMajor && B->colMajor && result->colMajor) {
        int aoffset = row * nrow;
        double* Adata = A->data;
        double* Bdata = B->data;
        double* resdata = result->data;

        for(int k = 0; k < ncol; k++) {
            int offset = k * resultrows;
            double factor = Bdata[k * brows + col];
            for(int j = 0; j < nrow; j++) {
                resdata[offset + j] += Adata[aoffset + j] * factor;
            }
        }
    } else {
        for(int k = 0; k < ncol; k++) {
            for(int j = 0; j < nrow; j++) {
                omxAccumulateMatrixElement(result, j, k, omxMatrixElement(A, j, row) * omxMatrixElement(B, col, k));
            }
        }
    }
}

double trAB(omxMatrix* A, omxMatrix* B) {
    double output = 0.0;
    // Computes tr(AB)
    // Note: does not check for conformability
    for(int j = 0; j < A->rows; j++) 
        for(int k = 0; k < A->cols; k++)
            output += omxMatrixElement(A, j, k) * omxMatrixElement(B, k, j);
    
    return(output);
}

void omxCallRAMObjective(omxObjective* oo) {
    if(OMX_DEBUG) { Rprintf("RAM Subobjective called.\n"); }
	omxRAMObjective* oro = (omxRAMObjective*)(oo->argStruct);
	
	omxRecompute(oro->A);
	omxRecompute(oro->S);
	omxRecompute(oro->F);
	if(oro->M != NULL)
	    omxRecompute(oro->M);
	    
	omxCalculateRAMCovarianceAndMeans(oro->A, oro->S, oro->F, oro->M, oro->cov, oro->means, oro->numIters, oro->I, oro->Z, oro->Y, oro->X, oro->Ax);
}

void omxDestroyRAMObjective(omxObjective* oo) {

	if(OMX_DEBUG) { Rprintf("Destroying RAM Objective.\n"); }
	
	omxRAMObjective* argStruct = (omxRAMObjective*)(oo->argStruct);

	/* We allocated 'em, so we destroy 'em. */
	if(argStruct->cov != NULL)
		omxFreeMatrixData(argStruct->cov);

	if(argStruct->means != NULL) {
		omxFreeMatrixData(argStruct->dM);
		omxFreeMatrixData(argStruct->ZM);
		omxFreeMatrixData(argStruct->bCB);
		omxFreeMatrixData(argStruct->b);
		omxFreeMatrixData(argStruct->tempVec);
		omxFreeMatrixData(argStruct->bigSum);
		omxFreeMatrixData(argStruct->lilSum);
		omxFreeMatrixData(argStruct->beCov);
		omxFreeMatrixData(argStruct->means);
	}

	int nParam = argStruct->nParam;
	if(nParam >= 0) {
		for(int j = 0; j < nParam; j++) {
			if(argStruct->dAdts != NULL)
				omxFreeMatrixData(argStruct->dAdts[j]);
			if(argStruct->dSdts != NULL)
				omxFreeMatrixData(argStruct->dSdts[j]);
			if(argStruct->dMdts != NULL) 
				omxFreeMatrixData(argStruct->dMdts[j]);
			if(argStruct->eqnOuts != NULL)
				omxFreeMatrixData(argStruct->eqnOuts[0][j]);
		}
		omxFreeMatrixData(argStruct->paramVec);
	}

	if (argStruct->dA != NULL)
		omxFreeMatrixData(argStruct->dA);

	if (argStruct->dS != NULL)
		omxFreeMatrixData(argStruct->dS);

	omxFreeMatrixData(argStruct->I);
	omxFreeMatrixData(argStruct->lilI);
	omxFreeMatrixData(argStruct->X);
	omxFreeMatrixData(argStruct->Y);
	omxFreeMatrixData(argStruct->Z);
	omxFreeMatrixData(argStruct->Ax);

	omxFreeMatrixData(argStruct->W);
	omxFreeMatrixData(argStruct->U);
	omxFreeMatrixData(argStruct->EF);
	omxFreeMatrixData(argStruct->V);
	omxFreeMatrixData(argStruct->ZSBC);
	omxFreeMatrixData(argStruct->C);
	omxFreeMatrixData(argStruct->eCov);


	if(argStruct->ppmlData != NULL) 
		omxFreeData(argStruct->ppmlData);
	
}

void omxPopulateRAMAttributes(omxObjective *oo, SEXP algebra) {
    if(OMX_DEBUG) { Rprintf("Populating RAM Attributes.\n"); }

	omxRAMObjective* oro = (omxRAMObjective*) (oo->argStruct);
	omxMatrix* A = oro->A;
	omxMatrix* S = oro->S;
	omxMatrix* X = oro->X;
	omxMatrix* Ax= oro->Ax;
	omxMatrix* Z = oro->Z;
	omxMatrix* I = oro->I;
    int numIters = oro->numIters;
    double oned = 1.0, zerod = 0.0;
    
    omxRecompute(A);
    omxRecompute(S);
	
	omxFastRAMInverse(numIters, A, Z, Ax, I ); // Z = (I-A)^-1
	
	if(OMX_DEBUG_ALGEBRA) { Rprintf("....DGEMM: %c %c \n %d %d %d \n %f \n %x %d %x %d \n %f %x %d.\n", *(Z->majority), *(S->majority), (Z->rows), (S->cols), (S->rows), oned, Z->data, (Z->leading), S->data, (S->leading), zerod, Ax->data, (Ax->leading));}
	// F77_CALL(omxunsafedgemm)(Z->majority, S->majority, &(Z->rows), &(S->cols), &(S->rows), &oned, Z->data, &(Z->leading), S->data, &(S->leading), &zerod, Ax->data, &(Ax->leading)); 	// X = ZS
	omxDGEMM(FALSE, FALSE, oned, Z, S, zerod, Ax);

	if(OMX_DEBUG_ALGEBRA) { Rprintf("....DGEMM: %c %c %d %d %d %f %x %d %x %d %f %x %d.\n", *(Ax->majority), *(Z->minority), (Ax->rows), (Z->rows), (Z->cols), oned, X->data, (X->leading), Z->data, (Z->lagging), zerod, Ax->data, (Ax->leading));}
	// F77_CALL(omxunsafedgemm)(Ax->majority, Z->minority, &(Ax->rows), &(Z->rows), &(Z->cols), &oned, Ax->data, &(Ax->leading), Z->data, &(Z->leading), &zerod, Ax->data, &(Ax->leading)); 
	omxDGEMM(FALSE, TRUE, oned, Ax, Z, zerod, Ax);
	// Ax = ZSZ' = Covariance matrix including latent variables
	
	SEXP expCovExt;
	PROTECT(expCovExt = allocMatrix(REALSXP, Ax->rows, Ax->cols));
	for(int row = 0; row < Ax->rows; row++)
		for(int col = 0; col < Ax->cols; col++)
			REAL(expCovExt)[col * Ax->rows + row] =
				omxMatrixElement(Ax, row, col);
	setAttrib(algebra, install("UnfilteredExpCov"), expCovExt);
	UNPROTECT(1);
}

/*
 * omxFastRAMInverse
 * 			Calculates the inverse of (I-A) using an n-step Neumann series
 * Assumes that A reduces to all zeros after numIters steps
 *
 * params:
 * omxMatrix *A				: The A matrix.  I-A will be inverted.  Size MxM.
 * omxMatrix *Z				: On output: Computed (I-A)^-1. MxM.
 * omxMatrix *Ax			: Space for computation. MxM.
 * omxMatrix *I				: Identity matrix. Will not be changed on exit. MxM.
 */

void omxFastRAMInverse(int numIters, omxMatrix* A, omxMatrix* Z, omxMatrix* Ax, omxMatrix* I ) {
	
	omxMatrix* origZ = Z;
    double oned = 1, minusOned = -1.0;

	if(numIters == NA_INTEGER) {
		int ipiv[A->rows], lwork = 4 * A->rows * A->cols, k;		// TODO: Speedups can be made by preallocating this.
		double work[lwork];
		if(OMX_DEBUG_ALGEBRA) { Rprintf("RAM Algebra (I-A) inversion using standard (general) inversion.\n"); }

		/* Z = (I-A)^-1 */
		if(I->colMajor != A->colMajor) {
			omxTransposeMatrix(I);
		}
		omxCopyMatrix(Z, A);

		/* Z = (I-A)^-1 */
		// F77_CALL(omxunsafedgemm)(I->majority, Z->majority, &(I->cols), &(I->rows), &(Z->rows), &oned, I->data, &(I->cols), I->data, &(I->cols), &minusOned, Z->data, &(Z->cols));
		//omxDGEMM(FALSE, FALSE, oned, I, Z, minusOned, Z); //Tim, I think this is incorrect: 1.0*I*Z-Z = Z-Z = 0 but you want I-Z.  So this should be omxDGEMM(FALSE, FALSE, oned, I, I, minusOned, Z). -MDH
		omxDGEMM(FALSE, FALSE, oned, I, I, minusOned, Z);

		// F77_CALL(dgetrf)(&(Z->rows), &(Z->cols), Z->data, &(Z->leading), ipiv, &k);
		k = omxDGETRF(Z, ipiv);
		if(OMX_DEBUG) { Rprintf("Info on LU Decomp: %d\n", k); }
		if(k > 0) {
		        char errStr[250];
		        strncpy(errStr, "(I-A) is exactly singular.", 100);
		        omxRaiseError(A->currentState, -1, errStr);                    // Raise Error
		        return;
		}
		// F77_CALL(dgetri)(&(Z->rows), Z->data, &(Z->leading), ipiv, work, &lwork, &k);
		k = omxDGETRI(Z, ipiv, work, lwork);
		if(OMX_DEBUG_ALGEBRA) { Rprintf("Info on Invert: %d\n", k); }

		if(OMX_DEBUG_ALGEBRA) {omxPrint(Z, "Z");}

	} else {

		if(OMX_DEBUG_ALGEBRA) { Rprintf("RAM Algebra (I-A) inversion using optimized expansion.\n"); }

		/* Taylor Expansion optimized I-A calculation */
		if(I->colMajor != A->colMajor) {
			omxTransposeMatrix(I);
		}

		if(I->colMajor != Ax->colMajor) {
			omxTransposeMatrix(Ax);
		}

		omxCopyMatrix(Z, A);

		/* Optimized I-A inversion: Z = (I-A)^-1 */
		// F77_CALL(omxunsafedgemm)(I->majority, A->majority, &(I->cols), &(I->rows), &(A->rows), &oned, I->data, &(I->cols), I->data, &(I->cols), &oned, Z->data, &(Z->cols));  // Z = I + A = A^0 + A^1
		// omxDGEMM(FALSE, FALSE, 1.0, I, I, 1.0, Z); // Z == A + I
        
		for(int i = 0; i < A->rows; i++)
			omxSetMatrixElement(Z, i, i, 1);

		for(int i = 1; i <= numIters; i++) { // TODO: Efficiently determine how many times to do this
			// The sequence goes like this: (I + A), I + (I + A) * A, I + (I + (I + A) * A) * A, ...
			// Which means only one DGEMM per iteration.
			if(OMX_DEBUG_ALGEBRA) { Rprintf("....RAM: Iteration #%d/%d\n", i, numIters);}
			omxCopyMatrix(Ax, I);
			// F77_CALL(omxunsafedgemm)(A->majority, A->majority, &(Z->cols), &(Z->rows), &(A->rows), &oned, Z->data, &(Z->cols), A->data, &(A->cols), &oned, Ax->data, &(Ax->cols));  // Ax = Z %*% A + I
			omxDGEMM(FALSE, FALSE, oned, A, Z, oned, Ax);
			omxMatrix* m = Z; Z = Ax; Ax = m;	// Juggle to make Z equal to Ax
		}
		if(origZ != Z) { 	// Juggling has caused Ax and Z to swap
			omxCopyMatrix(Z, Ax);
		}
	}
    omxMatrixCompute(Z);    // Update the compute count for Z.
}

/*
 * omxCalculateRAMCovarianceAndMeans
 * 			Just like it says on the tin.  Calculates the mean and covariance matrices
 * for a RAM model.  M is the number of total variables, latent and manifest. N is
 * the number of manifest variables.
 *
 * params:
 * omxMatrix *A, *S, *F 	: matrices as specified in the RAM model.  MxM, MxM, and NxM
 * omxMatrix *M				: vector containing model implied means. 1xM
 * omxMatrix *Cov			: On output: model-implied manifest covariance.  NxN.
 * omxMatrix *Means			: On output: model-implied manifest means.  1xN.
 * int numIterations		: Precomputed number of iterations of taylor series expansion.
 * omxMatrix *I				: Identity matrix.  If left NULL, will be populated.  MxM.
 * omxMatrix *Z				: On output: Computed (I-A)^-1. MxM.
 * omxMatrix *Y, *X, *Ax	: Space for computation. NxM, NxM, MxM.  On exit, populated.
 */

void omxCalculateRAMCovarianceAndMeans(omxMatrix* A, omxMatrix* S, omxMatrix* F, omxMatrix* M, omxMatrix* Cov, omxMatrix* Means, int numIters, omxMatrix* I, omxMatrix* Z, omxMatrix* Y, omxMatrix* X, omxMatrix* Ax) {
	
	if(OMX_DEBUG) { Rprintf("Running RAM computation."); }
		
	double oned = 1.0, zerod=0.0;
	
	if(Ax == NULL || I == NULL || Z == NULL || Y == NULL || X == NULL) {
		error("Internal Error: RAM Metadata improperly populated.  Please report this to the OpenMx development team.");
	}
		
	if(Cov == NULL && Means == NULL) {
		return; // We're not populating anything, so why bother running the calculation?
	}
	
	// if(   (Cov->rows != Cov->cols)  || (A->rows  != A->cols)  // Conformance check
	// 	|| (X->rows  != Cov->cols)  || (X->cols  != A->rows)
	// 	|| (Y->rows  != Cov->cols)  || (Y->cols  != A->rows)
	// 	|| (Ax->rows != Cov->cols)  || (Ax->cols != A->rows)
	// 	|| (I->rows  != Cov->cols)  || (I->cols  != Cov->rows)
	// 	|| (Y->rows  != Cov->cols)  || (Y->cols  != A->rows)
	// 	|| (M->cols  != Cov->cols)  || (M->rows  != 1)
	// 	|| (Means->rows != 1)       || (Means->cols != Cov->cols) ) {
	// 		error("INTERNAL ERROR: Incorrectly sized matrices being passed to omxRAMObjective Calculation.\n Please report this to the OpenMx development team.");
	// }
	
	omxFastRAMInverse(numIters, A, Z, Ax, I );
	
    if(A->currentState->statusCode < 0) return;
	
	/* Cov = FZSZ'F' */
	if(OMX_DEBUG_ALGEBRA) { Rprintf("....DGEMM: %c %c %d %d %d %f %x %d %x %d %f %x %d.\n", *(F->majority), *(Z->majority), (F->rows), (Z->cols), (Z->rows), oned, F->data, (F->leading), Z->data, (Z->leading), zerod, Y->data, (Y->leading));}
	// F77_CALL(omxunsafedgemm)(F->majority, Z->majority, &(F->rows), &(Z->cols), &(Z->rows), &oned, F->data, &(F->leading), Z->data, &(Z->leading), &zerod, Y->data, &(Y->leading)); 	// Y = FZ
	omxDGEMM(FALSE, FALSE, 1.0, F, Z, 0.0, Y);

	if(OMX_DEBUG_ALGEBRA) { Rprintf("....DGEMM: %c %c %d %d %d %f %x %d %x %d %f %x %d.\n", *(Y->majority), *(S->majority), (Y->rows), (S->cols), (S->rows), oned, Y->data, (Y->leading), S->data, (S->leading), zerod, X->data, (X->leading));}
	// F77_CALL(omxunsafedgemm)(Y->majority, S->majority, &(Y->rows), &(S->cols), &(S->rows), &oned, Y->data, &(Y->leading), S->data, &(S->leading), &zerod, X->data, &(X->leading)); 	// X = FZS
	omxDGEMM(FALSE, FALSE, 1.0, Y, S, 0.0, X);

	if(OMX_DEBUG_ALGEBRA) { Rprintf("....DGEMM: %c %c %d %d %d %f %x %d %x %d %f %x %d.\n", *(X->majority), *(Y->minority), (X->rows), (Y->rows), (Y->cols), oned, X->data, (X->leading), Y->data, (Y->lagging), zerod, Cov->data, (Cov->leading));}
	// F77_CALL(omxunsafedgemm)(X->majority, Y->minority, &(X->rows), &(Y->rows), &(Y->cols), &oned, X->data, &(X->leading), Y->data, &(Y->leading), &zerod, Cov->data, &(Cov->leading));
	omxDGEMM(FALSE, TRUE, 1.0, X, Y, 0.0, Cov);
	 // Cov = FZSZ'F' (Because (FZ)' = Z'F')
	
	if(OMX_DEBUG_ALGEBRA) {omxPrintMatrix(Cov, "....RAM: Model-implied Covariance Matrix:");}
	
	if(M != NULL && Means != NULL) {
		// F77_CALL(omxunsafedgemv)(Y->majority, &(Y->rows), &(Y->cols), &oned, Y->data, &(Y->leading), M->data, &onei, &zerod, Means->data, &onei);
		omxDGEMV(FALSE, 1.0, Y, M, 0.0, Means);
		if(OMX_DEBUG_ALGEBRA) {omxPrintMatrix(Means, "....RAM: Model-implied Means Vector:");}
	}
}

void omxUpdateChildRAMObjective(omxObjective* tgt, omxObjective* src) {

	omxRAMObjective* tgtRAM = (omxRAMObjective*)(tgt->argStruct);
	omxRAMObjective* srcRAM = (omxRAMObjective*)(src->argStruct);

	omxUpdateMatrix(tgtRAM->cov, srcRAM->cov);
	if (tgtRAM->means && srcRAM->means) {
		omxUpdateMatrix(tgtRAM->means, srcRAM->means);	
	}

	if (tgt->subObjective != NULL) {
		tgt->subObjective->updateChildObjectiveFun(tgt->subObjective, src->subObjective);
	}

}

unsigned short int omxNeedsUpdateRAMObjective(omxObjective* oo) {
	if(OMX_DEBUG_ALGEBRA) { Rprintf("Checking if RAM needs update.  RAM uses A:0x%x, S:0x%x, F:0x%x, M:0x%x.\n",((omxRAMObjective*)oo->argStruct)->A, ((omxRAMObjective*)oo->argStruct)->S, ((omxRAMObjective*)oo->argStruct)->F, ((omxRAMObjective*)oo->argStruct)->M); }
	return(omxNeedsUpdate(((omxRAMObjective*)oo->argStruct)->A)
	 	|| omxNeedsUpdate(((omxRAMObjective*)oo->argStruct)->S)
	 	|| omxNeedsUpdate(((omxRAMObjective*)oo->argStruct)->F)
		|| omxNeedsUpdate(((omxRAMObjective*)oo->argStruct)->M));

	// Note: cov is data, and should never need updating.
}

void omxInitRAMObjective(omxObjective* oo, SEXP rObj) {
	
	omxState* currentState = oo->matrix->currentState;	

    if(OMX_DEBUG) { Rprintf("Initializing RAM objective.\n"); }
	
	int l, k;

	SEXP slotValue;
	
	/* Create and register subobjective */

	omxObjective *subObjective = omxCreateSubObjective(oo);
	strncpy(subObjective->objType, "omxRAMObjective", 16);
	
	omxRAMObjective *RAMobj = (omxRAMObjective*) R_alloc(1, sizeof(omxRAMObjective));
	
	/* Set Subobjective Calls and Structures */
	subObjective->objectiveFun = omxCallRAMObjective;
	subObjective->needsUpdateFun = omxNeedsUpdateRAMObjective;
	subObjective->destructFun = omxDestroyRAMObjective;
	subObjective->setFinalReturns = NULL;
	subObjective->populateAttrFun = omxPopulateRAMAttributes;
	subObjective->updateChildObjectiveFun = omxUpdateChildRAMObjective;
	subObjective->argStruct = (void*) RAMobj;
	
	/* Set up objective structures */
	if(OMX_DEBUG) { Rprintf("Initializing RAM Meta Data for objective function.\n"); }

	if(OMX_DEBUG) { Rprintf("Processing M.\n"); }
	RAMobj->M = omxNewMatrixFromIndexSlot(rObj, currentState, "M");

	if(OMX_DEBUG) { Rprintf("Processing A.\n"); }
	RAMobj->A = omxNewMatrixFromIndexSlot(rObj, currentState, "A");

	if(OMX_DEBUG) { Rprintf("Processing S.\n"); }
	RAMobj->S = omxNewMatrixFromIndexSlot(rObj, currentState, "S");

	if(OMX_DEBUG) { Rprintf("Processing F.\n"); }
	RAMobj->F = omxNewMatrixFromIndexSlot(rObj, currentState, "F");

	if(OMX_DEBUG) { Rprintf("Processing usePPML.\n"); }
	PROTECT(slotValue = GET_SLOT(rObj, install("usePPML")));
	RAMobj->usePPML = INTEGER(slotValue)[0]; 
	UNPROTECT(1);

	if(RAMobj->usePPML) {
		PROTECT(slotValue = GET_SLOT(rObj, install("ppmlData")));
		RAMobj->ppmlData = omxNewDataFromMxData(NULL, slotValue, currentState);
		UNPROTECT(1);

		RAMobj->cov = omxDataMatrix(RAMobj->ppmlData, NULL);

		if(OMX_DEBUG) { Rprintf("Processing PPML observed means.\n"); }
		RAMobj->ppmlMeans = omxDataMeans(RAMobj->ppmlData, 0, NULL);
		if(OMX_DEBUG && RAMobj->means == NULL) { Rprintf("RAM: No PPML Observed Means.\n"); }
	} else {
		RAMobj->ppmlData  = NULL;
		RAMobj->ppmlCov   = NULL;
		RAMobj->ppmlMeans = NULL;
	}

	/* Identity Matrix, Size Of A */
	if(OMX_DEBUG) { Rprintf("Generating I.\n"); }
	RAMobj->I = omxNewIdentityMatrix(RAMobj->A->rows, currentState);
	omxRecompute(RAMobj->I);
	RAMobj->lilI = omxNewIdentityMatrix(RAMobj->F->rows, currentState);
	omxRecompute(RAMobj->lilI);
	

	if(OMX_DEBUG) { Rprintf("Processing expansion iteration depth.\n"); }
	PROTECT(slotValue = GET_SLOT(rObj, install("depth")));
	RAMobj->numIters = INTEGER(slotValue)[0];
	if(OMX_DEBUG) { Rprintf("Using %d iterations.", RAMobj->numIters); }
	UNPROTECT(1);

	l = RAMobj->F->rows;
	k = RAMobj->A->cols;

	if(OMX_DEBUG) { Rprintf("Generating internals for computation.\n"); }

	RAMobj->Z = 	omxInitMatrix(NULL, k, k, TRUE, currentState);
	RAMobj->Ax = 	omxInitMatrix(NULL, k, k, TRUE, currentState);
	RAMobj->W = 	omxInitMatrix(NULL, k, k, TRUE, currentState);
	RAMobj->U = 	omxInitMatrix(NULL, l, k, TRUE, currentState);
	RAMobj->Y = 	omxInitMatrix(NULL, l, k, TRUE, currentState);
	RAMobj->X = 	omxInitMatrix(NULL, l, k, TRUE, currentState);
	RAMobj->EF= 	omxInitMatrix(NULL, k, l, TRUE, currentState);
	RAMobj->V = 	omxInitMatrix(NULL, l, k, TRUE, currentState);
	RAMobj->ZSBC = 	omxInitMatrix(NULL, k, l, TRUE, currentState);
	RAMobj->C = 	omxInitMatrix(NULL, l, l, TRUE, currentState);
	
	if(omxIsMatrix(RAMobj->A)) {
		RAMobj->dA = omxInitMatrix(NULL, k, k, TRUE, currentState);
	} else {
		RAMobj->dA = NULL;
	}
	if(omxIsMatrix(RAMobj->S)) {
		RAMobj->dS = omxInitMatrix(NULL, k, k, TRUE, currentState);
	} else {
		RAMobj->dS = NULL;
	}
	if(RAMobj->M != NULL) {
		RAMobj->dM = 	omxInitMatrix(NULL, 1, k, TRUE, currentState);
		RAMobj->ZM = 	omxInitMatrix(NULL, k, 1, TRUE, currentState);
		RAMobj->bCB = 	omxInitMatrix(NULL, k, 1, TRUE, currentState);
		RAMobj->b = 	omxInitMatrix(NULL, l, 1, TRUE, currentState);
		RAMobj->tempVec = 	omxInitMatrix(NULL, k, 1, TRUE, currentState);
		RAMobj->bigSum = 	omxInitMatrix(NULL, k, 1, TRUE, currentState);
		RAMobj->lilSum = 	omxInitMatrix(NULL, l, 1, TRUE, currentState);
		RAMobj->beCov = 	omxInitMatrix(NULL, l, 1, TRUE, currentState);
		RAMobj->Mns = 	NULL;
    }

	RAMobj->cov = 		omxInitMatrix(NULL, l, l, TRUE, currentState);

	if(RAMobj->M != NULL) {
		RAMobj->means = 	omxInitMatrix(NULL, 1, l, TRUE, currentState);
	} else {
	    RAMobj->means  = 	NULL;
    }

    // Derivative space:
	RAMobj->eqnOuts = NULL;
	RAMobj->dAdts = NULL;
	RAMobj->dSdts = NULL;
	RAMobj->dMdts = NULL;
	RAMobj->paramVec = NULL;
	RAMobj->D =     NULL;
	RAMobj->pNums = NULL;
	RAMobj->nParam = -1;
	RAMobj->eCov=       omxInitMatrix(NULL, l, l, TRUE, currentState);

	/* Create parent objective */

	omxCreateMLObjective(oo, rObj, RAMobj->cov, RAMobj->means);

	// For fast gradient calculations
	// If this block doesn't execute, we don't use gradients.
	if(!strncmp("omxMLObjective", oo->objType, 14)) {
	    // Mode switch here.  Faster one is:
        omxSetMLObjectiveGradient(oo, fastRAMGradientML);
	    // Otherwise, call 
        // omxSetMLObjectiveGradientComponents(oo, calculateRAMGradientComponents);
	    // Note that once FIML kicks in, components should still work.
	    // Probably, we'll use the same thing for WLS.
	    RAMobj->D = ((omxMLObjective*)(oo->argStruct))->observedCov;
	    RAMobj->Mns = ((omxMLObjective*)(oo->argStruct))->observedMeans;
        RAMobj->n = ((omxMLObjective*)(oo->argStruct))->n;
        
    }
}


/*
 * fastRAMGradientML
 * 			Calculates the derivatives of the likelihood for a RAM model.  
 * Locations of free parameters are extracted from the omxObjective's state object.
 *
 * params:
 * omxMatrix *A, *S, *F 	: matrices as specified in the RAM model.  MxM, MxM, and NxM
 * omxMatrix *M				: vector containing model implied means. 1xM
 * omxMatrix *Cov			: On output: model-implied manifest covariance.  NxN.
 * omxMatrix *Means			: On output: model-implied manifest means.  1xN.
 * int numIterations		: Precomputed number of iterations of taylor series expansion.
 * omxMatrix *I				: Identity matrix.  If left NULL, will be populated.  MxM.
 * omxMatrix *Z				: On output: Computed (I-A)^-1. MxM.
 * omxMatrix *Y, *X, *Ax	: Space for computation. NxM, NxM, MxM.  On exit, populated.
 */

void fastRAMGradientML(omxObjective* oo, double* result) {
    
    // This calculation is based on von Oertzen and Brick, in prep.
    //
    // Steps:
    // 1) (Re) Calculate current A, S, F, M, and Z (where Z = (I-A)^-1)
    // 2) cov = current Expected Covariance (fxf)
    // 3) eCov = cov^(-1) (fxf) *
    // 4) ZS = Z S  (axa) *
    // 5) B = F Z   (fxa) *
    // 6) C = I - eCov D, where D is the observed covariance matrix (fxf) *
    // 7) BC = B^T C (axf) *
    // 8) ZSB = ZS (B^T) *
    // 8) ZSBC = ZS BC (axf) *
    // 9) eCovB = cov^(-1) B (fxa) *
    // For Means:
        // 10) Means = BM (fx1) *
        // 11) b = Means - d, where d is the observed means matrix (fx1)
        // 12) beCov = b cov^(-1) (1xf) *
        // 13) beCovB = b cov^(-1) B (1xa) *

    
    // 14) For each location in locations: (Note: parallelize, gradients here!)
    //      1A) Calculate dA/dt and dS/dt by substituting 1s into empty matrices
    //      1B) eqnList = 2 * ADB(eCovB, dAdt, ZSBC)
    //      1C) eqnList2 = ADB(eCovB, dSdt, BC)  // Preferably, build
    //      2) sum = eqnList1 + eqnList2
    //      If Means:
    //          3) sumVec = beCovB dAdt ZS
    //          4) sumVec += beCovB ZS dAdt
    //          5) sumVec += beCovB dSdt
    //          6) sumVec *= B^T
    //          7) sumVec += 2 * B dAdt Means
    //          8) sumVec += 2 * B dMdt
    //          9) sum += sumVec beCov

    //  Right.  All the * elements are saved for hessian computation.
 
    if(OMX_DEBUG) {Rprintf("Calculating fast RAM/ML gradient.\n"); }
    
    omxRAMObjective* oro = (omxRAMObjective*)(oo->subObjective->argStruct);
                                    // Size reference: A is axa, F is fxa
    omxMatrix* A = oro->A;          // axa
    omxMatrix* S = oro->S;          // axa
    omxMatrix* F = oro->F;          // fxa
    omxMatrix* M = oro->M;          // 1xa
    omxMatrix* Z = oro->Z;          // axa

    omxMatrix* cov = oro->cov;      // fxf
    omxMatrix* means = oro->means;  // fx1
    omxMatrix* eCov = oro->eCov;    // fxf

	omxMatrix* ZS = oro->Ax;        // axa
	
    omxMatrix* B = oro->X;          // fxa
    omxMatrix* C = oro->C;          // fxf
    omxMatrix* BC = oro->EF;        // fxa
    omxMatrix* ZSBC = oro->ZSBC;    // fxa
    omxMatrix* eCovB = oro->U;      // fxa

    omxMatrix* ZM = oro->ZM;        // 1xa
    omxMatrix* beCov = oro->beCov;  // 1xf
    omxMatrix* bCB = oro->bCB;      // 1xa
    omxMatrix* b = oro->b;          // 1xa
    
    omxMatrix** dAdts = oro->dAdts;
    omxMatrix** dSdts = oro->dSdts;
    omxMatrix** dMdts = oro->dMdts;
    
    omxMatrix*** eqnOuts = oro->eqnOuts;
    
    omxMatrix* tempVec = oro->tempVec;
    omxMatrix* bigSum  = oro->bigSum;
    omxMatrix* lilSum  = oro->lilSum;
    omxMatrix* D = oro->D;
    omxMatrix* Mns = oro->Mns;
    
    omxMatrix* lilI = oro->lilI;
    omxMatrix* paramVec = oro->paramVec;
    
    double n = oro->n;
    
    int* pNums = oro->pNums;
    int nParam = oro->nParam;
    
    int Amat = A->matrixNumber;
    int Smat = S->matrixNumber;
    int Mmat = 0;
    double oned = 1.0, minusoned = -1.0;
    int onei = 1;
    int info;
    char u = 'U';
    if(M != NULL) Mmat = M->matrixNumber;
    
    omxFreeVar* varList = oo->matrix->currentState->freeVarList;

    omxMatrix *eqnList1[1], *eqnList2[1];

    if(nParam < 0) {
        nParam = 0;
        int nTotalParams = oo->matrix->currentState->numFreeParams;
        if(OMX_DEBUG) { Rprintf("Planning Memory for Fast Gradient Calculation: Using %d params.\n", nParam); }
        unsigned short int calc[nTotalParams]; 
        // Work out the set of parameters for which we can calculate gradients
        // TODO: Potential speedup by splitting this to calculate dA, dS, and dM separately
        for(int parm = 0; parm < nTotalParams; parm++) {
            omxFreeVar ofv = varList[parm];
            calc[parm] = 0;
            for(int loc = 0; loc < ofv.numLocations; loc++) {
                int varMat = ~(ofv.matrices[loc]);
                if(varMat == Amat || varMat == Smat || (M != NULL && varMat == Mmat)) {
                    calc[parm] = 1;
                } else {
                    calc[parm] = 0;
                    break;
                }
            }
            if(calc[parm]) {
                nParam++;
            }
        }
        oro->nParam = nParam;
        pNums = (int*)R_alloc(nParam, sizeof(int));

        int nextFree = 0;
        // Populate pNums with numbers of the parameters to calculate
        for(int parm = 0; parm < nTotalParams && nextFree < nParam; parm++) {
            if(calc[parm]) {
                pNums[nextFree] = parm;
                nextFree++;

            }
        }
        oro->pNums = pNums;
        
        oro->dAdts = (omxMatrix**) R_alloc(nParam, sizeof(omxMatrix*));
        oro->dSdts = (omxMatrix**) R_alloc(nParam, sizeof(omxMatrix*));
        oro->dMdts = (omxMatrix**) R_alloc(nParam, sizeof(omxMatrix*));
        
        oro->eqnOuts = (omxMatrix***) R_alloc(1, sizeof(omxMatrix**));
        oro->eqnOuts[0] = (omxMatrix**) R_alloc(nParam, sizeof(omxMatrix*));
        
        int a = A->rows;
        omxState* currentState = oo->matrix->currentState;
        for(int i = 0; i < nParam; i++) {
            oro->dAdts[i] = omxInitMatrix(NULL, a, a, TRUE, currentState);
            oro->dSdts[i] = omxInitMatrix(NULL, a, a, TRUE, currentState);
            oro->dMdts[i] = omxInitMatrix(NULL, 1, a, TRUE, currentState);
            
            oro->eqnOuts[0][i] = omxInitMatrix(NULL, a, a, TRUE, currentState);
        }
        oro->paramVec = omxInitMatrix(NULL, nParam, 1, TRUE, currentState);
        paramVec = oro->paramVec;
        dAdts = oro->dAdts;
        dSdts = oro->dSdts;
        dMdts = oro->dMdts;
        
        eqnOuts = oro->eqnOuts;
    }

    double covInfluence[nParam];
    double meanInfluence[nParam];
    

    // Rprintf("Memory Set Up.\n"); //:::DEBUG:::

    // 1) (Re) Calculate current A, S, F, M, and Z (where Z = (I-A)^-1)
    // 2) cov = current Expected Covariance (fxf)
    // Theoretically, we should calculate current Objective values 
    // But we can safely assume this has already been done
    // That assumption may no longer be valid in FIML.
    // omxObjectiveCompute(oo);

    // 3) eCov = cov^(-1) (fxf)
    omxCopyMatrix(eCov, cov);				// But expected cov is destroyed in inversion
    
    F77_CALL(dpotrf)(&u, &(eCov->cols), eCov->data, &(eCov->cols), &info);

    if(OMX_DEBUG_ALGEBRA) { Rprintf("Info on LU Decomp: %d\n", info);}
    if(info > 0) {
        char *errstr = calloc(250, sizeof(char));
        sprintf(errstr, "Expected covariance matrix is non-positive-definite");
        if(oo->matrix->currentState->computeCount <= 0) {
            strncat(errstr, " at starting values", 20);
        }
        strncat(errstr, ".\n", 3);
        omxRaiseError(oo->matrix->currentState, -1, errstr);                        // Raise error
        free(errstr);
        return;                                                                     // Leave output untouched
    }
    
    F77_CALL(dpotri)(&u, &(eCov->rows), eCov->data, &(eCov->cols), &info);
    if(info > 0) {
        char *errstr = calloc(250, sizeof(char));
        sprintf(errstr, "Expected covariance matrix is not invertible");
        if(oo->matrix->currentState->computeCount <= 0) {
            strncat(errstr, " at starting values", 20);
        }
        strncat(errstr, ".\n", 3);
        omxRaiseError(oo->matrix->currentState, -1, errstr);                        // Raise error
        free(errstr);
        return;
    }

    // 4) ZS = Z S  (axa) *
    omxDSYMM(FALSE, 1.0, S, Z, 0.0, ZS);
    omxDGEMM(FALSE, FALSE, 1.0, Z, S, 0.0, ZS);
    
    // 5) B = F Z   (fxa) *
    omxDGEMM(FALSE, FALSE, 1.0, F, Z, 0.0, B);
    
    // 6) C = I - eCov D, where D is the observed covariance matrix (fxf)
    omxCopyMatrix(C, lilI);
    omxDSYMM(TRUE, -1.0, eCov, D, 1.0, C);

    
    // 7) BC = B^T C (axf) *
    omxDGEMM(TRUE, FALSE, 1.0, B, C, 0.0, BC);
    
    // 8) ZSBC = ZS BC (axf) *
    omxDGEMM(FALSE, FALSE, 1.0, ZS, BC, 0.0, ZSBC);
    
    // 9) eCovB = cov^(-1) B (fxa) *
    omxDSYMM(TRUE, 1.0, eCov, B, 0.0, eCovB);
    
    if(M != NULL) {
        // 10) Means = BM (fx1) *
        // This is calculated during objective computation.
        // 11) b = Means - d, where d is the observed means matrix (fx1)

        omxCopyMatrix(b, Mns);
    	F77_CALL(daxpy)(&(means->cols), &minusoned, means->data, &onei, b->data, &onei);
    
        // 11) beCov = b cov^(-1) (1xf) *
        omxDSYMV(1.0, eCov, b, 0.0, beCov);
        
        // 12) beCovB = b cov^(-1) B (1xa) *
        omxDGEMV(TRUE, 1.0, eCovB, b, 0.0, bCB);
        
        // 13) ZM = Z %*% M (1xa)
        omxDGEMV(FALSE, 1.0, Z, M, 0.0, ZM);
    }
    
    // 14) For each location in locations:
    //      1A) Calculate dA/dt, dS/dt, and dM/dt by substituting 1s into empty matrices
    //      1B) 1 = 2 * ADB(eCovB, dAdt, ZSBC)
    
    eqnList1[0] = eCovB;
    eqnList2[0] = ZSBC;
    ADB(eqnList1, eqnList2, 1, dAdts, Amat, varList, pNums, nParam, eqnOuts);
    omxMatrixTrace(eqnOuts[0], nParam, paramVec);
    
    for(int i = 0; i < nParam; i++)
        covInfluence[i] = 2 * omxVectorElement(paramVec, i);
        
    
    //      1C) eqnList2 = ADB(eCovB, dSdt, BC)
    eqnList1[0] = eCovB;
    eqnList2[0] = BC;
    ADB(eqnList1, eqnList2, 1, dSdts, Smat, varList, pNums, nParam, eqnOuts);
    omxMatrixTrace(eqnOuts[0], nParam, paramVec);

    //      2) sum = eqnList1 + eqnList2
    F77_CALL(daxpy)(&nParam, &oned, paramVec->data, &onei, covInfluence, &onei);

    if(M != NULL) {
    //      If Means:

        //          Just populate dMdts
        ADB(NULL, NULL, 0, dMdts, Mmat, varList, pNums, nParam, NULL);

        //  This needs to be done one-per-param.
        //  Parallelize here.
        for(int pNum = 0; pNum < nParam; pNum++) {
            //          3) sumVec = beCovB dAdt ZS (1xa)
            omxDGEMV(TRUE, 1.0, dAdts[pNum], bCB, 0.0, tempVec);
            omxDGEMV(TRUE, 1.0, ZS, tempVec, 0.0, bigSum);
            // Term with dSigma/dTheta
            //          4) sumVec += beCovB ZS dAdt (1xa)
            omxDGEMV(TRUE, 1.0, ZS, bCB, 0.0, tempVec);
            omxDGEMV(TRUE, 1.0, dAdts[pNum], tempVec, 1.0, bigSum);

            //          5) sumVec += beCovB dSdt (1xa)
            omxDGEMV(TRUE, 1.0, dSdts[pNum], bCB, 1.0, bigSum);

            //          6) sumVec *= B^T --> 1xf
            omxDGEMV(FALSE, 1.0, B, bigSum, 0.0, lilSum);

            // Factorizing dMudt
            //          7) sumVec += 2 * B dAdt ZM (1xa)
            //              Note: in the paper, expected manifest means are Bm
            omxDGEMV(FALSE, 1.0, dAdts[pNum], ZM, 0.0, tempVec);
            omxDGEMV(FALSE, 2.0, B, tempVec, 1.0, lilSum); // :::DEBUG:::<-- Save B %*% tempVec

            //          8) sumVec += 2 * B dMdt (1xf)
            omxDGEMV(FALSE, 2.0, B, dMdts[pNum], 1.0, lilSum); // :::DEBUG::: <-- Save B %*% dM

            //          9) sum = sumVec beCov (1x1)
            int len = lilSum->rows * lilSum->cols;
            meanInfluence[pNum] = F77_CALL(ddot)(&len, lilSum->data, &onei, beCov->data, &onei);
        }
            
    } else { 
        for(int i = 0; i < nParam; i++)
            meanInfluence[i] = 0;
    }
    
    for(int pNum = 0; pNum < nParam; pNum++) {
        int nextP = pNums[pNum]; // Original varList parameter number
        result[pNum] = (covInfluence[pNum] * (n-1)) - (meanInfluence[pNum] * n);
        if(OMX_DEBUG) {
            Rprintf("Revised calculation for Gradient value %d: Cov: %3.9f, Mean: %3.9f, total: %3.9f\n",
                nextP, covInfluence[pNum], meanInfluence[pNum], result[pNum]);
        }
    }
    
    // Phew.

}


void calculateRAMGradientComponents(omxObjective* oo, omxMatrix** dSigmas, omxMatrix** dMus, int* status) {

    // Steps:
    // 1) (Re) Calculate current A, S, F, and Z (where Z = (I-A)^-1)
    // 2) E = Z S Z^T
    // 3) B = F Z
    // 4) For each location in locations:
    //      1) Calculate dA/dt, dS/dt, and dM/dt by substituting 1s into empty matrices
    //      2) C = B dAdt E F^T
    //      3) dSigma/dT = C + C^T + B dS/dT B^T
    //      4) dM/dT = B dA/dT Z b + B dM/dT

    omxRAMObjective* oro = (omxRAMObjective*) (oo->subObjective->argStruct);
                                // Size reference: A is axa, F is fxf
    omxMatrix* A = oro->A;      // axa
    omxMatrix* S = oro->S;      // axa
    omxMatrix* F = oro->F;      // fxf
    omxMatrix* Z = oro->Z;      // axa
    // omxMatrix* I = oro->I;      // axa
    omxMatrix* M = oro->M;      // 1xf
    omxMatrix* B = oro->X;      // fxa
    omxMatrix* U = oro->U;      // fxa
    omxMatrix* EF = oro->EF;    // fxa
    omxMatrix* ZM = oro->ZM;    // 1xf
    omxMatrix* V = oro->V;      // fxa
    omxMatrix* W = oro->W;      // axa
	omxMatrix* E = oro->Ax;     // axa
    
    omxMatrix* dA = oro->dA;
    omxMatrix* dS = oro->dS;
    omxMatrix* dM = oro->dM;
    
    // int numIters = oro->numIters;
    int Amat = A->matrixNumber;
    int Smat = S->matrixNumber;
    int Mmat = 0;
    if(M != NULL) Mmat = M->matrixNumber;
    
    omxFreeVar* varList = oo->matrix->currentState->freeVarList;
    int nLocs = oo->matrix->currentState->numFreeParams;

    // Assumed.
    // if(omxNeedsUpdate(Z)) {
    //     // (Re)Calculate current A, S, F, and Z
    //     omxCalculateRAMCovarianceAndMeans(A, S, F, M, cov, means, numIters, I, Z, oro->Y, B, Ax);
    // }
    
    // TODO: Handle repeated-call cases.
    // E = Z S (Z^T)
    omxDGEMM(FALSE, FALSE, 1.0, Z, S, 0.0, W);
    omxDGEMM(FALSE, TRUE, 1.0, W, Z, 0.0, E);
    
    //EF = E F^T
    omxDGEMM(FALSE, TRUE, 1.0, E, F, 0.0, EF);

    // B = F Z
    omxDGEMM(FALSE, FALSE, 1.0, F, Z, 0.0, B);
    
    // ZM = ZM
    if(M != NULL)
        omxDGEMV(FALSE, 1.0, Z, M, 0.0, ZM);
    
    // Step through free params 
    // Note: This should be parallelized at the next level up.
    //  This function itself should serially perform one computation for each location
    //  given in the sequence.  Always write to the appropriate location so that writes
    //  can be shared across thread-level parallelism.
    
    for(int param = 0; param < nLocs; param++) {
        //  Repeated from above.  For each parameter:
        //      1) Calculate dA/dt, dS/dt, and dM/dt by substituting 1s into empty matrices
        
        omxFreeVar var = varList[param];
        
        // Zero dA, dS, and dM.  // TODO: speed
        for( int k = 0; k < dA->cols; k++) {
            for( int j = 0; j < dA->rows; j++) {
                omxSetMatrixElement(dA, j, k, 0.0);
                omxSetMatrixElement(dS, j, k, 0.0);
            }
            if(M != NULL) omxSetMatrixElement(dM, 0, k, 0.0);
        }
        status[param] = 0;

        // Create dA, dS, and dM mats for each Free Parameter
        for(int varLoc = 0; varLoc < var.numLocations; varLoc++) {
            int varMat = ~var.matrices[varLoc]; // Matrices are numbered from ~0 to ~N
            if(varMat == Amat) {
                omxSetMatrixElement(dA, var.row[varLoc], var.col[varLoc], 1);
            } else if(varMat == Smat) {
                omxSetMatrixElement(dS, var.row[varLoc], var.col[varLoc], 1);
            } else if(varMat == Mmat && M != NULL) {
                omxSetMatrixElement(dM, var.row[varLoc], var.col[varLoc], 1);
            } else {
                // This parameter has some outside effect
                // We cannot directly estimate its effects on the likelihood
                // For now, skip.
                // TODO: Find a way to deal with these situations
                status[param] = -1;
                if(OMX_DEBUG) { Rprintf("Skipping parameter %d because %dth element has outside influence %d. Looking for %d, %d, or %d.\n", param, varLoc, varMat, Amat, Smat, Mmat);}
                break;
            }
        }
        
        if(status[param] < 0) {
            continue;  // Skip this one.
        }
        
        omxMatrix* C = dSigmas[param];
        omxMatrix* dMu = dMus[param];
        
        //      2) C = B dAdt EF
        omxDGEMM(FALSE, FALSE, 1.0, B, dA, 0.0, U);
        omxDGEMM(FALSE, FALSE, 1.0, U, EF, 0.0, C);
        
        // Note: U is saved here for calculation of M, below.
        //      3) dSigma/dT = C + C^T + B dS/dT B^T
        omxAddOwnTranspose(&C, 0, C);
        omxDGEMM(FALSE, FALSE, 1.0, B, dS, 0.0, V);
        omxDGEMM(FALSE, TRUE,  1.0, V, B, 1.0, C);
        
        if(M != NULL) {
            //      4) dM/dT = B dA/dT Z M + B dM/dT
            omxDGEMV(FALSE, 1.0, U, ZM, 0.0, dMu);
            omxDGEMV(FALSE, 1.0, B, dM, 1.0, dMu);

        }

    }

}
