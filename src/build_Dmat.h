#ifndef __BUILD_DMAT_H__
#define __BUILD_DMAT_H__

#include "TinyDFT_typedef.h"

// All matrices used in this module is row-major, leading dimension = number of columns

#ifdef __cplusplus
extern "C" {
#endif

// Build initial density matrix using SAD (if SAD not available, D = 0)
// Input parameter:
//   TinyDFT : Initialized TinyDFT structure
// Output parameter:
//   D_mat : Initial density matrix, size nbf * nbf
void TinyDFT_build_Dmat_SAD(TinyDFT_p TinyDFT, double *D_mat);

// Do an incomplete Cholesky decomposition of D to form Cocc for density fitting
// Input parameters:
//   TinyDFT : Initialized TinyDFT structure
//   D_mat   : Density matrix, size nbf * nbf
// Output parameter:
//   Cocc_mat : Cocc matrix, size nbf * n_occ
void TinyDFT_build_Cocc_from_Dmat(TinyDFT_p TinyDFT, const double *D_mat, double *Cocc_mat);

// Do an eigen-decomposition of D to form Cocc for density fitting
// Input parameters:
//   TinyDFT : Initialized TinyDFT structure
//   D_mat   : Density matrix, size nbf * nbf
// Output parameter:
//   Cocc_mat : Cocc matrix, size nbf * n_occ
void TinyDFT_build_Cocc_from_Dmat_eig(TinyDFT_p TinyDFT, const double *D_mat, double *Cocc_mat);

// Build density matrix using eigen-decomposition
// Input parameter:
//   TinyDFT : Initialized TinyDFT structure
//   F_mat   : Fock matrix after DIIS, size nbf * nbf
//   X_mat   : Basis transformation matrix, size nbf * nbf
// Output parameter:
//   D_mat    : Density matrix, size nbf * nbf, D = Cocc * Cocc^T
//   Cocc_mat : Cocc matrix, size nbf * n_occ
void TinyDFT_build_Dmat_eig(TinyDFT_p TinyDFT, const double *F_mat, const double *X_mat, double *D_mat, double *Cocc_mat);


// Build density matrix and its complimentary using eigen-decomposition
// Input parameter:
//   TinyDFT : Initialized TinyDFT structure
//   F_mat   : Fock matrix after DIIS, size nbf * nbf
//   X_mat   : Basis transformation matrix, size nbf * nbf
// Output parameter:
//   D_mat    : Density matrix, size nbf * nbf, D = Cocc * Cocc^T
//   Cocc_mat : Cocc matrix, size nbf * n_occ
//   DC_mat   : Density complimentary matrix, size nbf * nbf, DC=Cvir * Cvir^T
//   Cvir_mat : Cvir matrix, size nbf * n_vir
void TinyDFT_build_MP2info_eig(TinyDFT_p TinyDFT, const double *F_mat, const double *X_mat, double *D_mat, double *Cocc_mat, double *DC_mat, double *Cvir_mat, double *orbitenergy_array);

void TinyDFT_MP2process(TinyDFT_p TinyDFT);

//static void qsort_dbl_key_int_val(double *key, int *val, int l, int r);


#ifdef __cplusplus
}
#endif

#endif
