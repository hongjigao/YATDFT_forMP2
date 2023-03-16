#include "build_Dmat.h"

#include <assert.h>
#include <float.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "TinyDFT_typedef.h"
#include "linalg_lib_wrapper.h"
#include "utils.h"

void TinyDFT_build_Dmat_SAD(TinyDFT_p TinyDFT, double *D_mat) {
  assert(TinyDFT != NULL);

  int natom = TinyDFT->natom;
  int charge = TinyDFT->charge;
  int electron = TinyDFT->electron;
  int nbf = TinyDFT->nbf;
  int mat_size = TinyDFT->mat_size;
  BasisSet_p basis = TinyDFT->basis;

  memset(D_mat, 0, DBL_MSIZE * mat_size);

  double *guess;
  int spos, epos, ldg;
  for (int i = 0; i < natom; i++) {
    CMS_getInitialGuess(basis, i, &guess, &spos, &epos);
    ldg = epos - spos + 1;
    double *D_mat_ptr = D_mat + spos * nbf + spos;
    copy_matrix_block(sizeof(double), ldg, ldg, guess, ldg, D_mat_ptr, nbf);
  }

  // Scaling the initial density matrix according to the charge and neutral
  double R = 1.0;
  if (charge != 0 && electron != 0)
    R = (double)(electron - charge) / (double)(electron);
  R *= 0.5;
  for (int i = 0; i < mat_size; i++) D_mat[i] *= R;
}

void TinyDFT_build_Cocc_from_Dmat(TinyDFT_p TinyDFT, const double *D_mat,
                                  double *Cocc_mat) {
  int nbf = TinyDFT->nbf;
  int n_occ = TinyDFT->n_occ;
  int mat_size = TinyDFT->mat_size;
  double *Chol_mat = TinyDFT->tmp_mat;

  int rank;
  int *piv = (int *)malloc(sizeof(int) * nbf);
  memcpy(Chol_mat, D_mat, DBL_MSIZE * mat_size);
  LAPACKE_dpstrf(LAPACK_ROW_MAJOR, 'L', nbf, Chol_mat, nbf, piv, &rank, 1e-12);

  if (rank < n_occ) {
    for (int i = 0; i < nbf; i++) {
      double *Chol_row = Chol_mat + i * nbf;
      for (int j = rank; j < n_occ; j++) Chol_row[j] = 0.0;
    }
  }

  for (int i = 0; i < n_occ; i++) {
    double *Cocc_row = Cocc_mat + i * n_occ;
    double *Chol_row = Chol_mat + i * nbf;
    for (int j = 0; j < i; j++) Cocc_row[j] = Chol_row[j];
    for (int j = i; j < n_occ; j++) Cocc_row[j] = 0.0;
  }
  for (int i = n_occ; i < nbf; i++) {
    double *Cocc_row = Cocc_mat + i * n_occ;
    double *Chol_row = Chol_mat + i * nbf;
    memcpy(Cocc_row, Chol_row, DBL_MSIZE * n_occ);
  }

  free(piv);
}

static void qsort_dbl_key_int_val(double *key, int *val, int l, int r) {
  int i = l, j = r, iswap;
  double mid = key[(i + j) / 2], dswap;
  while (i <= j) {
    while (key[i] < mid) i++;
    while (key[j] > mid) j--;
    if (i <= j) {
      if (key[i] > key[j]) {
        iswap = val[i];
        val[i] = val[j];
        val[j] = iswap;

        dswap = key[i];
        key[i] = key[j];
        key[j] = dswap;
      }

      i++;
      j--;
    }
  }
  if (i < r) qsort_dbl_key_int_val(key, val, i, r);
  if (j > l) qsort_dbl_key_int_val(key, val, l, j);
}

double Calcnorm(const double *mat, int siz) {
  double norms = 0;
  for (int i = 0; i < siz; i++)
    for (int j = 0; j < siz; j++) {
      norms = norms + mat[i * siz + j] * mat[i * siz + j];
    }

  return norms;
}

void TinyDFT_build_Cocc_from_Dmat_eig(TinyDFT_p TinyDFT, const double *D_mat,
                                      double *Cocc_mat) {
  int nbf = TinyDFT->nbf;
  int n_occ = TinyDFT->n_occ;
  int mat_size = TinyDFT->mat_size;
  int *ev_idx = TinyDFT->ev_idx;
  double *eigval = TinyDFT->eigval;
  double *tmp_mat = TinyDFT->tmp_mat;

  memcpy(tmp_mat, D_mat, DBL_MSIZE * mat_size);

  LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', nbf, tmp_mat, nbf, eigval);

  // Form the C_occ with eigenvectors corresponding to n_occ largest
  // abs(eigenvalues)
  for (int i = 0; i < nbf; i++) {
    eigval[i] = fabs(eigval[i]);
    ev_idx[i] = i;
  }
  qsort_dbl_key_int_val(eigval, ev_idx, 0, nbf - 1);
  for (int j = 0; j < n_occ; j++)
    for (int i = 0; i < nbf; i++)
      Cocc_mat[i * n_occ + j] = tmp_mat[i * nbf + ev_idx[j]];
}

void TinyDFT_build_Dmat_eig(TinyDFT_p TinyDFT, const double *F_mat,
                            const double *X_mat, double *D_mat,
                            double *Cocc_mat) {
  int nbf = TinyDFT->nbf;
  int n_occ = TinyDFT->n_occ;
  int mat_size = TinyDFT->mat_size;
  int *ev_idx = TinyDFT->ev_idx;
  double *eigval = TinyDFT->eigval;
  double *tmp_mat = TinyDFT->tmp_mat;

  // Notice: here F_mat is already = X^T * F * X
  memcpy(tmp_mat, F_mat, DBL_MSIZE * mat_size);

  // Diagonalize F = C0^T * epsilon * C0, and C = X * C0
  // [C0, E] = eig(F1), now C0 is stored in tmp_mat
  LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', nbf, tmp_mat, nbf,
                eigval);  // tmp_mat will be overwritten by eigenvectors
  // C = X * C0, now C is stored in D_mat
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nbf, nbf, nbf, 1.0,
              X_mat, nbf, tmp_mat, nbf, 0.0, D_mat, nbf);

  // Form the C_occ with eigenvectors corresponding to n_occ smallest
  // eigenvalues
  for (int i = 0; i < nbf; i++) ev_idx[i] = i;
  qsort_dbl_key_int_val(eigval, ev_idx, 0, nbf);
  for (int j = 0; j < n_occ; j++)
    for (int i = 0; i < nbf; i++)
      Cocc_mat[i * n_occ + j] = D_mat[i * nbf + ev_idx[j]];

  // D = C_occ * C_occ^T
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, nbf, nbf, n_occ, 1.0,
              Cocc_mat, n_occ, Cocc_mat, n_occ, 0.0, D_mat, nbf);
}

void TinyDFT_build_MP2info_eig(TinyDFT_p TinyDFT, const double *F_mat,
                               const double *X_mat, double *D_mat,
                               double *Cocc_mat, double *DC_mat,
                               double *Cvir_mat, double *orbitenergy_array) {
  int nbf = TinyDFT->nbf;
  int n_occ = TinyDFT->n_occ;
  int n_vir = TinyDFT->n_vir;
  int n_null= TinyDFT->n_null;
  int mat_size = TinyDFT->mat_size;
  int *ev_idx = TinyDFT->ev_idx;
  double *eigval = TinyDFT->eigval;
  double *tmp_mat = TinyDFT->tmp_mat;

  // Notice: here F_mat is already = X^T * F * X
  memcpy(tmp_mat, F_mat, DBL_MSIZE * mat_size);

  // Diagonalize F = C0^T * epsilon * C0, and C = X * C0
  // [C0, E] = eig(F1), now C0 is stored in tmp_mat
  LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', nbf, tmp_mat, nbf,
                eigval);  // tmp_mat will be overwritten by eigenvectors
  // C = X * C0, now C is stored in D_mat
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nbf, nbf, nbf, 1.0,
              X_mat, nbf, tmp_mat, nbf, 0.0, D_mat, nbf);
  // Form the C_occ with eigenvectors corresponding to n_occ smallest
  // eigenvalues
  for (int i = 0; i < nbf; i++) ev_idx[i] = i;
  qsort_dbl_key_int_val(eigval, ev_idx, 0, nbf);
  for (int j = 0; j < n_occ; j++)
    for (int i = 0; i < nbf; i++)
      Cocc_mat[i * n_occ + j] = D_mat[i * nbf + ev_idx[j]];
  // Form the C_vir with eigenvectors corresponding to n_vir largest eigenvalues
  for (int j = n_occ; j < n_occ+n_null; j++)
    for (int i = 0; i < nbf; i++)
      Cvir_mat[i * n_vir + j - n_occ] = 0;
  for (int j = n_occ+n_null; j < nbf; j++)
    for (int i = 0; i < nbf; i++)
      Cvir_mat[i * n_vir + j - n_occ] = D_mat[i * nbf + ev_idx[j]];
  // Form the orbitenergy_array with the eigenvalues
  for (int i = 0; i < nbf; i++) {
    orbitenergy_array[i] = eigval[ev_idx[i]];
  }

  // D = C_occ * C_occ^T
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, nbf, nbf, n_occ, 1.0,
              Cocc_mat, n_occ, Cocc_mat, n_occ, 0.0, D_mat, nbf);
  // DC=C_vir * C_vir^T
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, nbf, nbf, n_vir, 1.0,
              Cvir_mat, n_vir, Cvir_mat, n_vir, 0.0, DC_mat, nbf);
}





void TinyDFT_MP2process(TinyDFT_p TinyDFT) {
  int nbf = TinyDFT->nbf;
  int n_occ = TinyDFT->n_occ;
  int n_vir = TinyDFT->n_vir;
  int n_null= TinyDFT->n_null;
  int mat_size = TinyDFT->mat_size;
  int *ev_idx = TinyDFT->ev_idx;
  double *eigval = TinyDFT->eigval;
  double *tmp_mat = TinyDFT->tmp_mat;
  double *D_mat = TinyDFT->D_mat;
  double *DC_mat = TinyDFT->DC_mat;
  double *S_mat = TinyDFT->S_mat;
  double *X_mat = TinyDFT->X_mat;
  double *Cocc_mat = TinyDFT->Cocc_mat;
  double *Cvir_mat = TinyDFT->Cvir_mat;
  double *F_mat = TinyDFT->F_mat;

  double norm;

  printf(" %d basis functions, %d occupied orbits, %d virtual orbits (including %d rank deficient null orbits) \n",nbf,n_occ,n_vir,n_null);
  memcpy(tmp_mat, F_mat, DBL_MSIZE * mat_size);

  // Diagonalize F = C0^T * epsilon * C0, and C = X * C0
  // [C0, E] = eig(F1), now C0 is stored in tmp_mat
  LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', nbf, tmp_mat, nbf,
                eigval);  // tmp_mat will be overwritten by eigenvectors
  // C = X * C0, now C is stored in D_mat
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nbf, nbf, nbf, 1.0,
              X_mat, nbf, tmp_mat, nbf, 0.0, D_mat, nbf);
  // Form the C_occ with eigenvectors corresponding to n_occ smallest
  // eigenvalues
  for (int i = 0; i < nbf; i++) ev_idx[i] = i;
  qsort_dbl_key_int_val(eigval, ev_idx, 0, nbf);
  for (int j = 0; j < n_occ; j++)
    for (int i = 0; i < nbf; i++)
      Cocc_mat[i * n_occ + j] = D_mat[i * nbf + ev_idx[j]];
  // Form the C_vir with eigenvectors corresponding to n_vir largest eigenvalues
  for (int j = n_occ; j < n_occ+n_null; j++)
    for (int i = 0; i < nbf; i++)
      Cvir_mat[i * n_vir + j - n_occ] = 0;
  for (int j = n_occ+n_null; j < nbf; j++)
    for (int i = 0; i < nbf; i++)
      Cvir_mat[i * n_vir + j - n_occ] = D_mat[i * nbf + ev_idx[j]];

  // D = C_occ * C_occ^T
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, nbf, nbf, n_occ, 1.0,
              Cocc_mat, n_occ, Cocc_mat, n_occ, 0.0, D_mat, nbf);
  // DC=C_vir * C_vir^T
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, nbf, nbf, n_vir, 1.0,
              Cvir_mat, n_vir, Cvir_mat, n_vir, 0.0, DC_mat, nbf);
  
  //countx[0] means the number of elements larger than 1 and countx[1] means those larger than 0.1
  int countx[8];
  int county[8];
  double maxx=0;
  double maxy=0;
  for(int  i=0;i<8;i++)
  {
    countx[i]=0;
    county[i]=0;
  }
  double lgx,lgym2;
  for (int j = 0; j < nbf; j++)
    for (int i = 0; i < nbf; i++)
    {
      lgx=log10(fabs(D_mat[i*nbf+j]));
      if(lgx>0)
        countx[0]+=1;
      else if(lgx < -6)
        countx[7]+=1;
      else
      {
        countx[(int)ceil(-lgx)]+=1;
      }
      if(fabs(D_mat[i*nbf+j])>maxx)
        maxx=fabs(D_mat[i*nbf+j]);
    }

  for (int j = 0; j < nbf; j++)
    for (int i = 0; i < nbf; i++)
    {
      lgym2=log10(fabs(DC_mat[i*nbf+j]))-2;
      if(lgym2>0)
        county[0]+=1;
      else if(lgym2 < -6)
        county[7]+=1;
      else
      {
        county[(int)ceil(-lgym2)]+=1;
      }
      if(fabs(DC_mat[i*nbf+j])>maxy)
        maxy=fabs(DC_mat[i*nbf+j]);
    }
  
  printf("The number of elements in X and Y is %d\n", nbf*nbf);
  int sumx=0;
  for(int  i=0;i<8;i++)
  {
    printf("The number of elements in density matrix larger than 1e-%d is %d\n", i,countx[i]);
    sumx+=countx[i];
  }
  int sumy=0;
  for(int  i=0;i<8;i++)
  {
    printf("The number of elements in density complimentary matrix larger than 1e-%d is %d\n", i-2,county[i]);
    sumy+=county[i];
  }
  printf("The total counted X and Y elements are %d and %d\n",sumx,sumy);
  printf("The maximum absolute values of X and Y are %f and %f\n",maxx,maxy);
  int usefulx3=0;
  int usefuly3=0;
  int usefulx2=0;
  int usefuly2=0;
  int usefulx4=0;
  int usefuly4=0;
  for (int j = 0; j < nbf; j++)
    for (int i = 0; i < nbf; i++)
    {
            if(fabs(D_mat[i*nbf+j])>0.001*maxx)
                usefulx3+=1;
            if(fabs(DC_mat[i*nbf+j])>0.001*maxy)
                usefuly3+=1;
            if(fabs(D_mat[i*nbf+j])>0.01*maxx)
                usefulx2+=1;
            if(fabs(DC_mat[i*nbf+j])>0.01*maxy)
                usefuly2+=1;
            if(fabs(D_mat[i*nbf+j])>0.0001*maxx)
                usefulx4+=1;
            if(fabs(DC_mat[i*nbf+j])>0.0001*maxy)
                usefuly4+=1;
    }
  printf("If we choose the elements larger than 0.01*maximum, the number of elements in Xdense and Ydense are%d and %d\n",usefulx2,usefuly2);
  printf("If we choose the elements larger than 0.001*maximum, the number of elements in Xdense and Ydense are%d and %d\n",usefulx3,usefuly3);
  printf("If we choose the elements larger than 0.0001*maximum, the number of elements in Xdense and Ydense are%d and %d\n",usefulx4,usefuly4);


  for (int j = 0; j < nbf; j++)
    for (int i = 0; i < nbf; i++)
      D_mat[i * nbf + j] = D_mat[i * nbf + j] + DC_mat[i * nbf + j];

  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nbf, nbf, nbf, 1.0,
              D_mat, nbf, S_mat, nbf, 0.0, tmp_mat, nbf);
  norm = Calcnorm(tmp_mat, nbf);
  printf("The norm of (D+DC)*S is %f \n", norm);
  printf("--------------------------------------------------\n");
  printf("Test whether X*S*X is I\n");
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nbf, nbf, nbf, 1.0,
              TinyDFT->X_mat, nbf, TinyDFT->S_mat, nbf, 0.0, TinyDFT->tmp_mat,
              nbf);

  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nbf, nbf, nbf, 1.0,
              TinyDFT->tmp_mat, nbf, TinyDFT->X_mat, nbf, 0.0, TinyDFT->D_mat,
              nbf);

  norm = Calcnorm(D_mat, nbf);
  printf("The norm of XSX is %f\n", norm);
//  printf("--------------------------------------------------\n");
//  printf("The eigenvalues of S matrix are\n");

}


TinyDFT_build_energyweightedDDC(TinyDFT_p TinyDFT, const double *Cocc_mat, const double *Cvir_mat, const double *orbitenergy_array, double *D_mat, double *DC_mat, double Fermie, double talpha)
{
  int nbf = TinyDFT->nbf;
  int n_occ=TinyDFT->n_occ;
  int n_vir=TinyDFT->n_vir;
  double edif;
  double tmp_factor;
  //Initialize all the values in X(D) and Y(DC)
  for(int mu=0;mu<nbf;mu++)
      for(int nu=0;nu<nbf;nu++)
      {
        D_mat[mu*nbf+nu]=0;
        DC_mat[mu*nbf+nu]=0;
      }
  //Calculate X
  for(int i=0;i<n_occ;i++)
  {
    edif=orbitenergy_array[i]-Fermie;
    tmp_factor= exp(edif*talpha);
    for(int mu=0;mu<nbf;mu++)
      for(int nu=0;nu<nbf;nu++)
      {
        D_mat[mu*nbf+nu] += Cocc_mat[mu*n_occ+i]*Cocc_mat[nu*n_occ+i]*tmp_factor;
      }
  }
  //Calculate Y
  for(int a=n_occ;a<nbf;a++)
  {
    edif=Fermie - orbitenergy_array[a];
    tmp_factor= exp(edif*talpha);
    for(int mu=0;mu<nbf;mu++)
      for(int nu=0;nu<nbf;nu++)
      {
        DC_mat[mu*nbf+nu] += Cvir_mat[mu*n_vir+a-n_occ]*Cvir_mat[nu*n_vir+a-n_occ]*tmp_factor;
      }
  }
}