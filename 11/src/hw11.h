#ifndef HW11_H
#define HW11_H

#include <mpi.h>

int mpi_error(const int error_code);
int CreateGemmCommGroups(const int m, const int n, const MPI_Comm comm, MPI_Comm *comm_row, MPI_Comm *comm_col);
int PassTokenComm(const MPI_Comm comm, int *send_message, int *recv_message );
int CreateMatricies(MPI_Comm comm, int p, int q, int bm, int bn, int bk, int gm, int gn, int gk, double *a, double *b, double *c, double (*fa)(int i, int j), double (*fb)(int i, int j), double (*fc)(int i, int j));
int ShiftMPIMatrixLeft(MPI_Comm comm_row, int gm, int gn, int bm, int bn, int p, int q, double *a);
void PrintMPIMatrix(MPI_Comm comm, int gm, int gn, int bm, int bn, int p, int q, double *a);
int pdgemm(MPI_Comm comm, int p, int q, int bm, int bn, int bk, int gm, int gn, int gk, double *a, double *b, double *c);

#endif
