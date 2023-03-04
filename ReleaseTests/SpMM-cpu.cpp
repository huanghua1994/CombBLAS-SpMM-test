#include <chrono>
#include <iostream>
#include <string>

#include "CombBLAS/CombBLAS.h"

using namespace std;
using namespace combblas;

typedef int    TEST_IDX_T;
typedef double TEST_NNZ_T;

int main(int argc, char* argv[])
{
    {
    int nprocs, myrank, nthreads = 1;
    #ifdef _OPENMP
    int provided, flag, claimed;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided );
    MPI_Is_thread_main( &flag );
    if (!flag)
        SpParHelper::Print("This thread called init_thread but "
                            "Is_thread_main gave false\n");
    MPI_Query_thread( &claimed );
    if (claimed != provided)
        SpParHelper::Print("Query thread gave different thread "
                            "level than requested\n");
    nthreads = omp_get_max_threads();
    #else
    MPI_Init(&argc, &argv);
    #endif

    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    int k = atoi(argv[2]);
    int nruns = atoi(argv[3]);
    int alg = atoi(argv[4]);

    shared_ptr<CommGrid> fullWorld;
    fullWorld.reset(new CommGrid(MPI_COMM_WORLD, 0, 0));

    if (myrank == 0)
    {
        if (alg == 0) printf("Algorithm: stationary-A 1.5D, ");
        if (alg == 1) printf("Algorithm: stationary-A   2D, ");
        if (alg == 2) printf("Algorithm: stationary-C 1.5D, ");
        printf("proc grid: %d x %d, nthreads: %d\n", fullWorld->GetGridRows(), fullWorld->GetGridCols(), nthreads);
    }

	auto t_beg = std::chrono::high_resolution_clock::now();
    SpParMat<TEST_IDX_T, TEST_NNZ_T, SpCCols<TEST_IDX_T, TEST_NNZ_T> > A(fullWorld);
    A.ParallelReadMM(string(argv[1]), 1, maximum<TEST_NNZ_T>());
	auto t_end = std::chrono::high_resolution_clock::now();
	if (myrank == 0)
    {
        std::cout << "ParallelReadMM() time: " <<
            static_cast<std::chrono::duration<double> >(t_end-t_beg).count() << std::endl;        
    }

    TEST_IDX_T nr = A.getnrow(), nc = A.getncol(), nnz = A.getnnz();
    TEST_NNZ_T imb = A.LoadImbalance();
    if (myrank == 0) printf("Matrix A size: %ld x %ld, nnz: %ld, load imb: %f\n", nr, nc, nnz, imb);

    TEST_IDX_T nnz_loc = A.seqptr()->getnnz();
    TEST_IDX_T nnzs[2];
    MPI_Datatype IDX_dtype = sizeof(TEST_IDX_T) == 4 ? MPI_INT : MPI_LONG_LONG_INT;
    MPI_Reduce(&nnz_loc, &nnzs[0], 1, IDX_dtype, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&nnz_loc, &nnzs[1], 1, IDX_dtype, MPI_MAX, 0, MPI_COMM_WORLD);
    if (myrank == 0) printf("Min/Max nnz per proc: %ld/%ld\n", nnzs[0], nnzs[1]);

    DnParMat<TEST_IDX_T, TEST_NNZ_T> X12(fullWorld, A.getncol(), k, 1.0);
    FullyDistDMat<TEST_IDX_T, TEST_NNZ_T> X0(fullWorld, A.getncol(), k, 1.0);
    typedef PlusTimesSRing<TEST_NNZ_T, TEST_NNZ_T> PTSR;

    spmm_stats stats = {0};
    double t = 0.0;
    for (int i = 0; i < nruns; ++i)
    {
        t_beg = std::chrono::high_resolution_clock::now();
        if (alg == 0) auto Y = SpMM_sA_CPU<PTSR>(A, X0, stats);
        if (alg == 1) auto Y = SpMM_sA_2D_CPU<PTSR>(A, X12, stats);
        if (alg == 2) auto Y = SpMM_sC_CPU<PTSR>(A, X12, stats);
        t_end = std::chrono::high_resolution_clock::now();
        auto t = static_cast<std::chrono::duration<double> >(t_end-t_beg).count();
        if (myrank == 0) printf("%.2f s\n", t);
    }
    print_spmm_stats(stats, nruns);
    }  // Allow objects to be destroyed before MPI_Finalize()
    
    MPI_Finalize();    
    return 0;
}
