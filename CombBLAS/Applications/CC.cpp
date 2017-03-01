/****************************************************************/
/* Parallel Combinatorial BLAS Library (for Graph Computations) */
/* version 1.6 -------------------------------------------------*/
/* date: 11/15/2016 --------------------------------------------*/
/* authors: Ariful Azad, Aydin Buluc, Adam Lugowski ------------*/
/****************************************************************/
/*
 Copyright (c) 2010-2016, The Regents of the University of California
 
 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:
 
 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.
 
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
 */

#include <mpi.h>

// These macros should be defined before stdint.h is included
#ifndef __STDC_CONSTANT_MACROS
#define __STDC_CONSTANT_MACROS
#endif
#ifndef __STDC_LIMIT_MACROS
#define __STDC_LIMIT_MACROS
#endif
#include <stdint.h>

#include <sys/time.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <ctime>
#include <cmath>
#include "../CombBLAS.h"
#include "CC.h"

using namespace std;

/**
 ** Connected components based on Awerbuch-Shiloach algorithm
 **/


class Dist
{
public:
    typedef SpDCCols < int64_t, double > DCCols;
    typedef SpParMat < int64_t, double, DCCols > MPI_DCCols;
};



int main(int argc, char* argv[])
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    if (provided < MPI_THREAD_SERIALIZED)
    {
        printf("ERROR: The MPI library does not have MPI_THREAD_SERIALIZED support\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    int nthreads = 1;
#ifdef THREADED
#pragma omp parallel
    {
        nthreads = omp_get_num_threads();
    }
#endif
    
    int nprocs, myrank;
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    if(myrank == 0)
    {
        cout << "Process Grid (p x p x t): " << sqrt(nprocs) << " x " << sqrt(nprocs) << " x " << nthreads << endl;
    }
    
    if(argc < 3)
    {
        if(myrank == 0)
        {
            cout << "Usage: ./cc -M <FILENAME_MATRIX_MARKET> (required)\n";
            cout << "-base <BASE OF MATRIX MARKET> (default:1)\n";
            cout << "-rand <RANDOMLY PERMUTE VERTICES> (default:0)\n";
            cout << "Example (0-indexed mtx with random permutation): ./cc -M input.mtx -base 0 -rand 1" << endl;
        }
        MPI_Finalize();
        return -1;
    }
    {
        string ifilename = "";
        int base = 1;
        int randpermute = 0;
        
        for (int i = 1; i < argc; i++)
        {
            if (strcmp(argv[i],"-M")==0)
            {
                ifilename = string(argv[i+1]);
                if(myrank == 0) printf("filename: %s",ifilename.c_str());
            }
            else if (strcmp(argv[i],"-base")==0)
            {
                base = atoi(argv[i + 1]);
                if(myrank == 0) printf("\nBase of MM (1 or 0):%d",base);
            }
            else if (strcmp(argv[i],"-rand")==0)
            {
                randpermute = atoi(argv[i + 1]);
                if(myrank == 0) printf("\nRandomly permute the matrix? (1 or 0):%d",randpermute);
            }
        }
        
        double tIO = MPI_Wtime();
        Dist::MPI_DCCols A;	// construct object
        A.ParallelReadMM(ifilename, base, maximum<bool>());	// if base=0, then it is implicitly converted to Boolean false
        //A.ReadDistribute(ifilename, 0);
        //A.PrintInfo();
        
        ostringstream outs;
        outs << "File Read time: " << MPI_Wtime() - tIO << endl;
        SpParHelper::Print(outs.str());
        
        if(randpermute)
        {
            // randomly permute for load balance
            if(A.getnrow() == A.getncol())
            {
                FullyDistVec<int64_t, int64_t> p( A.getcommgrid());
                p.iota(A.getnrow(), 0);
                p.RandPerm();
                (A)(p,p,true);// in-place permute to save memory
                SpParHelper::Print("Applied symmetric permutation.\n");
            }
            else
            {
                SpParHelper::Print("Rectangular matrix: Can not apply symmetric permutation.\n");
            }
        }
        
        FullyDistVec<int64_t,double> ColSums = A.Reduce(Column, plus<double>(), 0.0);
        FullyDistVec<int64_t, int64_t> nonisov = ColSums.FindInds(bind2nd(greater<double>(), 0));
        cout << "isolated: " << nonisov.TotalLength() << endl;
        
        float balance = A.LoadImbalance();
        int64_t nnz = A.getnnz();
        outs.str("");
        outs.clear();
        outs << "Load balance: " << balance << endl;
        outs << "Nonzeros: " << nnz << endl;
        SpParHelper::Print(outs.str());
        
        int64_t nCC = 0;
        FullyDistVec<int64_t, int64_t> cclabels = CC(A, nCC);
    }
    
    MPI_Finalize();
    return 0;
}
