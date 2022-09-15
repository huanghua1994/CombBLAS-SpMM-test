#include <mpi.h>
#include <sys/time.h>
#include <iostream>
#include <functional>
#include <algorithm>
#include <vector>
#include <sstream>
#include <cstdlib>
#include "CombBLAS/CombBLAS.h"
#include "CombBLAS/CommGrid3D.h"
#include "CombBLAS/ParFriends.h"
#include "../CC.h"
#include "../WriteMCLClusters.h"

using namespace std;
using namespace combblas;

#define EPS 0.0001

#ifdef _OPENMP
int cblas_splits = omp_get_max_threads();
#else
int cblas_splits = 1;
#endif

typedef struct
{
    //Input/Output file
    string ifilename;
    string dirname; // Patch to dump per iteration matrix into the directory
    bool isInputMM;
    int base; // only useful for matrix market files
    
    string ofilename;
    
    //Preprocessing
    int randpermute;
    bool remove_isolated;

    //inflation
    double inflation;
    
    //pruning
    double prunelimit;
    int64_t select;
    int64_t recover_num;
    double recover_pct;
    int kselectVersion; // 0: adapt based on k, 1: kselect1, 2: kselect2
    bool preprune;
    
    //HipMCL optimization
    int phases;
    int perProcessMem;
    bool isDoublePrecision; // true: double, false: float
    bool is64bInt; // true: int64_t for local indexing, false: int32_t (for local indexing)
    int layers; // Number of layers to use in communication avoiding SpGEMM. 
    int compute;
    int maxIter;
    
    //debugging
    bool show;
    
    
}HipMCLParam;

void InitParam(HipMCLParam & param)
{
    //Input/Output file
    param.ifilename = "";
    param.dirname = "";
    param.isInputMM = false;
    param.ofilename = "mclinc";
    param.base = 1;
    
    //Preprocessing
    // mcl removes isolated vertices by default,
    // we don't do this because it will create different ordering of vertices!
    param.remove_isolated = false;
    param.randpermute = 0;
    
    //inflation
    param.inflation = 2.0;
    
    //pruning
    param.prunelimit = 1.0/10000.0;
    param.select = 1100;
    param.recover_num = 1400;
    param.recover_pct = .9; // we allow both 90 or .9 as input. Internally, we keep it 0.9
    param.kselectVersion = 1;
    param.preprune = false;
    
    //HipMCL optimization
    param.layers = 1;
    param.compute = 1; // 1 means hash-based computation, 2 means heap-based computation
    param.phases = 1;
    param.perProcessMem = 0;
    param.isDoublePrecision = true;
    param.is64bInt = true;
    param.maxIter = 1000; // No limit on number of iterations 
    
    //debugging
    param.show = false;
}



// base: base of items
// clusters are always numbered 0-based
template <typename IT, typename NT, typename DER>
FullyDistVec<IT, IT> Interpret(SpParMat<IT,NT,DER> & A)
{
    IT nCC;
    // A is a directed graph
    // symmetricize A
    
    SpParMat<IT,NT,DER> AT = A;
    AT.Transpose();
    A += AT;
    SpParHelper::Print("Finding connected components....\n");
    
    FullyDistVec<IT, IT> cclabels = CC(A, nCC);
    return cclabels;
}


template <typename IT, typename NT, typename DER>
void MakeColStochastic(SpParMat<IT,NT,DER> & A)
{
    FullyDistVec<IT, NT> colsums = A.Reduce(Column, plus<NT>(), 0.0);
    colsums.Apply(safemultinv<NT>());
    A.DimApply(Column, colsums, multiplies<NT>());    // scale each "Column" with the given vector
}

template <typename IT, typename NT, typename DER>
void MakeColStochastic3D(SpParMat3D<IT,NT,DER> & A3D)
{
    //SpParMat<IT, NT, DER> * ALayer = A3D.GetLayerMat();
    std::shared_ptr< SpParMat<IT, NT, DER> > ALayer = A3D.GetLayerMat();
    FullyDistVec<IT, NT> colsums = ALayer->Reduce(Column, plus<NT>(), 0.0);
    colsums.Apply(safemultinv<NT>());
    ALayer->DimApply(Column, colsums, multiplies<NT>());    // scale each "Column" with the given vector
}

template <typename IT, typename NT, typename DER>
NT Chaos(SpParMat<IT,NT,DER> & A)
{
    // sums of squares of columns
    FullyDistVec<IT, NT> colssqs = A.Reduce(Column, plus<NT>(), 0.0, bind2nd(exponentiate(), 2));
    // Matrix entries are non-negative, so max() can use zero as identity
    FullyDistVec<IT, NT> colmaxs = A.Reduce(Column, maximum<NT>(), 0.0);
    colmaxs -= colssqs;
    
    // multiplu by number of nonzeros in each column
    FullyDistVec<IT, NT> nnzPerColumn = A.Reduce(Column, plus<NT>(), 0.0, [](NT val){return 1.0;});
    colmaxs.EWiseApply(nnzPerColumn, multiplies<NT>());
    
    return colmaxs.Reduce(maximum<NT>(), 0.0);
}

template <typename IT, typename NT, typename DER>
NT Chaos3D(SpParMat3D<IT,NT,DER> & A3D)
{
    //SpParMat<IT, NT, DER> * ALayer = A3D.GetLayerMat();
    std::shared_ptr< SpParMat<IT, NT, DER> > ALayer = A3D.GetLayerMat();

    // sums of squares of columns
    FullyDistVec<IT, NT> colssqs = ALayer->Reduce(Column, plus<NT>(), 0.0, bind2nd(exponentiate(), 2));
    // Matrix entries are non-negative, so max() can use zero as identity
    FullyDistVec<IT, NT> colmaxs = ALayer->Reduce(Column, maximum<NT>(), 0.0);
    colmaxs -= colssqs;

    // multiply by number of nonzeros in each column
    FullyDistVec<IT, NT> nnzPerColumn = ALayer->Reduce(Column, plus<NT>(), 0.0, [](NT val){return 1.0;});
    colmaxs.EWiseApply(nnzPerColumn, multiplies<NT>());
    
    NT layerChaos = colmaxs.Reduce(maximum<NT>(), 0.0);

    NT totalChaos = 0.0;
    MPI_Allreduce( &layerChaos, &totalChaos, 1, MPIType<NT>(), MPI_MAX, A3D.getcommgrid3D()->GetFiberWorld());
    return totalChaos;
}

template <typename IT, typename NT, typename DER>
void Inflate(SpParMat<IT,NT,DER> & A, double power)
{
    A.Apply(bind2nd(exponentiate(), power));
}

template <typename IT, typename NT, typename DER>
void Inflate3D(SpParMat3D<IT,NT,DER> & A3D, double power)
{
    //SpParMat<IT, NT, DER> * ALayer = A3D.GetLayerMat();
    std::shared_ptr< SpParMat<IT, NT, DER> > ALayer = A3D.GetLayerMat();
    ALayer->Apply(bind2nd(exponentiate(), power));
}

// default adjustloop setting
// 1. Remove loops
// 2. set loops to max of all arc weights
template <typename IT, typename NT, typename DER>
void AdjustLoops(SpParMat<IT,NT,DER> & A)
{

    A.RemoveLoops();
    FullyDistVec<IT, NT> colmaxs = A.Reduce(Column, maximum<NT>(), numeric_limits<NT>::min());
    A.Apply([](NT val){return val==numeric_limits<NT>::min() ? 1.0 : val;}); // for isolated vertices
    A.AddLoops(colmaxs);
    ostringstream outs;
    outs << "Adjusting loops" << endl;
    SpParHelper::Print(outs.str());
}

template <typename IT, typename NT, typename DER>
void RemoveIsolated(SpParMat<IT,NT,DER> & A, HipMCLParam & param)
{
    ostringstream outs;
    FullyDistVec<IT, NT> ColSums = A.Reduce(Column, plus<NT>(), 0.0);
    FullyDistVec<IT, IT> nonisov = ColSums.FindInds(bind2nd(greater<NT>(), 0));
    IT numIsolated = A.getnrow() - nonisov.TotalLength();
    outs << "Number of isolated vertices: " << numIsolated << endl;
    SpParHelper::Print(outs.str());
    
    A(nonisov, nonisov, true);
    SpParHelper::Print("Removed isolated vertices.\n");
    if(param.show)
    {
        A.PrintInfo();
    }
    
}

//TODO: handle reordered cluster ids
template <typename IT, typename NT, typename DER>
void RandPermute(SpParMat<IT,NT,DER> & A, HipMCLParam & param)
{
    // randomly permute for load balance
    if(A.getnrow() == A.getncol())
    {
        FullyDistVec<IT, IT> p( A.getcommgrid());
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

template <typename IT, typename NT, typename DER>
FullyDistVec<IT, IT> HipMCL(SpParMat<IT,NT,DER> & A, HipMCLParam & param)
{
    if(param.remove_isolated)
        RemoveIsolated(A, param);
    
    if(param.randpermute)
        RandPermute(A, param);

    // Adjust self loops
    AdjustLoops(A);

    // Make stochastic
    MakeColStochastic(A);
    //SpParHelper::Print("Made stochastic\n");

    //if(!param.dirname.empty()) {
        //std::string fname = param.dirname + "\/" + std::to_string(0);
        //A.ParallelWriteMM(fname, true);
    //}
    
    
    IT nnz = A.getnnz();
    IT nv = A.getnrow();
    IT avgDegree = nnz/nv;
    if(avgDegree > std::max(param.select, param.recover_num))
    {
        SpParHelper::Print("Average degree of the input graph is greater than max{S,R}.\n");
        param.preprune = true;
    }
    if(param.preprune)
    {
        SpParHelper::Print("Applying the prune/select/recovery logic before the first iteration\n\n");
        MCLPruneRecoverySelect(A, (NT)param.prunelimit, (IT)param.select, (IT)param.recover_num, (NT)param.recover_pct, param.kselectVersion);
    }

    if(param.show)
    {
        A.PrintInfo();
    }
    

    // chaos doesn't make sense for non-stochastic matrices
    // it is in the range {0,1} for stochastic matrices
    NT chaos = 1;
    int it=1;
    double tInflate = 0;
    double tExpand = 0;
    typedef PlusTimesSRing<NT, NT> PTFF;
	SpParMat3D<IT,NT,DER> A3D_cs(param.layers);
	if(param.layers > 1) {
    	SpParMat<IT,NT,DER> A2D_cs = SpParMat<IT, NT, DER>(A);
		A3D_cs = SpParMat3D<IT,NT,DER>(A2D_cs, param.layers, true, false);    // Non-special column split
	}
    // while there is an epsilon improvement
    while( chaos > EPS)
    {
		SpParMat3D<IT,NT,DER> A3D_rs(param.layers);
		if(param.layers > 1) {
			A3D_rs  = SpParMat3D<IT,NT,DER>(A3D_cs, false); // Create new rowsplit copy of matrix from colsplit copy
		}

        double t1 = MPI_Wtime();
        //A.Square<PTFF>() ;        // expand
		if(param.layers == 1){
			A = MemEfficientSpGEMM<PTFF, NT, DER>(A, A, param.phases, param.prunelimit, (IT)param.select, (IT)param.recover_num, param.recover_pct, param.kselectVersion, param.compute, param.perProcessMem);
		}
		else{
			A3D_cs = MemEfficientSpGEMM3D<PTFF, NT, DER, IT, NT, NT, DER, DER >(
                A3D_cs, A3D_rs, 
                param.phases, 
                param.prunelimit, 
                (IT)param.select, 
                (IT)param.recover_num, 
                param.recover_pct, 
                param.kselectVersion,
                param.compute,
                param.perProcessMem
         	);
		}
        
		if(param.layers == 1){
			MakeColStochastic(A);
		}
		else{
            MakeColStochastic3D(A3D_cs);
		}
        tExpand += (MPI_Wtime() - t1);
        
        //if(param.show)
        //{
            //SpParHelper::Print("After expansion\n");
            //A.PrintInfo();
        //}
        if(param.layers == 1) chaos = Chaos(A);
        else chaos = Chaos3D(A3D_cs);
        
        //double tInflate1 = MPI_Wtime();
        if (param.layers == 1) Inflate(A, param.inflation);
        else Inflate3D(A3D_cs, param.inflation);

        if(param.layers == 1) MakeColStochastic(A);
        else MakeColStochastic3D(A3D_cs);

        //tInflate += (MPI_Wtime() - tInflate1);
        
        //if(param.show)
        //{
            //SpParHelper::Print("After inflation\n");
            //A.PrintInfo();
        //}
        
        //if(!param.dirname.empty()) {
            //std::string fname = param.dirname + "\/" + std::to_string(it);
            //A.ParallelWriteMM(fname, true);
        //}
        
        double newbalance = A.LoadImbalance();
        double t3=MPI_Wtime();
        stringstream s;
        s << "Iteration# "  << setw(3) << it << " : "  << " chaos: " << setprecision(3) << chaos << "  load-balance: "<< newbalance << " Time: " << (t3-t1) << endl;
        //SpParHelper::Print(s.str());
        it++;
        
        
        
    }
    
    
#ifdef TIMING    
    double tcc1 = MPI_Wtime();
#endif
    
    // bool can not be used because
    // bool does not work in A.AddLoops(1) used in LACC: can not create a fullydist vector with Bool
    // SpParMat<IT,NT,DER> A does not work because int64_t and float promote trait not defined
    // hence, we are forcing this with IT and double
    SpParMat<IT,double, SpDCCols < IT, double >> ADouble(MPI_COMM_WORLD);
    if(param.layers == 1) ADouble = A;
    else ADouble = A3D_cs.Convert2D();
    FullyDistVec<IT, IT> cclabels = Interpret(ADouble);
    
    return cclabels;
}

template <typename IT, typename NT, typename DER>
FullyDistVec<IT, IT> IncrementalMCL(SpParMat<IT,NT,DER> & A, HipMCLParam & param)
{
    if(param.remove_isolated)
        RemoveIsolated(A, param);
    
    if(param.randpermute)
        RandPermute(A, param);

    // Adjust self loops
    AdjustLoops(A);

    // Make stochastic
    MakeColStochastic(A);
    //SpParHelper::Print("Made stochastic\n");

    //if(!param.dirname.empty()) {
        //std::string fname = param.dirname + "\/" + std::to_string(0);
        //A.ParallelWriteMM(fname, true);
    //}
    
    
    IT nnz = A.getnnz();
    IT nv = A.getnrow();
    IT avgDegree = nnz/nv;
    if(avgDegree > std::max(param.select, param.recover_num))
    {
        SpParHelper::Print("Average degree of the input graph is greater than max{S,R}.\n");
        param.preprune = true;
    }
    if(param.preprune)
    {
        SpParHelper::Print("Applying the prune/select/recovery logic before the first iteration\n\n");
        MCLPruneRecoverySelect(A, (NT)param.prunelimit, (IT)param.select, (IT)param.recover_num, (NT)param.recover_pct, param.kselectVersion);
    }

    if(param.show)
    {
        A.PrintInfo();
    }
    

    // chaos doesn't make sense for non-stochastic matrices
    // it is in the range {0,1} for stochastic matrices
    NT chaos = 1;
    int it=1;
    double tInflate = 0;
    double tExpand = 0;
    typedef PlusTimesSRing<NT, NT> PTFF;
	SpParMat3D<IT,NT,DER> A3D_cs(param.layers);
	if(param.layers > 1) {
    	SpParMat<IT,NT,DER> A2D_cs = SpParMat<IT, NT, DER>(A);
		A3D_cs = SpParMat3D<IT,NT,DER>(A2D_cs, param.layers, true, false);    // Non-special column split
	}
    // while there is an epsilon improvement
    while( (chaos > EPS) && (it <= param.maxIter) )
    {
		SpParMat3D<IT,NT,DER> A3D_rs(param.layers);
		if(param.layers > 1) {
			A3D_rs  = SpParMat3D<IT,NT,DER>(A3D_cs, false); // Create new rowsplit copy of matrix from colsplit copy
		}

        double t1 = MPI_Wtime();
        //A.Square<PTFF>() ;        // expand
		if(param.layers == 1){
			A = MemEfficientSpGEMM<PTFF, NT, DER>(A, A, param.phases, param.prunelimit, (IT)param.select, (IT)param.recover_num, param.recover_pct, param.kselectVersion, param.compute, param.perProcessMem);
		}
		else{
			A3D_cs = MemEfficientSpGEMM3D<PTFF, NT, DER, IT, NT, NT, DER, DER >(
                A3D_cs, A3D_rs, 
                param.phases, 
                param.prunelimit, 
                (IT)param.select, 
                (IT)param.recover_num, 
                param.recover_pct, 
                param.kselectVersion,
                param.compute,
                param.perProcessMem
         	);
		}
        
		if(param.layers == 1){
			MakeColStochastic(A);
		}
		else{
            MakeColStochastic3D(A3D_cs);
		}
        tExpand += (MPI_Wtime() - t1);
        
        //if(param.show)
        //{
            //SpParHelper::Print("After expansion\n");
            //A.PrintInfo();
        //}
        if(param.layers == 1) chaos = Chaos(A);
        else chaos = Chaos3D(A3D_cs);
        
        //double tInflate1 = MPI_Wtime();
        if (param.layers == 1) Inflate(A, param.inflation);
        else Inflate3D(A3D_cs, param.inflation);

        if(param.layers == 1) MakeColStochastic(A);
        else MakeColStochastic3D(A3D_cs);

        //tInflate += (MPI_Wtime() - tInflate1);
        
        //if(param.show)
        //{
            //SpParHelper::Print("After inflation\n");
            //A.PrintInfo();
        //}
        
        //if(!param.dirname.empty()) {
            //std::string fname = param.dirname + "\/" + std::to_string(it);
            //A.ParallelWriteMM(fname, true);
        //}
        
        double newbalance = A.LoadImbalance();
        double t3=MPI_Wtime();
        stringstream s;
        s << "Iteration# "  << setw(3) << it << " : "  << " chaos: " << setprecision(3) << chaos << "  load-balance: "<< newbalance << " Time: " << (t3-t1) << endl;
        SpParHelper::Print(s.str());
        A.PrintInfo();
        it++;
        
        
        
    }
    
    
#ifdef TIMING    
    double tcc1 = MPI_Wtime();
#endif
    
    // bool can not be used because
    // bool does not work in A.AddLoops(1) used in LACC: can not create a fullydist vector with Bool
    // SpParMat<IT,NT,DER> A does not work because int64_t and float promote trait not defined
    // hence, we are forcing this with IT and double
    SpParMat<IT,double, SpDCCols < IT, double >> ADouble(MPI_COMM_WORLD);
    if(param.layers == 1) ADouble = A;
    else ADouble = A3D_cs.Convert2D();
    FullyDistVec<IT, IT> cclabels = Interpret(ADouble);
    
    return cclabels;
}

// Given an adjacency matrix, and cluster assignment vector, removes inter cluster edges
template <class IT, class NT, class DER>
void RemoveInterClusterEdges(SpParMat<IT, NT, DER>& M, FullyDistVec<IT, IT>& C){
    FullyDistVec<IT, NT> Ctemp(C); // To convert value types from IT to NT. Because DimApply and PruneColumn requires that.
    SpParMat<IT, NT, DER> Mask(M);
    Mask.DimApply(Row, Ctemp, [](NT mv, NT vv){return vv;});
    Mask.PruneColumn(Ctemp, [](NT mv, NT vv){return static_cast<NT>(vv == mv);}, true);

    //Mask.PrintInfo();
    M.SetDifference(Mask);
}

// Given an adjacency matrix, and cluster assignment vector, removes inter cluster edges
template <class IT, class NT, class DER>
void ProcessNewEdges(SpParMat<IT, NT, DER>& M){
    return;
}

int main(int argc, char* argv[])
{
    int nprocs, myrank, nthreads = 1;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
#ifdef THREADED
#pragma omp parallel
    {
        nthreads = omp_get_num_threads();
    }
#endif
    if(myrank == 0)
    {
        cout << "Process Grid (p x p x t): " << sqrt(nprocs) << " x " << sqrt(nprocs) << " x " << nthreads << endl;
    }
    if(argc < 7)
    {
        if(myrank == 0)
        {
            cout << "Usage: ./inc -I <mm|triples> -M <MATRIX_FILENAME> -N <NUMBER OF SPLITS>\n";
            cout << "-I <INPUT FILE TYPE> (mm: matrix market, triples: (vtx1, vtx2, edge_weight) triples, default: mm)\n";
            cout << "-M <MATRIX FILE NAME>\n";
            cout << "-base <BASE OF MATRIX MARKET> (default:1)\n";
            cout << "-N <NUMBER OF SPLITS>\n";
        }
        MPI_Finalize();
        return -1;
    }
    else{
        string Mname = "";
        int nSplit = 2;
        int base = 1;
        bool isMatrixMarket = true;
        
        for (int i = 1; i < argc; i++)
        {
            if (strcmp(argv[i],"-I")==0)
            {
                string mfiletype = string(argv[i+1]);
                if(mfiletype == "triples") isMatrixMarket = false;
            }
            else if (strcmp(argv[i],"-M")==0)
            {
                Mname = string(argv[i+1]);
                if(myrank == 0) printf("Matrix filename: %s\n",Mname.c_str());
            }
            else if (strcmp(argv[i],"-base")==0)
            {
                base = atoi(argv[i + 1]);
                if(myrank == 0) printf("Base of MM (1 or 0):%d\n",base);
            }
            else if (strcmp(argv[i],"-N")==0)
            {
                nSplit = atoi(argv[i+1]);
                if(myrank == 0) printf("Number of splits: %d\n", nSplit);
            }
        }

        shared_ptr<CommGrid> fullWorld;
        fullWorld.reset( new CommGrid(MPI_COMM_WORLD, 0, 0) );

        typedef int64_t IT;
        typedef double NT;
        typedef SpDCCols < int64_t, double > DER;
        typedef PlusTimesSRing<double, double> PTFF;
        typedef PlusTimesSRing<bool, double> PTBOOLNT;
        typedef PlusTimesSRing<double, bool> PTNTBOOL;
        
        double t0, t1, t2, t3, t4, t5;

        SpParMat<IT, NT, DER> M(fullWorld);

        if(isMatrixMarket)
            M.ParallelReadMM(Mname, base, maximum<double>());
        else
            M.ReadGeneralizedTuples(Mname,  maximum<double>());
        M.PrintInfo();
        
        /*
         * Prepare split:
         * Prepare a vector of FullyDistVec where each element would represent list of vertices in a particular incremental batch
         * */
        std::mt19937 rng;
        rng.seed(myrank);
        std::uniform_int_distribution<int64_t> udist(0, 9999);

        IT gnRow = M.getnrow();
        IT nRowPerProc = gnRow / nprocs;
        IT lRowStart = myrank * nRowPerProc;
        IT lRowEnd = (myrank == nprocs - 1) ? gnRow : (myrank + 1) * nRowPerProc;

        std::vector < std::vector < IT > > lvList(nSplit);
        std::vector < std::vector < std::array<char, MAXVERTNAME> > > lvListLabels(nSplit); // MAXVERTNAME is 64, defined in SpDefs
                                                                                           
        for (IT r = lRowStart; r < lRowEnd; r++) {
            IT randomNum = udist(rng);
            IT s = randomNum % nSplit;
            lvList[s].push_back(r);
            
            // Convert the integer vertex id to label as string
            std::string labelStr = std::to_string(r); 
            // Make a std::array of char with the label
            std::array<char, MAXVERTNAME> labelArr = {};
            for ( IT i = 0; (i < labelStr.length()) && (i < MAXVERTNAME); i++){
                labelArr[i] = labelStr[i]; 
            }
            lvListLabels[s].push_back( labelArr );
        }

        std::vector < FullyDistVec<IT,IT>* > dvList;
        std::vector < FullyDistVec<IT, std::array<char, MAXVERTNAME> >* > dvListLabels;
        for (int s = 0; s < nSplit; s++){
            dvList.push_back(new FullyDistVec<IT, IT>(lvList[s], fullWorld));
            dvListLabels.push_back(new FullyDistVec<IT, std::array<char, MAXVERTNAME> >(lvListLabels[s], fullWorld));
        }

        SpParMat<IT, NT, DER> M11(fullWorld);
        SpParMat<IT, NT, DER> M12(fullWorld);
        SpParMat<IT, NT, DER> M21(fullWorld);
        SpParMat<IT, NT, DER> M22(fullWorld);

        SpParMat<IT, NT, DER> Minc(fullWorld);
        SpParMat<IT, NT, DER> MincFake(fullWorld);
        SpParMat<IT, NT, DER> Mall(fullWorld);

        SpParMat<IT, NT, DER> MincTemp(fullWorld);
        SpParMat<IT, NT, DER> MincFakeTemp(fullWorld);
        SpParMat<IT, NT, DER> MallTemp(fullWorld);

        FullyDistVec<IT, IT> Cinc(fullWorld);
        FullyDistVec<IT, IT> CincFake(fullWorld);
        FullyDistVec<IT, IT> Call(fullWorld);

        HipMCLParam mclParam;
        InitParam(mclParam);


        std::string incFileName = Mname + std::string(".") + std::to_string(nSplit) + std::string(".inc");
        std::string incFakeFileName = Mname + std::string(".") + std::to_string(nSplit) + std::string(".incfake");
        std::string fullFileName = Mname + std::string(".") + std::to_string(nSplit) + std::string(".full");

        FullyDistVec<IT, IT> prevVertices(*(dvList[0])); // Create a distributed vector to keep track of the vertices being considered at each incremental step
        FullyDistVec<IT, std::array<char, MAXVERTNAME>> prevVerticesLabels(*(dvListLabels[0])); // Create a distributed vector to keep track of the vertex labels being considered at each incremental step
                                                                                                //
        M11 = M.SubsRef_SR < PTNTBOOL, PTBOOLNT> (prevVertices, prevVertices, false);
        Mall = M.SubsRef_SR < PTNTBOOL, PTBOOLNT> (prevVertices, prevVertices, false);

        MallTemp = Mall;
        Call = HipMCL(MallTemp, mclParam);
        //Mall = MallTemp;
        RemoveInterClusterEdges(Mall, Call);

        HipMCLParam incParam;
        InitParam(incParam);
		incParam.maxIter = 5;

        Minc = M11;
        Cinc = IncrementalMCL(Minc, incParam);
		M11 = Minc;
        RemoveInterClusterEdges(M11, Cinc);

        //WriteMCLClusters(incFileName + std::string(".") + std::to_string(0), C11, prevVerticesLabels);
        WriteMCLClusters(incFileName + std::string(".") + std::to_string(0), Cinc, base);
        WriteMCLClusters(incFakeFileName + std::string(".") + std::to_string(0), Cinc, base);
        WriteMCLClusters(fullFileName + std::string(".") + std::to_string(0), Call, base);
        
        std::vector< FullyDistVec<IT, IT> > toConcatenate(2, FullyDistVec<IT, IT>(fullWorld));
        std::vector< FullyDistVec<IT, std::array<char, MAXVERTNAME> > > toConcatenateLabels(2, FullyDistVec<IT, std::array<char, MAXVERTNAME> >(fullWorld));

        for(int s = 1; s < nSplit; s++){
			if(s == nSplit - 1){
				incParam.maxIter = 1000;
			}
            MPI_Barrier(MPI_COMM_WORLD);
            if(myrank == 0) printf("***\n", s);
            if(myrank == 0) printf("Processing %dth split\n", s);
            if(myrank == 0) printf("***\n", s);

            t0 = MPI_Wtime();

            FullyDistVec<IT, IT> newVertices(*(dvList[s]));
            FullyDistVec<IT, std::array<char, MAXVERTNAME> > newVerticesLabels(*(dvListLabels[s]));

            toConcatenate[0] = prevVertices;
            toConcatenate[1] = newVertices;
            FullyDistVec<IT, IT> allVertices = Concatenate(toConcatenate);

            toConcatenateLabels[0] = prevVerticesLabels;
            toConcatenateLabels[1] = newVerticesLabels;
            FullyDistVec<IT, std::array<char, MAXVERTNAME> > allVerticesLabels = Concatenate(toConcatenateLabels);

            t1 = MPI_Wtime();
            if(myrank == 0) printf("[Step: %d]\tTime to prepate vertex lists: %lf\n", s, t1 - t0);

            t2 = MPI_Wtime();

            t0 = MPI_Wtime();
            M12 = M.SubsRef_SR <PTNTBOOL, PTBOOLNT> (prevVertices, newVertices, false);
            t1 = MPI_Wtime();
            if(myrank == 0) printf("[Step: %d]\tTime to extract M12: %lf\n", s, t1 - t0);

            t0 = MPI_Wtime();
            M21 = M.SubsRef_SR <PTNTBOOL, PTBOOLNT> (newVertices, prevVertices, false);
            t1 = MPI_Wtime();
            if(myrank == 0) printf("[Step: %d]\tTime to extract M21: %lf\n", s, t1 - t0);

            t0 = MPI_Wtime();
            M22 = M.SubsRef_SR <PTNTBOOL, PTBOOLNT> (newVertices, newVertices, false); // Get subgraph induced by newly added vertices in current step
            t1 = MPI_Wtime();
            if(myrank == 0) printf("[Step: %d]\tTime to extract M22: %lf\n", s, t1 - t0);

            t3 = MPI_Wtime();
            if(myrank == 0) printf("[Step: %d]\tTime extract subgraphs: %lf\n", s, t3 - t2);

            ProcessNewEdges(M12);
            ProcessNewEdges(M21);
            
            SpParMat<IT, NT, DER> M22Temp = M22;
            t0 = MPI_Wtime();
            FullyDistVec<IT, IT> C22 = IncrementalMCL(M22Temp, incParam); // Cluster M22
            t1 = MPI_Wtime();
            if(myrank == 0) printf("[Step: %d]\tTime to find clusters in M22: %lf\n", s, t1 - t0);
			M22 = M22Temp;

            t0 = MPI_Wtime();
            RemoveInterClusterEdges(M22, C22); //Summarize M22
            t1 = MPI_Wtime();
            if(myrank == 0) printf("[Step: %d]\tTime to remove inter-cluster edges of M22: %lf\n", s, t1 - t0);

            t2 = MPI_Wtime();
            t0 = MPI_Wtime();
            FullyDistVec<IT, IT> prevVerticesRemapped( fullWorld );
            prevVerticesRemapped.iota(M11.getnrow(), 0);
            FullyDistVec<IT, IT> newVerticesRemapped( fullWorld );
            newVerticesRemapped.iota(M22.getnrow(), M11.getnrow());
            t1 = MPI_Wtime();
            if(myrank == 0) printf("[Step: %d]\tTime to prepare vertex remapping: %lf\n", s, t1 - t0);

            t0 = MPI_Wtime();
            Minc = SpParMat<IT,NT,DER>(M11.getnrow() + M22.getnrow(), 
                    M11.getnrow() + M22.getnrow(), 
                    FullyDistVec<IT,IT>(fullWorld), 
                    FullyDistVec<IT,IT>(fullWorld), 
                    FullyDistVec<IT,IT>(fullWorld), true); 
            t1 = MPI_Wtime();
            if(myrank == 0) printf("[Step: %d]\tTime to prepare empty Minc: %lf\n", s, t1 - t0);

            t0 = MPI_Wtime();
            Minc.SpAsgn(prevVerticesRemapped, prevVerticesRemapped, M11);
            t1 = MPI_Wtime();
            if(myrank == 0) printf("[Step: %d]\tTime to perform SpAssign of M11: %lf\n", s, t1 - t0);

            t0 = MPI_Wtime();
            Minc.SpAsgn(prevVerticesRemapped, newVerticesRemapped, M12);
            t1 = MPI_Wtime();
            if(myrank == 0) printf("[Step: %d]\tTime to perform SpAssign of M12: %lf\n", s, t1 - t0);

            t0 = MPI_Wtime();
            Minc.SpAsgn(newVerticesRemapped, prevVerticesRemapped, M21);
            t1 = MPI_Wtime();
            if(myrank == 0) printf("[Step: %d]\tTime to perform SpAssign of M21: %lf\n", s, t1 - t0);

            t0 = MPI_Wtime();
            Minc.SpAsgn(newVerticesRemapped, newVerticesRemapped, M22);
            t1 = MPI_Wtime();
            if(myrank == 0) printf("[Step: %d]\tTime to perform SpAssign of M22: %lf\n", s, t1 - t0);

            MPI_Barrier(MPI_COMM_WORLD);
            if(myrank == 0) printf("Minc prepared\n");

            t3 = MPI_Wtime();
            if(myrank == 0) printf("[Step: %d]\tTime to prepare Minc: %lf\n", s, t3 - t2);
            Minc.PrintInfo();
            float MincLB = Minc.LoadImbalance();
            if(myrank == 0) printf("Minc.LoadImbalance() = %f\n", MincLB);
            //M11.PrintInfo();
            //M12.PrintInfo();
            //M21.PrintInfo();
            //M22.PrintInfo();
            
            MincTemp = Minc;
            t0 = MPI_Wtime();
			Cinc = IncrementalMCL(MincTemp, incParam);
            MPI_Barrier(MPI_COMM_WORLD);
            if(myrank == 0) printf("Ran IncrementalMCL on Minc\n");
            t1 = MPI_Wtime();
            if(myrank == 0) printf("[Step: %d]\tTime to find clusters in Minc: %lf\n", s, t1 - t0);
			Minc = MincTemp;

            t2 = MPI_Wtime();
            //MincFake = SpParMat<IT, NT, DER>(Minc);
            MincFake = SpParMat<IT,NT,DER>(Mall.getnrow() + M22.getnrow(), 
                    Mall.getnrow() + M22.getnrow(), 
                    FullyDistVec<IT,IT>(fullWorld), 
                    FullyDistVec<IT,IT>(fullWorld), 
                    FullyDistVec<IT,IT>(fullWorld), true); 
            MincFake.SpAsgn(prevVerticesRemapped, prevVerticesRemapped, Mall);
            MincFake.SpAsgn(prevVerticesRemapped, newVerticesRemapped, M12);
            MincFake.SpAsgn(newVerticesRemapped, prevVerticesRemapped, M21);
            MincFake.SpAsgn(newVerticesRemapped, newVerticesRemapped, M22);

            MPI_Barrier(MPI_COMM_WORLD);
            if(myrank == 0) printf("MincFake prepared\n");

            t3 = MPI_Wtime();
            if(myrank == 0) printf("[Step: %d]\tTime to prepare MincFake: %lf\n", s, t3 - t2);

            MincFake.PrintInfo();
            float MincFakeLB = MincFake.LoadImbalance();
            if(myrank == 0) printf("MincFake.LoadImbalance() = %f\n", MincFakeLB);
            
            MincFakeTemp = MincFake;
            CincFake = IncrementalMCL(MincFakeTemp, incParam);
            MPI_Barrier(MPI_COMM_WORLD);
            if(myrank == 0) printf("Ran HipMCL on MincFake\n");
            //MincFake = MincFakeTemp;

            t0 = MPI_Wtime();
            Mall = M.SubsRef_SR <PTNTBOOL, PTBOOLNT> (allVertices, allVertices, false);
            MPI_Barrier(MPI_COMM_WORLD);
            if(myrank == 0) printf("Mall prepared\n");
            t1 = MPI_Wtime();
            if(myrank == 0) printf("[Step: %d]\tTime to prepare Mall: %lf\n", s, t1 - t0);
            Mall.PrintInfo();
            float MallLB = Mall.LoadImbalance();
            if(myrank == 0) printf("Mall.LoadImbalance() = %f\n", MallLB);
            
            MallTemp = Mall;
            t0 = MPI_Wtime();
            Call = HipMCL(MallTemp, mclParam);
            MPI_Barrier(MPI_COMM_WORLD);
            if(myrank == 0) printf("Ran HipMCL on Mall\n");
            t1 = MPI_Wtime();
            if(myrank == 0) printf("[Step: %d]\tTime to find clusters in Mall: %lf\n", s, t1 - t0);

            WriteMCLClusters(incFileName + std::string(".") + std::to_string(s), Cinc, base);
            WriteMCLClusters(incFakeFileName + std::string(".") + std::to_string(s), CincFake, base);
            WriteMCLClusters(fullFileName + std::string(".") + std::to_string(s), Call, base);
            
            t0 = MPI_Wtime();
			M11 = Minc;
            //M11 = Mall;
            RemoveInterClusterEdges(M11, Cinc);
            MPI_Barrier(MPI_COMM_WORLD);
            if(myrank == 0) printf("Prepared M11 for next step\n");
            t1 = MPI_Wtime();
            if(myrank == 0) printf("[Step: %d]\tTime to prepare M11 for next step: %lf\n", s, t1 - t0);

            M11.PrintInfo();

            RemoveInterClusterEdges(Mall, Call);
            MPI_Barrier(MPI_COMM_WORLD);
            if(myrank == 0) printf("Prepared M11Fake for next step\n");
            Mall.PrintInfo();

            prevVertices = FullyDistVec<IT, IT>(allVertices);
            prevVerticesLabels = FullyDistVec<IT, std::array<char, MAXVERTNAME> >(allVerticesLabels);
        }

        for(IT s = 0; s < dvList.size(); s++){
            delete dvList[s];
            delete dvListLabels[s];
        }

    }
    MPI_Finalize();
    return 0;
}
