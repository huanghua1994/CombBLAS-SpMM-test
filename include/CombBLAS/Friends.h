/****************************************************************/
/* Parallel Combinatorial BLAS Library (for Graph Computations) */
/* version 1.6 -------------------------------------------------*/
/* date: 6/15/2017 ---------------------------------------------*/
/* authors: Ariful Azad, Aydin Buluc  --------------------------*/
/****************************************************************/
/*
 Copyright (c) 2010-2017, The Regents of the University of California
 
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


#ifndef _FRIENDS_H_
#define _FRIENDS_H_

#include <iostream>
#include <chrono>
#include "SpMat.h"	// Best to include the base class first
#include "SpHelper.h"
#include "StackEntry.h"
#include "Isect.h"
#include "Deleter.h"
#include "SpImpl.h"
#include "SpParHelper.h"
#include "Compare.h"
#include "CombBLAS.h"
#include "PreAllocatedSPA.h"

#include "mkl.h"
#ifdef USE_CUDA
#include "cuda.h"
#include "cusparse.h"
#endif
#include <string>
using std::string;


namespace combblas {

template <class IU, class NU>	
class SpTuples;

template <class IU, class NU>	
class SpDCCols;

template <class IU, class NU>	
class SpCCols;

template <class IU, class NU>	
class Dcsc;

template <class IU, class NU>	
class Csc;

/*************************************************************************************************/
/**************************** SHARED ADDRESS SPACE FRIEND FUNCTIONS ******************************/
/****************************** MULTITHREADED LOGIC ALSO GOES HERE *******************************/
/*************************************************************************************************/


//! SpMV with dense vector
template <typename SR, typename IU, typename NU, typename RHS, typename LHS>
void dcsc_gespmv (const SpDCCols<IU, NU> & A, const RHS * x, LHS * y)
{
	if(A.nnz > 0)
	{	
		for(IU j =0; j<A.dcsc->nzc; ++j)	// for all nonzero columns
		{
			IU colid = A.dcsc->jc[j];
			for(IU i = A.dcsc->cp[j]; i< A.dcsc->cp[j+1]; ++i)
			{
				IU rowid = A.dcsc->ir[i];
				SR::axpy(A.dcsc->numx[i], x[colid], y[rowid]);
			}
		}
	}
}

//! SpMV with dense vector (multithreaded version)
template <typename SR, typename IU, typename NU, typename RHS, typename LHS>
void dcsc_gespmv_threaded_nosplit (const SpDCCols<IU, NU> & A, const RHS * x, LHS * y)
{
	if(A.nnz > 0)
	{	
		int nthreads=1;
		#ifdef _OPENMP
		#pragma omp parallel
		{
                	nthreads = omp_get_num_threads();
            	}
		#endif          

		IU nlocrows =  A.getnrow();
		LHS ** tomerge = SpHelper::allocate2D<LHS>(nthreads, nlocrows);
		auto id = SR::id();
		
		for(int i=0; i<nthreads; ++i)
		{
			std::fill_n(tomerge[i], nlocrows, id);		
		}

		#pragma omp parallel for
		for(IU j =0; j<A.dcsc->nzc; ++j)	// for all nonzero columns
		{
			int curthread = 1;
			#ifdef _OPENMP
			curthread = omp_get_thread_num();
			#endif
			
			LHS * loc2merge = tomerge[curthread];

			IU colid = A.dcsc->jc[j];
			for(IU i = A.dcsc->cp[j]; i< A.dcsc->cp[j+1]; ++i)
			{
				IU rowid = A.dcsc->ir[i];
				SR::axpy(A.dcsc->numx[i], x[colid], loc2merge[rowid]);
			}
		}

		#pragma omp parallel for
		for(IU j=0; j < nlocrows; ++j)
		{
			for(int i=0; i< nthreads; ++i)
			{
				y[j] = SR::add(y[j], tomerge[i][j]);
			}
		}
		SpHelper::deallocate2D(tomerge, nthreads);
	}
}
    
    
    
    
/**
 * Multithreaded SpMV with dense vector
 */
template <typename SR, typename IU, typename NU, typename RHS, typename LHS>
void dcsc_gespmv_threaded (const SpDCCols<IU, NU> & A, const RHS * x, LHS * y)
{
	if(A.nnz > 0)
	{
		int splits = A.getnsplit();
		if(splits > 0)
		{
			IU nlocrows = A.getnrow();
			IU perpiece = nlocrows / splits;
			std::vector<int> disp(splits, 0);
			for(int i=1; i<splits; ++i)
				disp[i] = disp[i-1] + perpiece;
#ifdef _OPENMP
#pragma omp parallel for
#endif
			for(int s=0; s<splits; ++s)
			{
				Dcsc<IU, NU> * dcsc = A.GetInternal(s);
				for(IU j =0; j<dcsc->nzc; ++j)    // for all nonzero columns
				{
					IU colid = dcsc->jc[j];
					for(IU i = dcsc->cp[j]; i< dcsc->cp[j+1]; ++i)
					{
						IU rowid = dcsc->ir[i] + disp[s];
						SR::axpy(dcsc->numx[i], x[colid], y[rowid]);
					}
				}
			}
		}
		else
		{
			dcsc_gespmv_threaded_nosplit<SR>(A,x,y);
		}
	}
}


/** 
  * Multithreaded SpMV with sparse vector
  * the assembly of outgoing buffers sendindbuf/sendnumbuf are done here
  */
template <typename SR, typename IU, typename NUM, typename DER, typename IVT, typename OVT>
int generic_gespmv_threaded (const SpMat<IU,NUM,DER> & A, const int32_t * indx, const IVT * numx, int32_t nnzx,
		int32_t * & sendindbuf, OVT * & sendnumbuf, int * & sdispls, int p_c, PreAllocatedSPA<OVT> & SPA)
{
	// FACTS: Split boundaries (for multithreaded execution) are independent of recipient boundaries
	// Two splits might create output to the same recipient (needs to be merged)
	// However, each split's output is distinct (no duplicate elimination is needed after merge) 

	sdispls = new int[p_c]();	// initialize to zero (as all indy might be empty)
	if(A.getnnz() > 0 && nnzx > 0)
	{
		int splits = A.getnsplit();
		if(splits > 0)
		{
			int32_t nlocrows = static_cast<int32_t>(A.getnrow());
			int32_t perpiece = nlocrows / splits;
			std::vector< std::vector< int32_t > > indy(splits);
			std::vector< std::vector< OVT > > numy(splits);

			// Parallelize with OpenMP
			#ifdef _OPENMP
			#pragma omp parallel for // num_threads(6)
			#endif
			for(int i=0; i<splits; ++i)
			{
                if(SPA.initialized)
                {
                    if(i != splits-1)
                         SpMXSpV_ForThreading<SR>(*(A.GetInternal(i)), perpiece, indx, numx, nnzx, indy[i], numy[i], i*perpiece, SPA.V_localy[i], SPA.V_isthere[i], SPA.V_inds[i]);
                    else
                        SpMXSpV_ForThreading<SR>(*(A.GetInternal(i)), nlocrows - perpiece*i, indx, numx, nnzx, indy[i], numy[i], i*perpiece, SPA.V_localy[i], SPA.V_isthere[i], SPA.V_inds[i]);
                }
                else
                {
                    if(i != splits-1)
                        SpMXSpV_ForThreading<SR>(*(A.GetInternal(i)), perpiece, indx, numx, nnzx, indy[i], numy[i], i*perpiece);
                    else
                        SpMXSpV_ForThreading<SR>(*(A.GetInternal(i)), nlocrows - perpiece*i, indx, numx, nnzx, indy[i], numy[i], i*perpiece);
                }
			}

			std::vector<int> accum(splits+1, 0);
			for(int i=0; i<splits; ++i)
				accum[i+1] = accum[i] + indy[i].size();

			sendindbuf = new int32_t[accum[splits]];
			sendnumbuf = new OVT[accum[splits]];
			int32_t perproc = nlocrows / p_c;	
			int32_t last_rec = p_c-1;
			
			// keep recipients of last entries in each split (-1 for an empty split)
			// so that we can delete indy[] and numy[] contents as soon as they are processed		
			std::vector<int32_t> end_recs(splits);
			for(int i=0; i<splits; ++i)
			{
				if(indy[i].empty())
					end_recs[i] = -1;
				else
					end_recs[i] = std::min(indy[i].back() / perproc, last_rec);
			}
			#ifdef _OPENMP
			#pragma omp parallel for // num_threads(6)
			#endif	
			for(int i=0; i<splits; ++i)
			{
				if(!indy[i].empty())	// guarantee that .begin() and .end() are not null
				{
					// FACT: Data is sorted, so if the recipient of begin is the same as the owner of end, 
					// then the whole data is sent to the same processor
					int32_t beg_rec = std::min( indy[i].front() / perproc, last_rec); 			

					// We have to test the previous "split", to see if we are marking a "recipient head" 
					// set displacement markers for the completed (previous) buffers only
					if(i != 0)
					{
						int k = i-1;
						while (k >= 0 && end_recs[k] == -1) k--;	// loop backwards until seeing an non-empty split
						if(k >= 0)	// we found a non-empty split
						{
							std::fill(sdispls+end_recs[k]+1, sdispls+beg_rec+1, accum[i]);	// last entry to be set is sdispls[beg_rec]
						}
						// else fill sdispls[1...beg_rec] with zero (already done)
					}
					// else set sdispls[0] to zero (already done)
					if(beg_rec == end_recs[i])	// fast case
					{
						std::transform(indy[i].begin(), indy[i].end(), indy[i].begin(), std::bind2nd(std::minus<int32_t>(), perproc*beg_rec));
            std::copy(indy[i].begin(), indy[i].end(), sendindbuf+accum[i]);
            std::copy(numy[i].begin(), numy[i].end(), sendnumbuf+accum[i]);
					}
					else	// slow case
					{
						// FACT: No matter how many splits or threads, there will be only one "recipient head"
						// Therefore there are no race conditions for marking send displacements (sdispls)
						int end = indy[i].size();
						for(int cur=0; cur< end; ++cur)	
						{
							int32_t cur_rec = std::min( indy[i][cur] / perproc, last_rec); 			
							while(beg_rec != cur_rec)	
							{
								sdispls[++beg_rec] = accum[i] + cur;	// first entry to be set is sdispls[beg_rec+1]
							}
							sendindbuf[ accum[i] + cur ] = indy[i][cur] - perproc*beg_rec;	// convert to receiver's local index
							sendnumbuf[ accum[i] + cur ] = numy[i][cur];
						}
					}
					std::vector<int32_t>().swap(indy[i]);
					std::vector<OVT>().swap(numy[i]);
					bool lastnonzero = true;	// am I the last nonzero split?
					for(int k=i+1; k < splits; ++k)
					{
						if(end_recs[k] != -1)
							lastnonzero = false;
					} 
					if(lastnonzero)
						std::fill(sdispls+end_recs[i]+1, sdispls+p_c, accum[i+1]);
				}	// end_if(!indy[i].empty)
			}	// end parallel for	
			return accum[splits];
		}
		else
		{
			std::cout << "Something is wrong, splits should be nonzero for multithreaded execution" << std::endl;
			return 0;
		}
	}
	else
	{
		sendindbuf = NULL;
		sendnumbuf = NULL;
		return 0;
	}
}


/** 
 * Multithreaded SpMV with sparse vector and preset buffers
 * the assembly of outgoing buffers sendindbuf/sendnumbuf are done here
 * IVT: input vector numerical type
 * OVT: output vector numerical type
 */
template <typename SR, typename IU, typename NUM, typename DER, typename IVT, typename OVT>
void generic_gespmv_threaded_setbuffers (const SpMat<IU,NUM,DER> & A, const int32_t * indx, const IVT * numx, int32_t nnzx,
				 int32_t * sendindbuf, OVT * sendnumbuf, int * cnts, int * dspls, int p_c)
{
	if(A.getnnz() > 0 && nnzx > 0)
	{
		int splits = A.getnsplit();
		if(splits > 0)
		{
			std::vector< std::vector<int32_t> > indy(splits);
			std::vector< std::vector< OVT > > numy(splits);
			int32_t nlocrows = static_cast<int32_t>(A.getnrow());
			int32_t perpiece = nlocrows / splits;
			
			#ifdef _OPENMP
			#pragma omp parallel for 
			#endif
			for(int i=0; i<splits; ++i)
			{
				if(i != splits-1)
					SpMXSpV_ForThreading<SR>(*(A.GetInternal(i)), perpiece, indx, numx, nnzx, indy[i], numy[i], i*perpiece);
				else
					SpMXSpV_ForThreading<SR>(*(A.GetInternal(i)), nlocrows - perpiece*i, indx, numx, nnzx, indy[i], numy[i], i*perpiece);
			}
			
			int32_t perproc = nlocrows / p_c;	
			int32_t last_rec = p_c-1;
			
			// keep recipients of last entries in each split (-1 for an empty split)
			// so that we can delete indy[] and numy[] contents as soon as they are processed		
			std::vector<int32_t> end_recs(splits);
			for(int i=0; i<splits; ++i)
			{
				if(indy[i].empty())
					end_recs[i] = -1;
				else
					end_recs[i] = std::min(indy[i].back() / perproc, last_rec);
			}
			
			int ** loc_rec_cnts = new int *[splits];	
			#ifdef _OPENMP	
			#pragma omp parallel for
			#endif	
			for(int i=0; i<splits; ++i)
			{
				loc_rec_cnts[i]  = new int[p_c](); // thread-local recipient data
				if(!indy[i].empty())	// guarantee that .begin() and .end() are not null
				{
					int32_t cur_rec = std::min( indy[i].front() / perproc, last_rec);
					int32_t lastdata = (cur_rec+1) * perproc;  // one past last entry that goes to this current recipient
					for(typename std::vector<int32_t>::iterator it = indy[i].begin(); it != indy[i].end(); ++it)
					{

						if( ( (*it) >= lastdata ) && cur_rec != last_rec )
						{
							cur_rec = std::min( (*it) / perproc, last_rec);
							lastdata = (cur_rec+1) * perproc;
						}
						++loc_rec_cnts[i][cur_rec];
					}
				}
			}
			#ifdef _OPENMP	
			#pragma omp parallel for 
			#endif
			for(int i=0; i<splits; ++i)
			{
				if(!indy[i].empty())	// guarantee that .begin() and .end() are not null
				{
					// FACT: Data is sorted, so if the recipient of begin is the same as the owner of end, 
					// then the whole data is sent to the same processor
					int32_t beg_rec = std::min( indy[i].front() / perproc, last_rec); 
					int32_t alreadysent = 0;	// already sent per recipient 
					for(int before = i-1; before >= 0; before--)
						 alreadysent += loc_rec_cnts[before][beg_rec];
						
					if(beg_rec == end_recs[i])	// fast case
					{
            std::transform(indy[i].begin(), indy[i].end(), indy[i].begin(), std::bind2nd(std::minus<int32_t>(), perproc*beg_rec));
            std::copy(indy[i].begin(), indy[i].end(), sendindbuf + dspls[beg_rec] + alreadysent);
            std::copy(numy[i].begin(), numy[i].end(), sendnumbuf + dspls[beg_rec] + alreadysent);
					}
					else	// slow case
					{
						int32_t cur_rec = beg_rec;
						int32_t lastdata = (cur_rec+1) * perproc;  // one past last entry that goes to this current recipient
						for(typename std::vector<int32_t>::iterator it = indy[i].begin(); it != indy[i].end(); ++it)
						{
							if( ( (*it) >= lastdata ) && cur_rec != last_rec )
							{
								cur_rec = std::min( (*it) / perproc, last_rec);
								lastdata = (cur_rec+1) * perproc;

								// if this split switches to a new recipient after sending some data
								// then it's sure that no data has been sent to that recipient yet
						 		alreadysent = 0;
							}
							sendindbuf[ dspls[cur_rec] + alreadysent ] = (*it) - perproc*cur_rec;	// convert to receiver's local index
							sendnumbuf[ dspls[cur_rec] + (alreadysent++) ] = *(numy[i].begin() + (it-indy[i].begin()));
						}
					}
				}
			}
			// Deallocated rec counts serially once all threads complete
			for(int i=0; i< splits; ++i)	
			{
				for(int j=0; j< p_c; ++j)
					cnts[j] += loc_rec_cnts[i][j];
				delete [] loc_rec_cnts[i];
			}
			delete [] loc_rec_cnts;
		}
		else
		{
			std::cout << "Something is wrong, splits should be nonzero for multithreaded execution" << std::endl;
		}
	}
}

//! SpMV with sparse vector
//! MIND: Matrix index type
//! VIND: Vector index type (optimized: int32_t, general: int64_t)
template <typename SR, typename MIND, typename VIND, typename DER, typename NUM, typename IVT, typename OVT>
void generic_gespmv (const SpMat<MIND,NUM,DER> & A, const VIND * indx, const IVT * numx, VIND nnzx, std::vector<VIND> & indy, std::vector<OVT>  & numy, PreAllocatedSPA<OVT> & SPA)
{
	if(A.getnnz() > 0 && nnzx > 0)
	{
		if(A.getnsplit() > 0)
		{
			std::cout << "Call dcsc_gespmv_threaded instead" << std::endl;
		}
		else
		{
			SpMXSpV<SR>(*(A.GetInternal()), (VIND) A.getnrow(), indx, numx, nnzx, indy, numy, SPA);
		}
	}
}

/** SpMV with sparse vector
  * @param[in] indexisvalue is only used for BFS-like computations, if true then we can call the optimized version that skips SPA
  */
template <typename SR, typename IU, typename DER, typename NUM, typename IVT, typename OVT>
void generic_gespmv (const SpMat<IU,NUM,DER> & A, const int32_t * indx, const IVT * numx, int32_t nnzx,
		int32_t * indy, OVT * numy, int * cnts, int * dspls, int p_c, bool indexisvalue)
{
	if(A.getnnz() > 0 && nnzx > 0)
	{
		if(A.getnsplit() > 0)
		{
			SpParHelper::Print("Call dcsc_gespmv_threaded instead\n");
		}
		else
		{
            SpMXSpV<SR>(*(A.GetInternal()), (int32_t) A.getnrow(), indx, numx, nnzx, indy, numy, cnts, dspls, p_c);
		}
	}
}


template<typename IU>
void BooleanRowSplit(SpDCCols<IU, bool> & A, int numsplits)
{
    if(A.m < numsplits)
    {
        std::cerr<< "Warning: Matrix is too small to be splitted for multithreading" << std::endl;
        return;
    }
	A.splits = numsplits;
	IU perpiece = A.m / A.splits;
	std::vector<IU> prevcolids(A.splits, -1);	// previous column id's are set to -1
	std::vector<IU> nzcs(A.splits, 0);
	std::vector<IU> nnzs(A.splits, 0);
	std::vector < std::vector < std::pair<IU,IU> > > colrowpairs(A.splits);
	if(A.nnz > 0 && A.dcsc != NULL)
	{
		for(IU i=0; i< A.dcsc->nzc; ++i)
		{
			for(IU j = A.dcsc->cp[i]; j< A.dcsc->cp[i+1]; ++j)
			{
				IU colid = A.dcsc->jc[i];
				IU rowid = A.dcsc->ir[j];
				IU owner = std::min(rowid / perpiece, static_cast<IU>(A.splits-1));
				colrowpairs[owner].push_back(std::make_pair(colid, rowid - owner*perpiece));

				if(prevcolids[owner] != colid)
				{
					prevcolids[owner] = colid;
					++nzcs[owner];
				}
				++nnzs[owner];
			}
		}
	}
	delete A.dcsc;	// claim memory
	//copy(nzcs.begin(), nzcs.end(), ostream_iterator<IU>(cout," " )); cout << endl;
	//copy(nnzs.begin(), nnzs.end(), ostream_iterator<IU>(cout," " )); cout << endl;	
	A.dcscarr = new Dcsc<IU,bool>*[A.splits];	
	
	// To be parallelized with OpenMP
	for(int i=0; i< A.splits; ++i)
	{
		sort(colrowpairs[i].begin(), colrowpairs[i].end());	// sort w.r.t. columns
        if(nzcs[i]>0)
        {
            A.dcscarr[i] = new Dcsc<IU,bool>(nnzs[i],nzcs[i]);
            std::fill(A.dcscarr[i]->numx, A.dcscarr[i]->numx+nnzs[i], static_cast<bool>(1));
            IU curnzc = 0;                // number of nonzero columns constructed so far
            IU cindex = colrowpairs[i][0].first;
            IU rindex = colrowpairs[i][0].second;
            
            A.dcscarr[i]->ir[0] = rindex;
            A.dcscarr[i]->jc[curnzc] = cindex;
            A.dcscarr[i]->cp[curnzc++] = 0;
            
            for(IU j=1; j<nnzs[i]; ++j)
            {
                cindex = colrowpairs[i][j].first;
                rindex = colrowpairs[i][j].second;
                
                A.dcscarr[i]->ir[j] = rindex;
                if(cindex != A.dcscarr[i]->jc[curnzc-1])
                {
                    A.dcscarr[i]->jc[curnzc] = cindex;
                    A.dcscarr[i]->cp[curnzc++] = j;
                }
            }
            A.dcscarr[i]->cp[curnzc] = nnzs[i];
        }
        else
        {
            A.dcscarr[i] = new Dcsc<IU,bool>();
        }
	}
}


/**
 * SpTuples(A*B') (Using OuterProduct Algorithm)
 * Returns the tuples for efficient merging later
 * Support mixed precision multiplication
 * The multiplication is on the specified semiring (passed as parameter)
 */
template<class SR, class NUO, class IU, class NU1, class NU2>
SpTuples<IU, NUO> * Tuples_AnXBt 
					(const SpDCCols<IU, NU1> & A, 
					 const SpDCCols<IU, NU2> & B,
					bool clearA = false, bool clearB = false)
{
	IU mdim = A.m;	
	IU ndim = B.m;	// B is already transposed

	if(A.isZero() || B.isZero())
	{
		if(clearA)	delete const_cast<SpDCCols<IU, NU1> *>(&A);
		if(clearB)	delete const_cast<SpDCCols<IU, NU2> *>(&B);
		return new SpTuples< IU, NUO >(0, mdim, ndim);	// just return an empty matrix
	}
	Isect<IU> *isect1, *isect2, *itr1, *itr2, *cols, *rows;
	SpHelper::SpIntersect(*(A.dcsc), *(B.dcsc), cols, rows, isect1, isect2, itr1, itr2);
	
	IU kisect = static_cast<IU>(itr1-isect1);		// size of the intersection ((itr1-isect1) == (itr2-isect2))
	if(kisect == 0)
	{
		if(clearA)	delete const_cast<SpDCCols<IU, NU1> *>(&A);
		if(clearB)	delete const_cast<SpDCCols<IU, NU2> *>(&B);
		DeleteAll(isect1, isect2, cols, rows);
		return new SpTuples< IU, NUO >(0, mdim, ndim);
	}
	
	StackEntry< NUO, std::pair<IU,IU> > * multstack;

	IU cnz = SpHelper::SpCartesian< SR > (*(A.dcsc), *(B.dcsc), kisect, isect1, isect2, multstack);  
	DeleteAll(isect1, isect2, cols, rows);

	if(clearA)	delete const_cast<SpDCCols<IU, NU1> *>(&A);
	if(clearB)	delete const_cast<SpDCCols<IU, NU2> *>(&B);
	return new SpTuples<IU, NUO> (cnz, mdim, ndim, multstack);
}

/**
 * SpTuples(A*B) (Using ColByCol Algorithm)
 * Returns the tuples for efficient merging later
 * Support mixed precision multiplication
 * The multiplication is on the specified semiring (passed as parameter)
 */
template<class SR, class NUO, class IU, class NU1, class NU2>
SpTuples<IU, NUO> * Tuples_AnXBn 
					(const SpDCCols<IU, NU1> & A, 
					 const SpDCCols<IU, NU2> & B,
					bool clearA = false, bool clearB = false)
{
	IU mdim = A.m;	
	IU ndim = B.n;	
	if(A.isZero() || B.isZero())
	{
		return new SpTuples<IU, NUO>(0, mdim, ndim);
	}
	StackEntry< NUO, std::pair<IU,IU> > * multstack;
	IU cnz = SpHelper::SpColByCol< SR > (*(A.dcsc), *(B.dcsc), A.n,  multstack);  
	
	if(clearA)	
		delete const_cast<SpDCCols<IU, NU1> *>(&A);
	if(clearB)
		delete const_cast<SpDCCols<IU, NU2> *>(&B);

	return new SpTuples<IU, NUO> (cnz, mdim, ndim, multstack);
}


template<class SR, class NUO, class IU, class NU1, class NU2>
SpTuples<IU, NUO> * Tuples_AtXBt 
					(const SpDCCols<IU, NU1> & A, 
					 const SpDCCols<IU, NU2> & B, 
					bool clearA = false, bool clearB = false)
{
	IU mdim = A.n;	
	IU ndim = B.m;	
	std::cout << "Tuples_AtXBt function has not been implemented yet !" << std::endl;
		
	return new SpTuples<IU, NUO> (0, mdim, ndim);
}

template<class SR, class NUO, class IU, class NU1, class NU2>
SpTuples<IU, NUO> * Tuples_AtXBn 
					(const SpDCCols<IU, NU1> & A, 
					 const SpDCCols<IU, NU2> & B,
					bool clearA = false, bool clearB = false)
{
	IU mdim = A.n;	
	IU ndim = B.n;	
	std::cout << "Tuples_AtXBn function has not been implemented yet !" << std::endl;
		
	return new SpTuples<IU, NUO> (0, mdim, ndim);
}

// Performs a balanced merge of the array of SpTuples
// Assumes the input parameters are already column sorted
template<class SR, class IU, class NU>
SpTuples<IU,NU> MergeAll( const std::vector<SpTuples<IU,NU> *> & ArrSpTups, IU mstar = 0, IU nstar = 0, bool delarrs = false )
{
	int hsize =  ArrSpTups.size();		
	if(hsize == 0)
	{
		return SpTuples<IU,NU>(0, mstar,nstar);
	}
	else
	{
		mstar = ArrSpTups[0]->m;
		nstar = ArrSpTups[0]->n;
	}
	for(int i=1; i< hsize; ++i)
	{
		if((mstar != ArrSpTups[i]->m) || nstar != ArrSpTups[i]->n)
		{
			std::cerr << "Dimensions do not match on MergeAll()" << std::endl;
			return SpTuples<IU,NU>(0,0,0);
		}
	}
	if(hsize > 1)
	{
		ColLexiCompare<IU,int> heapcomp;
		std::tuple<IU, IU, int> * heap = new std::tuple<IU, IU, int> [hsize];	// (rowindex, colindex, source-id)
		IU * curptr = new IU[hsize];
		std::fill_n(curptr, hsize, static_cast<IU>(0)); 
		IU estnnz = 0;

		for(int i=0; i< hsize; ++i)
		{
			estnnz += ArrSpTups[i]->getnnz();
			heap[i] = std::make_tuple(std::get<0>(ArrSpTups[i]->tuples[0]), std::get<1>(ArrSpTups[i]->tuples[0]), i);
		}	
    std::make_heap(heap, heap+hsize, std::not2(heapcomp));

		std::tuple<IU, IU, NU> * ntuples = new std::tuple<IU,IU,NU>[estnnz]; 
		IU cnz = 0;

		while(hsize > 0)
		{
      std::pop_heap(heap, heap + hsize, std::not2(heapcomp));         // result is stored in heap[hsize-1]
			int source = std::get<2>(heap[hsize-1]);

			if( (cnz != 0) && 
				((std::get<0>(ntuples[cnz-1]) == std::get<0>(heap[hsize-1])) && (std::get<1>(ntuples[cnz-1]) == std::get<1>(heap[hsize-1]))) )
			{
				std::get<2>(ntuples[cnz-1])  = SR::add(std::get<2>(ntuples[cnz-1]), ArrSpTups[source]->numvalue(curptr[source]++)); 
			}
			else
			{
				ntuples[cnz++] = ArrSpTups[source]->tuples[curptr[source]++];
			}
			
			if(curptr[source] != ArrSpTups[source]->getnnz())	// That array has not been depleted
			{
				heap[hsize-1] = std::make_tuple(std::get<0>(ArrSpTups[source]->tuples[curptr[source]]), 
								std::get<1>(ArrSpTups[source]->tuples[curptr[source]]), source);
        std::push_heap(heap, heap+hsize, std::not2(heapcomp));
			}
			else
			{
				--hsize;
			}
		}
		SpHelper::ShrinkArray(ntuples, cnz);
		DeleteAll(heap, curptr);
	
		if(delarrs)
		{	
			for(size_t i=0; i<ArrSpTups.size(); ++i)
				delete ArrSpTups[i];
		}
		return SpTuples<IU,NU> (cnz, mstar, nstar, ntuples);
	}
	else
	{
		SpTuples<IU,NU> ret = *ArrSpTups[0];
		if(delarrs)
			delete ArrSpTups[0];
		return ret;
	}
}

/**
 * @param[in]   exclude if false,
 *      \n              then operation is A = A .* B
 *      \n              else operation is A = A .* not(B) 
 **/
template <typename IU, typename NU1, typename NU2>
Dcsc<IU, typename promote_trait<NU1,NU2>::T_promote> EWiseMult(const Dcsc<IU,NU1> & A, const Dcsc<IU,NU2> * B, bool exclude)
{
	typedef typename promote_trait<NU1,NU2>::T_promote N_promote;
	IU estnzc, estnz;
	if(exclude)
	{	
		estnzc = A.nzc;
		estnz = A.nz; 
	} 
	else
	{
		estnzc = std::min(A.nzc, B->nzc);
		estnz  = std::min(A.nz, B->nz);
	}

	Dcsc<IU,N_promote> temp(estnz, estnzc);

	IU curnzc = 0;
	IU curnz = 0;
	IU i = 0;
	IU j = 0;
	temp.cp[0] = 0;
	
	if(!exclude)	// A = A .* B
	{
		while(i< A.nzc && B != NULL && j<B->nzc)
		{
			if(A.jc[i] > B->jc[j]) 		++j;
			else if(A.jc[i] < B->jc[j]) 	++i;
			else
			{
				IU ii = A.cp[i];
				IU jj = B->cp[j];
				IU prevnz = curnz;		
				while (ii < A.cp[i+1] && jj < B->cp[j+1])
				{
					if (A.ir[ii] < B->ir[jj])	++ii;
					else if (A.ir[ii] > B->ir[jj])	++jj;
					else
					{
						temp.ir[curnz] = A.ir[ii];
						temp.numx[curnz++] = A.numx[ii++] * B->numx[jj++];	
					}
				}
				if(prevnz < curnz)	// at least one nonzero exists in this column
				{
					temp.jc[curnzc++] = A.jc[i];	
					temp.cp[curnzc] = temp.cp[curnzc-1] + curnz-prevnz;
				}
				++i;
				++j;
			}
		}
	}
	else	// A = A .* not(B)
	{
		while(i< A.nzc && B != NULL && j< B->nzc)
		{
			if(A.jc[i] > B->jc[j])		++j;
			else if(A.jc[i] < B->jc[j])
			{
				temp.jc[curnzc++] = A.jc[i++];
				for(IU k = A.cp[i-1]; k< A.cp[i]; k++)	
				{
					temp.ir[curnz] 		= A.ir[k];
					temp.numx[curnz++] 	= A.numx[k];
				}
				temp.cp[curnzc] = temp.cp[curnzc-1] + (A.cp[i] - A.cp[i-1]);
			}
			else
			{
				IU ii = A.cp[i];
				IU jj = B->cp[j];
				IU prevnz = curnz;		
				while (ii < A.cp[i+1] && jj < B->cp[j+1])
				{
					if (A.ir[ii] > B->ir[jj])	++jj;
					else if (A.ir[ii] < B->ir[jj])
					{
						temp.ir[curnz] = A.ir[ii];
						temp.numx[curnz++] = A.numx[ii++];
					}
					else	// eliminate those existing nonzeros
					{
						++ii;	
						++jj;	
					}
				}
				while (ii < A.cp[i+1])
				{
					temp.ir[curnz] = A.ir[ii];
					temp.numx[curnz++] = A.numx[ii++];
				}

				if(prevnz < curnz)	// at least one nonzero exists in this column
				{
					temp.jc[curnzc++] = A.jc[i];	
					temp.cp[curnzc] = temp.cp[curnzc-1] + curnz-prevnz;
				}
				++i;
				++j;
			}
		}
		while(i< A.nzc)
		{
			temp.jc[curnzc++] = A.jc[i++];
			for(IU k = A.cp[i-1]; k< A.cp[i]; ++k)
			{
				temp.ir[curnz] 	= A.ir[k];
				temp.numx[curnz++] = A.numx[k];
			}
			temp.cp[curnzc] = temp.cp[curnzc-1] + (A.cp[i] - A.cp[i-1]);
		}
	}

	temp.Resize(curnzc, curnz);
	return temp;
}	

template <typename N_promote, typename IU, typename NU1, typename NU2, typename _BinaryOperation>
Dcsc<IU, N_promote> EWiseApply(const Dcsc<IU,NU1> & A, const Dcsc<IU,NU2> * B, _BinaryOperation __binary_op, bool notB, const NU2& defaultBVal)
{
	//typedef typename promote_trait<NU1,NU2>::T_promote N_promote;
	IU estnzc, estnz;
	if(notB)
	{	
		estnzc = A.nzc;
		estnz = A.nz; 
	} 
	else
	{
		estnzc = std::min(A.nzc, B->nzc);
		estnz  = std::min(A.nz, B->nz);
	}

	Dcsc<IU,N_promote> temp(estnz, estnzc);

	IU curnzc = 0;
	IU curnz = 0;
	IU i = 0;
	IU j = 0;
	temp.cp[0] = 0;
	
	if(!notB)	// A = A .* B
	{
		while(i< A.nzc && B != NULL && j<B->nzc)
		{
			if(A.jc[i] > B->jc[j]) 		++j;
			else if(A.jc[i] < B->jc[j]) 	++i;
			else
			{
				IU ii = A.cp[i];
				IU jj = B->cp[j];
				IU prevnz = curnz;		
				while (ii < A.cp[i+1] && jj < B->cp[j+1])
				{
					if (A.ir[ii] < B->ir[jj])	++ii;
					else if (A.ir[ii] > B->ir[jj])	++jj;
					else
					{
						temp.ir[curnz] = A.ir[ii];
						temp.numx[curnz++] = __binary_op(A.numx[ii++], B->numx[jj++]);	
					}
				}
				if(prevnz < curnz)	// at least one nonzero exists in this column
				{
					temp.jc[curnzc++] = A.jc[i];	
					temp.cp[curnzc] = temp.cp[curnzc-1] + curnz-prevnz;
				}
				++i;
				++j;
			}
		}
	}
	else	// A = A .* not(B)
	{
		while(i< A.nzc && B != NULL && j< B->nzc)
		{
			if(A.jc[i] > B->jc[j])		++j;
			else if(A.jc[i] < B->jc[j])
			{
				temp.jc[curnzc++] = A.jc[i++];
				for(IU k = A.cp[i-1]; k< A.cp[i]; k++)	
				{
					temp.ir[curnz] 		= A.ir[k];
					temp.numx[curnz++] 	= __binary_op(A.numx[k], defaultBVal);
				}
				temp.cp[curnzc] = temp.cp[curnzc-1] + (A.cp[i] - A.cp[i-1]);
			}
			else
			{
				IU ii = A.cp[i];
				IU jj = B->cp[j];
				IU prevnz = curnz;		
				while (ii < A.cp[i+1] && jj < B->cp[j+1])
				{
					if (A.ir[ii] > B->ir[jj])	++jj;
					else if (A.ir[ii] < B->ir[jj])
					{
						temp.ir[curnz] = A.ir[ii];
						temp.numx[curnz++] = __binary_op(A.numx[ii++], defaultBVal);
					}
					else	// eliminate those existing nonzeros
					{
						++ii;	
						++jj;	
					}
				}
				while (ii < A.cp[i+1])
				{
					temp.ir[curnz] = A.ir[ii];
					temp.numx[curnz++] = __binary_op(A.numx[ii++], defaultBVal);
				}

				if(prevnz < curnz)	// at least one nonzero exists in this column
				{
					temp.jc[curnzc++] = A.jc[i];	
					temp.cp[curnzc] = temp.cp[curnzc-1] + curnz-prevnz;
				}
				++i;
				++j;
			}
		}
		while(i< A.nzc)
		{
			temp.jc[curnzc++] = A.jc[i++];
			for(IU k = A.cp[i-1]; k< A.cp[i]; ++k)
			{
				temp.ir[curnz] 	= A.ir[k];
				temp.numx[curnz++] = __binary_op(A.numx[k], defaultBVal);
			}
			temp.cp[curnzc] = temp.cp[curnzc-1] + (A.cp[i] - A.cp[i-1]);
		}
	}

	temp.Resize(curnzc, curnz);
	return temp;
}


template<typename IU, typename NU1, typename NU2>
SpDCCols<IU, typename promote_trait<NU1,NU2>::T_promote > EWiseMult (const SpDCCols<IU,NU1> & A, const SpDCCols<IU,NU2> & B, bool exclude)
{
	typedef typename promote_trait<NU1,NU2>::T_promote N_promote; 
	assert(A.m == B.m);
	assert(A.n == B.n);

	Dcsc<IU, N_promote> * tdcsc = NULL;
	if(A.nnz > 0 && B.nnz > 0)
	{ 
		tdcsc = new Dcsc<IU, N_promote>(EWiseMult(*(A.dcsc), B.dcsc, exclude));
		return 	SpDCCols<IU, N_promote> (A.m , A.n, tdcsc);
	}
	else if (A.nnz > 0 && exclude) // && B.nnz == 0
	{
		tdcsc = new Dcsc<IU, N_promote>(EWiseMult(*(A.dcsc), (const Dcsc<IU,NU2>*)NULL, exclude));
		return 	SpDCCols<IU, N_promote> (A.m , A.n, tdcsc);
	}
	else
	{
		return 	SpDCCols<IU, N_promote> (A.m , A.n, tdcsc);
	}
}


template<typename N_promote, typename IU, typename NU1, typename NU2, typename _BinaryOperation>
SpDCCols<IU, N_promote> EWiseApply (const SpDCCols<IU,NU1> & A, const SpDCCols<IU,NU2> & B, _BinaryOperation __binary_op, bool notB, const NU2& defaultBVal)
{
	//typedef typename promote_trait<NU1,NU2>::T_promote N_promote; 
	assert(A.m == B.m);
	assert(A.n == B.n);

	Dcsc<IU, N_promote> * tdcsc = NULL;
	if(A.nnz > 0 && B.nnz > 0)
	{ 
		tdcsc = new Dcsc<IU, N_promote>(EWiseApply<N_promote>(*(A.dcsc), B.dcsc, __binary_op, notB, defaultBVal));
		return 	SpDCCols<IU, N_promote> (A.m , A.n, tdcsc);
	}
	else if (A.nnz > 0 && notB) // && B.nnz == 0
	{
		tdcsc = new Dcsc<IU, N_promote>(EWiseApply<N_promote>(*(A.dcsc), (const Dcsc<IU,NU2>*)NULL, __binary_op, notB, defaultBVal));
		return 	SpDCCols<IU, N_promote> (A.m , A.n, tdcsc);
	}
	else
	{
		return 	SpDCCols<IU, N_promote> (A.m , A.n, tdcsc);
	}
}

/** 
 * Implementation based on operator +=
 * Element wise apply with the following constraints
 * The operation to be performed is __binary_op
 * The operation `c = __binary_op(a, b)` is only performed if `do_op(a, b)` returns true
 * If allowANulls is true, then if A is missing an element that B has, then ANullVal is used
 * In that case the operation becomes c[i,j] = __binary_op(ANullVal, b[i,j])
 * If both allowANulls and allowBNulls is false then the function degenerates into intersection
 */
template <typename RETT, typename IU, typename NU1, typename NU2, typename _BinaryOperation, typename _BinaryPredicate>
Dcsc<IU, RETT> EWiseApply(const Dcsc<IU,NU1> * Ap, const Dcsc<IU,NU2> * Bp, _BinaryOperation __binary_op, _BinaryPredicate do_op, bool allowANulls, bool allowBNulls, const NU1& ANullVal, const NU2& BNullVal, const bool allowIntersect)
{
	if (Ap == NULL && Bp == NULL)
		return Dcsc<IU,RETT>(0, 0);
	
	if (Ap == NULL && Bp != NULL)
	{
		if (!allowANulls)
			return Dcsc<IU,RETT>(0, 0);
			
		const Dcsc<IU,NU2> & B = *Bp;
		IU estnzc = B.nzc;
		IU estnz  = B.nz;
		Dcsc<IU,RETT> temp(estnz, estnzc);
	
		IU curnzc = 0;
		IU curnz = 0;
		//IU i = 0;
		IU j = 0;
		temp.cp[0] = 0;
		while(j<B.nzc)
		{
			// Based on the if statement below which handles A null values.
			j++;
			IU prevnz = curnz;		
			temp.jc[curnzc++] = B.jc[j-1];
			for(IU k = B.cp[j-1]; k< B.cp[j]; ++k)
			{
				if (do_op(ANullVal, B.numx[k], true, false))
				{
					temp.ir[curnz] 		= B.ir[k];
					temp.numx[curnz++] 	= __binary_op(ANullVal, B.numx[k], true, false);
				}
			}
			//temp.cp[curnzc] = temp.cp[curnzc-1] + (B.cp[j] - B.cp[j-1]);
			temp.cp[curnzc] = temp.cp[curnzc-1] + curnz-prevnz;
		}
		temp.Resize(curnzc, curnz);
		return temp;
	}
	
	if (Ap != NULL && Bp == NULL)
	{
		if (!allowBNulls)
			return Dcsc<IU,RETT>(0, 0);

		const Dcsc<IU,NU1> & A = *Ap;
		IU estnzc = A.nzc;
		IU estnz  = A.nz;
		Dcsc<IU,RETT> temp(estnz, estnzc);
	
		IU curnzc = 0;
		IU curnz = 0;
		IU i = 0;
		//IU j = 0;
		temp.cp[0] = 0;
		while(i< A.nzc)
		{
			i++;
			IU prevnz = curnz;		
			temp.jc[curnzc++] = A.jc[i-1];
			for(IU k = A.cp[i-1]; k< A.cp[i]; k++)
			{
				if (do_op(A.numx[k], BNullVal, false, true))
				{
					temp.ir[curnz] 		= A.ir[k];
					temp.numx[curnz++] 	= __binary_op(A.numx[k], BNullVal, false, true);
				}
			}
			//temp.cp[curnzc] = temp.cp[curnzc-1] + (A.cp[i] - A.cp[i-1]);
			temp.cp[curnzc] = temp.cp[curnzc-1] + curnz-prevnz;
		}
		temp.Resize(curnzc, curnz);
		return temp;
	}
	
	// both A and B are non-NULL at this point
	const Dcsc<IU,NU1> & A = *Ap;
	const Dcsc<IU,NU2> & B = *Bp;
	
	IU estnzc = A.nzc + B.nzc;
	IU estnz  = A.nz + B.nz;
	Dcsc<IU,RETT> temp(estnz, estnzc);

	IU curnzc = 0;
	IU curnz = 0;
	IU i = 0;
	IU j = 0;
	temp.cp[0] = 0;
	while(i< A.nzc && j<B.nzc)
	{
		if(A.jc[i] > B.jc[j])
		{
			j++;
			if (allowANulls)
			{
				IU prevnz = curnz;		
				temp.jc[curnzc++] = B.jc[j-1];
				for(IU k = B.cp[j-1]; k< B.cp[j]; ++k)
				{
					if (do_op(ANullVal, B.numx[k], true, false))
					{
						temp.ir[curnz] 		= B.ir[k];
						temp.numx[curnz++] 	= __binary_op(ANullVal, B.numx[k], true, false);
					}
				}
				//temp.cp[curnzc] = temp.cp[curnzc-1] + (B.cp[j] - B.cp[j-1]);
				temp.cp[curnzc] = temp.cp[curnzc-1] + curnz-prevnz;
			}
		}
		else if(A.jc[i] < B.jc[j])
		{
			i++;
			if (allowBNulls)
			{
				IU prevnz = curnz;		
				temp.jc[curnzc++] = A.jc[i-1];
				for(IU k = A.cp[i-1]; k< A.cp[i]; k++)
				{
					if (do_op(A.numx[k], BNullVal, false, true))
					{
						temp.ir[curnz] 		= A.ir[k];
						temp.numx[curnz++] 	= __binary_op(A.numx[k], BNullVal, false, true);
					}
				}
				//temp.cp[curnzc] = temp.cp[curnzc-1] + (A.cp[i] - A.cp[i-1]);
				temp.cp[curnzc] = temp.cp[curnzc-1] + curnz-prevnz;
			}
		}
		else
		{
			temp.jc[curnzc++] = A.jc[i];
			IU ii = A.cp[i];
			IU jj = B.cp[j];
			IU prevnz = curnz;		
			while (ii < A.cp[i+1] && jj < B.cp[j+1])
			{
				if (A.ir[ii] < B.ir[jj])
				{
					if (allowBNulls && do_op(A.numx[ii], BNullVal, false, true))
					{
						temp.ir[curnz] = A.ir[ii];
						temp.numx[curnz++] = __binary_op(A.numx[ii++], BNullVal, false, true);
					}
					else
						ii++;
				}
				else if (A.ir[ii] > B.ir[jj])
				{
					if (allowANulls && do_op(ANullVal, B.numx[jj], true, false))
					{
						temp.ir[curnz] = B.ir[jj];
						temp.numx[curnz++] = __binary_op(ANullVal, B.numx[jj++], true, false);
					}
					else
						jj++;
				}
				else
				{
					if (allowIntersect && do_op(A.numx[ii], B.numx[jj], false, false))
					{
						temp.ir[curnz] = A.ir[ii];
						temp.numx[curnz++] = __binary_op(A.numx[ii++], B.numx[jj++], false, false);	// might include zeros
					}
					else
					{
						ii++;
						jj++;
					}
				}
			}
			while (ii < A.cp[i+1])
			{
				if (allowBNulls && do_op(A.numx[ii], BNullVal, false, true))
				{
					temp.ir[curnz] = A.ir[ii];
					temp.numx[curnz++] = __binary_op(A.numx[ii++], BNullVal, false, true);
				}
				else
					ii++;
			}
			while (jj < B.cp[j+1])
			{
				if (allowANulls && do_op(ANullVal, B.numx[jj], true, false))
				{
					temp.ir[curnz] = B.ir[jj];
					temp.numx[curnz++] = __binary_op(ANullVal, B.numx[jj++], true, false);
				}
				else
					jj++;
			}
			temp.cp[curnzc] = temp.cp[curnzc-1] + curnz-prevnz;
			++i;
			++j;
		}
	}
	while(allowBNulls && i< A.nzc) // remaining A elements after B ran out
	{
		IU prevnz = curnz;		
		temp.jc[curnzc++] = A.jc[i++];
		for(IU k = A.cp[i-1]; k< A.cp[i]; ++k)
		{
			if (do_op(A.numx[k], BNullVal, false, true))
			{
				temp.ir[curnz] 	= A.ir[k];
				temp.numx[curnz++] = __binary_op(A.numx[k], BNullVal, false, true);
			}
		}
		//temp.cp[curnzc] = temp.cp[curnzc-1] + (A.cp[i] - A.cp[i-1]);
		temp.cp[curnzc] = temp.cp[curnzc-1] + curnz-prevnz;
	}
	while(allowANulls && j < B.nzc) // remaining B elements after A ran out
	{
		IU prevnz = curnz;		
		temp.jc[curnzc++] = B.jc[j++];
		for(IU k = B.cp[j-1]; k< B.cp[j]; ++k)
		{
			if (do_op(ANullVal, B.numx[k], true, false))
			{
				temp.ir[curnz] 	= B.ir[k];
				temp.numx[curnz++] 	= __binary_op(ANullVal, B.numx[k], true, false);
			}
		}
		//temp.cp[curnzc] = temp.cp[curnzc-1] + (B.cp[j] - B.cp[j-1]);
		temp.cp[curnzc] = temp.cp[curnzc-1] + curnz-prevnz;
	}
	temp.Resize(curnzc, curnz);
	return temp;
}

template <typename RETT, typename IU, typename NU1, typename NU2, typename _BinaryOperation, typename _BinaryPredicate> 
SpDCCols<IU,RETT> EWiseApply (const SpDCCols<IU,NU1> & A, const SpDCCols<IU,NU2> & B, _BinaryOperation __binary_op, _BinaryPredicate do_op, bool allowANulls, bool allowBNulls, const NU1& ANullVal, const NU2& BNullVal, const bool allowIntersect)
{
	assert(A.m == B.m);
	assert(A.n == B.n);

	Dcsc<IU, RETT> * tdcsc = new Dcsc<IU, RETT>(EWiseApply<RETT>(A.dcsc, B.dcsc, __binary_op, do_op, allowANulls, allowBNulls, ANullVal, BNullVal, allowIntersect));
	return 	SpDCCols<IU, RETT> (A.m , A.n, tdcsc);
}


////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////// SpMM /////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
struct
spmm_stats
{
	float t_g_h2d_memcpy;
	float t_g_spmm_buf;
	float t_g_spmm;
	float t_g_d2h_memcpy;
	float t_g_mtx_create;

	double t_comp;
	double t_tot;

	double t_sA_comm_pre;
	double t_sA_comm_post;
	
	double t_sC_comm_bcastA;
	double t_sC_comm_bcastX;

	double t_sA_2D_comm_bcastX;
	double t_sA_2D_comm_reduceY;	

    size_t A_comm_size;
    size_t B_comm_size;
    size_t C_comm_size;
};

	
// no split version
template <typename SR,
		  typename IU,
		  typename NU,
		  typename RHS,
		  typename LHS>
void
csc_gespmm_threaded
(
    const SpCCols<IU, NU>	&A,
	const RHS				*X,
	LHS						*Y,
	int						 d,	// dense mat dimension
	std::ofstream			&ofs
)
{
	assert(A.getnsplit() == 0);

	if (A.nnz == 0)
		return;
		
	int splits = A.getnsplit();
	// ofs << "splits " << splits << std::endl;

	int nthreads = 1;
	#ifdef _OPENMP
	#pragma omp parallel
	{
		nthreads = omp_get_num_threads();
	}
	#endif

	IU		  nr	  = A.getnrow();
	LHS		**tomerge = SpHelper::allocate2D<LHS>(nthreads, nr * d);
	auto	  id	  = SR::id();

	for (int i = 0; i < nthreads; ++i)
		std::fill_n(tomerge[i], nr * d, id);

	#pragma omp parallel for
	for (IU j = 0; j < A.csc->n; ++j)
	{
		int thd_idx = 1;	// should be 0 ?
		#ifdef _OPENMP
		thd_idx = omp_get_thread_num();
		#endif

		LHS *tmp = tomerge[thd_idx];
		for (IU i = A.csc->jc[j]; i < A.csc->jc[j+1]; ++i)
		{
			IU r = A.csc->ir[i];
			for (IU k = 0; k < d; ++k)
				SR::axpy(A.csc->num[i], X[j*d + k], tmp[r*d + k]);
		}
	}

	#pragma omp parallel for
	for (IU j = 0; j < nr * d; ++j)
	{
		for (int i = 0; i < nthreads; ++i)
			Y[j] = SR::add(Y[j], tomerge[i][j]);
	} 

	SpHelper::deallocate2D(tomerge, nthreads);

}



// no split version
template <typename SR,
		  typename IU,
		  typename NU,
		  typename RHS,
		  typename LHS>
void
csc_gespmm_mkl
(
    const SpCCols<IU, NU>	&A,
	const RHS				*X,
	LHS						*Y,
	int						 d,	// dense mat dimension
	NU                       beta,
	spmm_stats              &stats
)
{
	if ((A.csc == NULL) || (A.csc->nz == 0)) return;
	sparse_matrix_t A_mkl = NULL;
	mkl_sparse_d_create_csc(
		&A_mkl, SPARSE_INDEX_BASE_ZERO, A.getnrow(), A.getncol(),
		A.csc->jc, A.csc->jc + 1, A.csc->ir, A.csc->num
	);
	struct matrix_descr descr_type_gen;
	descr_type_gen.type = SPARSE_MATRIX_TYPE_GENERAL;
	descr_type_gen.mode = SPARSE_FILL_MODE_FULL;
	descr_type_gen.diag = SPARSE_DIAG_NON_UNIT;
	mkl_sparse_set_mm_hint(A_mkl, SPARSE_OPERATION_NON_TRANSPOSE, descr_type_gen, SPARSE_LAYOUT_ROW_MAJOR, d, 1);
	mkl_sparse_optimize(A_mkl);
	double alpha = 1.0;
	mkl_sparse_d_mm(
		SPARSE_OPERATION_NON_TRANSPOSE, alpha, A_mkl, descr_type_gen, 
		SPARSE_LAYOUT_ROW_MAJOR, X, d, d, beta, Y, d
	);
	mkl_sparse_destroy(A_mkl);
}


#ifdef USE_CUDA
template <typename SR,
		  typename IU,
		  typename NU,
		  typename RHS,
		  typename LHS>
void
csc_gespmm_cusparse
(
    const SpCCols<IU, NU>	&A,
	const RHS				*X,
	LHS						*Y,
	int						 d,	// dense mat dimension
	NU                       beta,
	spmm_stats              &stats
)
{
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	assert(sizeof(RHS) == sizeof(LHS));
	assert(sizeof(RHS) == sizeof(NU));

	if (A.nnz == 0)
		return;
	
	IU		*d_jc	   = NULL, *d_ir = NULL;
	NU		*d_num	   = NULL;
	RHS		*d_Xvals   = NULL;
	LHS		*d_Yvals   = NULL;
	NU		 alpha;
	IU		 nr		   = A.getnrow();
	IU		 nc		   = A.getncol();
	IU		 nnz	   = A.getnnz();
	float	 t_elapsed = 0.0;

	if (nnz == 0 || d == 0)
		return;
	
	cudaEvent_t t_beg, t_end;
	cudaEventCreate(&t_beg); cudaEventCreate(&t_end);
	
	cudaMalloc((void **)&d_jc, (nc+1)*sizeof(*d_jc));
    cudaMalloc((void **)&d_ir, nnz*sizeof(*d_ir));
    cudaMalloc((void **)&d_num, nnz*sizeof(*d_num));
	cudaMalloc((void **)&d_Xvals, (nc*d)*sizeof(*d_Xvals));
	cudaMalloc((void **)&d_Yvals, (nr*d)*sizeof(*d_Yvals));
	
	assert(d_Xvals != NULL && d_Yvals != NULL);

	cusparseIndexType_t cusparse_idx = CUSPARSE_INDEX_32I;
	if (sizeof(IU) == 8)
		cusparse_idx = CUSPARSE_INDEX_64I;
	cudaDataType cusparse_val = CUDA_R_32F;
	if (sizeof(NU) == 8)
		cusparse_val = CUDA_R_64F;

	
	cudaEventRecord(t_beg);
	
	cudaMemcpy(d_jc, A.csc->jc, (nc+1)*sizeof(*d_jc), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ir, A.csc->ir, nnz*sizeof(*d_ir), cudaMemcpyHostToDevice);
    cudaMemcpy(d_num, A.csc->num, nnz*sizeof(*d_num), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Xvals, X, (nc*d)*sizeof(*X), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Yvals, Y, (nr*d)*sizeof(*Y), cudaMemcpyHostToDevice);
	
	cudaEventRecord(t_end);
	cudaEventSynchronize(t_end);
	cudaEventElapsedTime(&t_elapsed, t_beg, t_end);
	stats.t_g_h2d_memcpy += t_elapsed;

	cudaEventRecord(t_beg);
	
	cusparseHandle_t cusparseHandle = 0;
    cusparseCreate(&cusparseHandle);
	cusparseSpMatDescr_t d_A = NULL;
	cusparseCreateCsr(&d_A, nc, nr, nnz, d_jc, d_ir, d_num,
					  cusparse_idx, cusparse_idx,
					  CUSPARSE_INDEX_BASE_ZERO,
					  cusparse_val);	
	cusparseDnMatDescr_t d_X = NULL;
	cusparseCreateDnMat(&d_X, nc, d, d, d_Xvals,
						cusparse_val, CUSPARSE_ORDER_ROW);
	cusparseDnMatDescr_t d_Y = NULL;
	cusparseCreateDnMat(&d_Y, nr, d, d, d_Yvals,
						cusparse_val, CUSPARSE_ORDER_ROW);

	cudaEventRecord(t_end);
	cudaEventSynchronize(t_end);
	cudaEventElapsedTime(&t_elapsed, t_beg, t_end);
	stats.t_g_mtx_create += t_elapsed;
	
	cudaEventRecord(t_beg);
	
	size_t	bufsize = 0;
	alpha			= 1.0;
	// beta			= 0.0;
	cusparseSpMM_bufferSize(cusparseHandle,
							CUSPARSE_OPERATION_TRANSPOSE,
							CUSPARSE_OPERATION_NON_TRANSPOSE,
							&alpha,
							d_A,
							d_X,
							&beta,
							d_Y,
							cusparse_val,
							CUSPARSE_SPMM_CSR_ALG2,
							&bufsize);

	cudaEventRecord(t_end);
	cudaEventSynchronize(t_end);
	cudaEventElapsedTime(&t_elapsed, t_beg, t_end);
	stats.t_g_spmm_buf += t_elapsed;
	
	void *buf_spmm = NULL;
	cudaMalloc(&buf_spmm, bufsize);

	cudaEventRecord(t_beg);
	
	cusparseSpMM(cusparseHandle,
				 CUSPARSE_OPERATION_TRANSPOSE,
				 CUSPARSE_OPERATION_NON_TRANSPOSE,
				 &alpha,
				 d_A,
				 d_X,
				 &beta,
				 d_Y,
				 cusparse_val,
				 CUSPARSE_SPMM_CSR_ALG2,
				 buf_spmm);

	cudaEventRecord(t_end);
	cudaEventSynchronize(t_end);
	cudaEventElapsedTime(&t_elapsed, t_beg, t_end);
	stats.t_g_spmm += t_elapsed;

	cudaEventRecord(t_beg);
	cudaMemcpy(Y, d_Yvals, (nr*d)*sizeof(*Y), cudaMemcpyDeviceToHost);
	cudaEventRecord(t_end);
	cudaEventSynchronize(t_end);
	cudaEventElapsedTime(&t_elapsed, t_beg, t_end);
	stats.t_g_d2h_memcpy += t_elapsed;

	cudaFree(d_jc);
	cudaFree(d_ir);
	cudaFree(d_num);
	cudaFree(d_Xvals);
	cudaFree(d_Yvals);
	cudaFree(buf_spmm);

	return;
}



// multi-gpu
// @DONT-USE
template <typename SR,
		  typename IU,
		  typename NU,
		  typename RHS,
		  typename LHS>
void
csc_gespmm_cusparse_v2
(
    const SpCCols<IU, NU>	&A,
	const RHS				*X,
	LHS						*Y,
	int						 d,	// dense mat dimension
	std::ofstream			&ofs,
	spmm_stats              &stats
)
{
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if (A.nnz == 0)
		return;
	
	assert(sizeof(RHS) == sizeof(LHS));
	assert(sizeof(RHS) == sizeof(NU));

	IU	 nr	 = A.getnrow();
	IU	 nc	 = A.getncol();
	IU	 nnz = A.getnnz();
	IU	*jc	 = A.csc->jc;
	IU	*ir	 = A.csc->ir;
	NU	*num = A.csc->num;

	cusparseIndexType_t cusparse_idx = CUSPARSE_INDEX_32I;
	if (sizeof(IU) == 8)
		cusparse_idx = CUSPARSE_INDEX_64I;
	cudaDataType cusparse_val = CUDA_R_32F;
	if (sizeof(NU) == 8)
		cusparse_val = CUDA_R_64F;

	int nthreads = 1;
	#ifdef _OPENMP
	#pragma omp parallel
	{
		nthreads = omp_get_num_threads();
	}
	#endif
	int ndevices;
	cudaGetDeviceCount(&ndevices);

	ofs << "nthreads " << nthreads << " ndevices " << ndevices << "\n";
	assert(nthreads >= ndevices);

	IU	  nelems  = nc / ndevices;
	LHS	**tomerge = SpHelper::allocate2D<LHS>(ndevices, nr * d);
	omp_set_num_threads(ndevices);

	string		thd_logs[ndevices];
	spmm_stats	thd_stats[ndevices];

	#pragma omp parallel
	{
		cudaEvent_t t_beg, t_end;
		cudaEventCreate(&t_beg); cudaEventCreate(&t_end);
		
		int tid = omp_get_thread_num();
		cudaSetDevice(tid);

		int gpu_id = -1;
		int num_cpu_threads = omp_get_num_threads();
		cudaGetDevice(&gpu_id);

		IU beg = tid * nelems;
		IU end = -1;
		if (tid == ndevices - 1)
			end = nc;
		else
			end = (tid + 1) * nelems;
		IU ncols = end - beg;
		IU *jc_tmp = (IU *)malloc(sizeof(*jc_tmp) * (ncols + 1));
		for (IU i = 0; i <= ncols; ++i)
			jc_tmp[i] = jc[beg + i] - jc[beg];
		IU nnzs = jc[end] - jc[beg];

		thd_logs[tid] += "beg " + std::to_string(beg) +
			" end " + std::to_string(end) + " nnzs " +
			std::to_string(nnzs) + " jc[beg] " +
			std::to_string(jc[beg]) + "\n";

		// device variables
		IU		*d_jc	 = NULL, *d_ir = NULL;
		NU		*d_num	 = NULL;
		RHS		*d_Xvals = NULL;
		LHS		*d_Yvals = NULL;
		double	 alpha, beta;

		cudaMalloc((void **)&d_jc, (ncols+1)*sizeof(*d_jc));
		cudaMalloc((void **)&d_ir, nnzs*sizeof(*d_ir));
		cudaMalloc((void **)&d_num, nnzs*sizeof(*d_num));
		cudaMalloc((void **)&d_Xvals, (ncols*d)*sizeof(*d_Xvals));
		cudaMalloc((void **)&d_Yvals, (nr*d)*sizeof(*d_Yvals));

		cudaEventRecord(t_beg);
		cusparseHandle_t cusparseHandle = 0;
    	cusparseCreate(&cusparseHandle);
		cusparseSpMatDescr_t d_A = NULL;
		cusparseCreateCsr(&d_A, ncols, nr, nnzs, d_jc, d_ir, d_num,
						  cusparse_idx, cusparse_idx,
						  CUSPARSE_INDEX_BASE_ZERO,
						  cusparse_val);
		cusparseDnMatDescr_t d_X = NULL;
		cusparseCreateDnMat(&d_X, ncols, d, d, d_Xvals,
							cusparse_val, CUSPARSE_ORDER_ROW);	
		cusparseDnMatDescr_t d_Y = NULL;
		cusparseCreateDnMat(&d_Y, nr, d, d, d_Yvals,
							cusparse_val, CUSPARSE_ORDER_ROW);
		cudaEventRecord(t_end);
		cudaEventSynchronize(t_end);
		cudaEventElapsedTime(&thd_stats[tid].t_g_mtx_create, t_beg, t_end);
		// float tmp = -1;
		// cudaEventElapsedTime(&tmp, t_beg, t_end);
		// ofs << tmp << std::endl;

		cudaEventRecord(t_beg);
		cudaMemcpy(d_jc, jc_tmp, (ncols+1)*sizeof(*d_jc),
				   cudaMemcpyHostToDevice);
    	cudaMemcpy(d_ir, &ir[jc[beg]], nnzs*sizeof(*d_ir),
				   cudaMemcpyHostToDevice);
    	cudaMemcpy(d_num, &num[jc[beg]], nnzs*sizeof(*d_num),
				   cudaMemcpyHostToDevice);
		cudaMemcpy(d_Xvals, &X[beg*d], (ncols*d)*sizeof(*X),
				   cudaMemcpyHostToDevice);
		cudaEventRecord(t_end);
		cudaEventSynchronize(t_end);
		cudaEventElapsedTime(&thd_stats[tid].t_g_h2d_memcpy, t_beg, t_end);	

		free(jc_tmp);

		cudaEventRecord(t_beg);
		size_t	bufsize = 0;
		alpha			= 1.0;
		beta			= 0.0;
		cusparseSpMM_bufferSize(cusparseHandle,
							CUSPARSE_OPERATION_TRANSPOSE,
							CUSPARSE_OPERATION_NON_TRANSPOSE,
							&alpha,
							d_A,
							d_X,
							&beta,
							d_Y,
							cusparse_val,
							CUSPARSE_SPMM_CSR_ALG2,
							&bufsize);
		cudaEventRecord(t_end);
		cudaEventSynchronize(t_end);
		cudaEventElapsedTime(&thd_stats[tid].t_g_spmm_buf, t_beg, t_end);
		void *buf_spmm = NULL;
		cudaMalloc(&buf_spmm, bufsize);

		cudaEventRecord(t_beg);
		cusparseSpMM(cusparseHandle,
				 CUSPARSE_OPERATION_TRANSPOSE,
				 CUSPARSE_OPERATION_NON_TRANSPOSE,
				 &alpha,
				 d_A,
				 d_X,
				 &beta,
				 d_Y,
				 cusparse_val,
				 CUSPARSE_SPMM_CSR_ALG2,
				 buf_spmm);
		cudaEventRecord(t_end);
		cudaEventSynchronize(t_end);
		cudaEventElapsedTime(&thd_stats[tid].t_g_spmm, t_beg, t_end);

		cudaEventRecord(t_beg);
		cudaMemcpy(tomerge[tid], d_Yvals, (nr*d)*sizeof(*Y),
				   cudaMemcpyDeviceToHost);
		cudaEventRecord(t_end);
		cudaEventSynchronize(t_end);
		cudaEventElapsedTime(&thd_stats[tid].t_g_d2h_memcpy, t_beg, t_end);

		cudaFree(d_jc);
		cudaFree(d_ir);
		cudaFree(d_num);
		cudaFree(d_Xvals);
		cudaFree(d_Yvals);
		cudaFree(buf_spmm);		
	}


	for (int i = 0; i < ndevices; ++i)
	{
		ofs << "thread " << i << " log \n";
		ofs << thd_logs[i];

		// stats.t_g_h2d_memcpy += thd_stats[i].t_g_h2d_memcpy;
		// stats.t_g_spmm_buf	 += thd_stats[i].t_g_spmm_buf;
		// stats.t_g_spmm		 += thd_stats[i].t_g_spmm;
		// stats.t_g_d2h_memcpy += thd_stats[i].t_g_d2h_memcpy;
		// stats.t_g_mtx_create += thd_stats[i].t_g_mtx_create;
	}

	// stats.t_g_h2d_memcpy /= ndevices;
	// stats.t_g_spmm_buf /= ndevices;
	// stats.t_g_spmm /= ndevices;
	// stats.t_g_d2h_memcpy /= ndevices;
	// stats.t_g_mtx_create /= ndevices;	

	omp_set_num_threads(nthreads);

	#pragma omp parallel for
	for (IU j = 0; j < nr * d; ++j)
	{
		for (int i = 0; i < ndevices; ++i)
			Y[j] += tomerge[i][j];
	} 

	SpHelper::deallocate2D(tomerge, ndevices);	
	
	
	return;
}

#endif // #ifdef USE_CUDA	
	

}

#endif
