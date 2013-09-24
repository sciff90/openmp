#include<iostream>
#include<cmath>
#include<random>
#include<omp.h>
#include<string>
#include<fstream>
#include<vector>
int main()
{
	std::cout << "Testing openmp Random chain" << std::endl;
	
	const int N = 1000000;	

	int nthreads,tid;
	
	#pragma omp parallel private(tid)
	{
		tid = omp_get_thread_num();
	
		if(tid == 0)
		{
			nthreads = omp_get_num_threads();
		}
	}

	double randTable[nthreads][N];

	std::cout << "nthreads = " << nthreads << std::endl;
	std::mt19937 generator[nthreads];
	for(int ii = 0;ii<nthreads;ii++)
	std::fill(generator+ii,generator+ii+1,std::mt19937(ii*256*256));


	std::normal_distribution<double> distribution[nthreads];
	std::fill(distribution,distribution+nthreads,std::normal_distribution<double>(0,1));
	double start = omp_get_wtime();
	int count;

	#pragma omp parallel private(tid,count) 
	{
		tid = omp_get_thread_num();	
		#pragma omp barrier		
		for (count = 0; count < N; count++)
		{	
			randTable[tid][count] = distribution[tid](generator[tid]);
		}		
		#pragma omp barrier			
	}

	std::cout<<"Time: \t"<<omp_get_wtime()-start<<std::endl;
	
	std::ofstream myfile;
	std::string fn = "data/normal.dat";
	myfile.open(fn);
	for(int jj = 0; jj<nthreads ;jj++)
	{
		for(int ii=0;ii<N;ii++)
		{
			myfile << randTable[jj][ii]<<std::endl;
		
		}
	}
	myfile.close();

	
	return 0;
}
