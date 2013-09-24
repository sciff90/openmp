#include<iostream>
#include<cmath>
#include<random>
#include<omp.h>
#include<string>
#include<fstream>
#include<vector>

//Function Declarations
double RSS(double y1[],double y2[],int Npts);
void filter_out(double a[],double b[],double y[],double u[],int N,int n_order);

int main(int argc, char *argv[])
{
	std::cout << "Parallel MCMC" << std::endl;

	//Setup OMP part nthreads etc
	int nthreads,tid;	
	#pragma omp parallel private(tid)
	{
		tid = omp_get_thread_num();
	
		if(tid == 0)
		{
			nthreads = omp_get_num_threads();
		}
	}
	
	std::cout << "nthreads = " << nthreads << std::endl;

	//Read in Parameters
	int order,num_samples;
	double fs,fc,fnorm,dt,t1,t0;

	std::ifstream params;
	std::string fn = "data/params.dat";
	params.open(fn);
	params.ignore(10000,'\n');
	params>>order;
	params>>fs;
	params>>fc;
	params>>fnorm;
	params>>dt;
	params>>t1;
	params>>t0;
	params>>num_samples;

	params.close();

	//Read in u t and D
	double u[num_samples],D[num_samples],t[num_samples];
	std::ifstream u_dat,D_dat,t_dat;
	u_dat.open("data/u.dat");
	D_dat.open("data/D.dat");
	t_dat.open("data/t.dat");
	std::cout<<"order = "<<order<<std::endl;
	std::cout<<"num_samples = "<<num_samples<<std::endl;
	for(int ii=0;ii<num_samples;ii++)
	{
		u_dat>>u[ii];
		D_dat>>D[ii];
		t_dat>>t[ii];
	}
	u_dat.close();
	D_dat.close();
	t_dat.close();

	
	//Define a and b coeffs
	double b_curr[order+1],a_curr[order+1],b_best[order+1],a_best[order+1];
	//Define Output
	double y_curr[num_samples];
	
	for(int ii=0;ii<order+1;ii++)
	{
		b_curr[ii] = rand()%1;
		a_curr[ii] = rand()%1;
	}
	a_curr[0] = 1.0;

	filter_out(a_curr,b_curr,y_curr,u,num_samples,order);
	
	double chi_curr,chi_cand,chi_best,sigma,ratio;
	chi_curr = RSS(D,y_curr,num_samples);
	chi_best = chi_curr;
	sigma = 1.0;
	int count = 0;
	int flg = 0;
	int accepted = 0;
	int n = 0;
	int Nmax = 100000;
	int burnin = 0;
	double a_cand[order+1],b_cand[order+1];
	double y_cand[num_samples];
	double a_save[Nmax][nthreads][order+1],b_save[Nmax][nthreads][order+1];
	double num,num2;
	
	//Random number generator
	std::mt19937 generator;
	//Random number distributions
	std::normal_distribution<double> distribution(0,1.0);
	std::uniform_real_distribution<double> udistribution(0.0,1.0);

	//Parallel part
	
	double start = omp_get_wtime();
	#pragma omp parallel private(tid,a_cand,b_cand,a_curr,b_curr,burnin,accepted,flg,chi_best,chi_curr,chi_cand,sigma,ratio,count,y_cand,n)
	{
	tid = omp_get_thread_num();
	while(n<=Nmax)
	{	
		
		for(int ii = 0; ii < order+1; ii++)
		{
			a_cand[ii] = a_curr[ii] + sigma*distribution(generator);
			b_cand[ii] = b_curr[ii] + sigma*distribution(generator);
		}
		a_cand[0] = 1.0;
		filter_out(a_cand,b_cand,y_cand,u,num_samples,order);
		chi_cand = RSS(D,y_cand,num_samples);
		ratio = exp(-(chi_cand)+(chi_curr));
		if(udistribution(generator)<ratio)
		{	
			//std::cout<<"accepted"<<std::endl;
			for(int ii=0;ii<order+1;ii++)
			{
				a_curr[ii] = a_cand[ii];
				b_curr[ii] = b_cand[ii];			
			}
			chi_curr = chi_cand;
			if(chi_cand<chi_best)
			{
				chi_best = chi_cand;
				for(int ii=0;ii<order+1;ii++)
				{
					a_best[ii] = a_curr[ii];
					b_best[ii] = a_curr[ii];
				}
			}
			accepted++;
		}

		if(count%1000==0 && count!=0 &&flg==0)
		{	
			double a_ratio = (double)(accepted)/count;
			std::cout<<"a_ratio = "<<a_ratio<<std::endl;
			if(a_ratio<0.3)
			{
				sigma = sigma/1.2;
				count = 0;
				accepted = 0;
				//distribution = std::normal_distribution<double>(0,sigma);
			}
			else if(a_ratio>0.4)
			{
				sigma = sigma*1.2;
				count = 0;
				accepted = 0;
				//distribution = std::normal_distribution<double>(0,sigma);
			}
			else
			{
				burnin = n-1;
				flg=1;
			}
		}
		count++;		
		n++;
		if(flg==1)
		{
			for(int ii=0;ii<order+1;ii++)
			{
				a_save[n-(burnin+1)][tid][ii] = a_curr[ii];
				b_save[n-(burnin+1)][tid][ii] = b_curr[ii];
			}
		}
	}
	}

	std::ofstream a_dat,b_dat;
	a_dat.open("data/a.dat");
	b_dat.open("data/b.dat");
	for(int ii=0;ii<Nmax-burnin;ii++)
	{
		for(int jj=0;jj<order+1;jj++)
		{
			a_dat << a_save[ii][jj] << "\t";
			b_dat << b_save[ii][jj] << "\t";

		}
		a_dat << std::endl;
		b_dat <<std::endl;
	}
	std::cout<<"a_best[1] = "<<a_best[1]<<std::endl;
	std::cout<<"sigma = " <<sigma<<std::endl;
	return 0;
}

void filter_out(double a[],double b[],double y[],double u[],int N,int n_order)
{
	int ii,jj;

	for(ii=0;ii<N;ii++)y[ii] = 0;
	for (ii = n_order; ii < (N); ii++)
	{
		for (jj = 1; jj <= n_order; jj++)
		{
			y[ii] = y[ii] - a[jj]*y[ii-jj];
		}
		for (jj = 0; jj <= n_order; jj++)
		{
			y[ii] = y[ii]+b[jj]*u[ii-jj];
		}
		
		y[ii] = y[ii]/a[0];
		//if(abs(y[ii])>1)y[ii] = 10;
	}
}


double RSS(double y1[],double y2[],int Npts)	
{
	int ii;
	double total = 0;
	for (ii = 0; ii < Npts; ii++)
	{
		total = total + pow((y1[ii]-y2[ii]),2);
	}
	return total;
}
