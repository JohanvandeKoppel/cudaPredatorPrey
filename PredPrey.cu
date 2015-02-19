/*
ROHIT GUPTA & JOHAN VAN DE KOPPEL
Predator-Prey Spirals
June 2010
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>

// include CUDA
#include <cuda_runtime.h>

// include parameter kernel
#include "predprey.h"

////////////////////////////////////////////////////////////////////////////////
// A set of blocks as initial values that quickly generates spirals
////////////////////////////////////////////////////////////////////////////////

void blockInit(float* data, int x_siz, int y_siz, int type)
{
 int i,j;
 int Mi=100;
 int Mj=100;
 float Ci,Cj;

 for(i=0;i<y_siz;i++)
    {
	 for(j=0;j<x_siz;j++)
 		{ 
			//for every element find the correct initial
			//value using the conditions below			
			if(i==0||i==y_siz-1||j==0||j==x_siz-1)
				data[i*y_siz+j]=0.0f; // This value for the boundaries
			else 
			   {
				Ci=((float)pow(i,1.005)/(float)Mi)-floor((float)i/(float)Mi);
                Cj=((float)pow(j,1.005)/(float)Mj)-floor((float)j/(float)Mj);
				   
				if(type==PREY)
					{
                     if(Ci>0.5||Cj>0.5)
					    {
                         data[i*y_siz+j]=0.5f;
					    }
					}
				else if(type==PREDATOR)
				    {
                     if(Ci>0.97||Cj>0.75)
					    {
                         data[i*y_siz+j]=5.0f;
					    }
				    }
				   
			   }
		}
   }
} // End BlockInit

////////////////////////////////////////////////////////////////////////////////
// Allocates a matrix with random float entries
////////////////////////////////////////////////////////////////////////////////

void randomInit(float* data, int x_siz, int y_siz, int type)
{
	int i,j;
	for(i=0;i<y_siz;i++)
	{
		for(j=0;j<x_siz;j++)
		{
			//for every element find the correct initial
			//value using the conditions below
			if(i==0||i==y_siz-1||j==0||j==x_siz-1)
			    data[i*y_siz+j]=0.0f; // This value for the boundaries
			else
			{
				if(type==PREDATOR)					
				{
                // A randomized initiation here
				if((rand() / (float)RAND_MAX)<0.0005f)
							data[i*y_siz+j] = 1.0f;
						else
							data[i*y_siz+j] = 0.0f;
				}
				else if(type==PREY)
					data[i*y_siz+j]=(float)1.0f;

			}
		}
	}			

} // End randomInit


////////////////////////////////////////////////////////////////////////////////
// Laplacation operator definition, to calculate diffusive fluxes
////////////////////////////////////////////////////////////////////////////////

__device__ float
LaplacianXY(float* pop, int row, int column)
{
	float retval;
	int current, left, right, top, bottom;	
	float dx = dX;
	float dy = dY;
	
	current=row * WidthGrid + column;	
	left=row * WidthGrid + column-1;
	right=row * WidthGrid + column+1;
	top=(row-1) * WidthGrid + column;
	bottom=(row+1) * WidthGrid + column;

	retval = ( (( pop[current] - pop[left] )/dx ) 
		      -(( pop[right] - pop[current] )/dx )) / dx + 
		     ( (( pop[current] - pop[top] )/dy  ) 
			  -(( pop[bottom] - pop[current] )/dy ) ) / dy;

	return retval;
}

////////////////////////////////////////////////////////////////////////////////
// Simulation kernel
////////////////////////////////////////////////////////////////////////////////

__global__ void 
predpreyKernel (float* Prey, float* Pred)
{

	//run for U X V times. For every U times completed store in the array storeA and storeM

	int current;

	float d2Preydxy2,d2Preddxy2;
	float drPrey,drPred;
	
	int row    = blockIdx.y*Block_Size_Y+threadIdx.y;
	int column = blockIdx.x*Block_Size_X+threadIdx.x;	

	current=row * WidthGrid + column;	

	if (row > 0 && row < WidthGrid-1 && column > 0 && column < WidthGrid-1) 
	   {
		//calcualte diffusions for predator and prey in X and Y directions
		// update the current grid values

		//Now calculating terms for the Prey Matrix
		d2Preydxy2 =  LaplacianXY(Prey, row, column);
		drPrey = Prey[current] * (1.0f - Prey[current]) - C/(1.0f + C*Prey[current]) * Prey[current] * Pred[current];
		Prey[current]=Prey[current]+(drPrey + -DifPrey*d2Preydxy2)*dT;

		//Now calculating terms for the Predator Matrix
		d2Preddxy2 =  LaplacianXY(Pred, row, column);
		drPred = Pred[current]/(A * B)*( A * C * Prey[current]/(1.0f + C*Prey[current]) - 1.0f);
		Pred[current]=Pred[current]+(drPred + -DifPred*d2Preddxy2)*dT;
	   }
       
	__syncthreads();

	// HANDLE Boundaries (Periodic)

	if(row==0)
			{
				Prey[row * WidthGrid+column]=Prey[(HeightGrid-2) * WidthGrid+column];
				Pred[row * WidthGrid+column]=Pred[(HeightGrid-2) * WidthGrid+column];
			}

	else if(row==HeightGrid-1)
			{
				Prey[row * WidthGrid + column]=Prey[1*WidthGrid+column];
				Pred[row * WidthGrid + column]=Pred[1*WidthGrid+column];
			}
	
	else if(column==0)
			{
				Prey[row * WidthGrid + column]=Prey[row * WidthGrid + WidthGrid-2];
				Pred[row * WidthGrid + column]=Pred[row * WidthGrid + WidthGrid-2];
			}	

	else if(column==WidthGrid-1)
			{
				Prey[row * WidthGrid + column]=Prey[row * WidthGrid + 1];
				Pred[row * WidthGrid + column]=Pred[row * WidthGrid + 1];
			}	
	
} // End PredPreyKernel

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////

// ---- Program setup ------------------------------------------
int
main(int argc, char** argv)
{

	long frame_count;      // The number of timesteps that have past since the last frame was stores
	int store_count;       // The number of times a frame was stored
	float time_elapsed;    // The amount of time that has passed
	 
	float StoreSteps;      // The number of timesteps that passes before a frame is stored
	int NumStored;
	int store_i;
	unsigned int size_grid = WidthGrid * HeightGrid;
	unsigned int mem_size_grid = sizeof(float) * size_grid;
	unsigned int size_storegrid = WidthGrid * HeightGrid * MAX_STORE;
	unsigned int mem_size_storegrid = sizeof(float) * size_storegrid;

	float* h_store_Prey;
	float* h_store_Pred;

	float* h_Prey;
	float* h_Pred;		

	float* d_Prey;
	float* d_Pred;   
 
	FILE *fp;
    
	int height_matrix=HeightGrid;
	int width_matrix=WidthGrid;

	/*--------------------INITIALIZATIONS ON HOST-------------------*/
	time_elapsed=Time;
   	frame_count=EndTime;
	store_count=0;
	StoreSteps=float(EndTime)/float(NumFrames)/float(dT);
	
	// set seed for rand()
	srand((unsigned)time( NULL ));
    
	//allocate host memory for matrices Prey and Pred
	h_Prey = (float*) malloc(mem_size_grid);
	h_Pred = (float*) malloc(mem_size_grid);

	//allocate host memory for matrices store_Prey and store_Pred
	h_store_Prey = (float*) malloc(mem_size_storegrid);
	h_store_Pred = (float*) malloc(mem_size_storegrid);

    /*----------------- INITIALIZING THE ARRAYS ON THE CPU -------------*/
	//blockInit(h_Prey, WidthGrid, HeightGrid, PREY);
	//blockInit(h_Pred, WidthGrid, HeightGrid, PREDATOR);

	randomInit(h_Prey, WidthGrid, HeightGrid, PREY);
	randomInit(h_Pred, WidthGrid, HeightGrid, PREDATOR);

    /*----------------- INITIALIZATION ON THE GPU -----------------------------*/
	// allocate device memory
	cudaMalloc((void**) &d_Prey, mem_size_grid);
	cudaMalloc((void**) &d_Pred, mem_size_grid);
	
	//copy host memory to device
	cudaMemcpy(d_Prey, h_Prey, mem_size_grid, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Pred, h_Pred, mem_size_grid, cudaMemcpyHostToDevice);
    
    /*---------------------SETUP EXECUTION PARAMETERS-------------------------*/	
	dim3 threads;      // Setting up the GPU setting, thread block size
	dim3 grid;         // Setting up the GPU setting, grid structure

	threads.x= Block_Size_X;
	threads.y= Block_Size_Y;
	grid.x=DIVIDE_INTO(WidthGrid,threads.x);
	grid.y=DIVIDE_INTO(WidthGrid,threads.y);
  
	// create and start timer    
    struct timeval Time_Measured;
    gettimeofday(&Time_Measured, NULL);
    double Time_Begin=Time_Measured.tv_sec+(Time_Measured.tv_usec/1000000.0);	
   
    /*----- Printing info to the screen --------------------------------*/
	system("clear");
        printf("\n");
	printf(" * * * * * * * * * * * * * * * * * * * * * * * * * * * * * \n");
	printf(" * Predator Prey Spirals                                 * \n");		
	printf(" * CUDA implementation : Rohit Gupta, 2009               * \n");
	printf(" * Following a model by Sherrat et al 2001               * \n");
	printf(" * * * * * * * * * * * * * * * * * * * * * * * * * * * * * \n\n");

	printf(" Current frame dimensions: %d x %d\n\n", WidthGrid, HeightGrid);

    /*----- The simulation loop ----------------------------------------*/
	while(time_elapsed<EndTime)
	   {
	    // execute the kernel
		predpreyKernel<<< grid, threads >>>(d_Prey, d_Pred);

		frame_count=frame_count+1;

		if(frame_count>=StoreSteps)
		   {		
		    cudaMemcpy((void *)h_Prey, (void *)d_Prey, mem_size_grid,
					   cudaMemcpyDeviceToHost);
			cudaMemcpy((void *)h_Pred, (void *)d_Pred, mem_size_grid,
					   cudaMemcpyDeviceToHost);	

			//Store values of current frame.
			memcpy(h_store_Prey+(store_count*size_grid),h_Prey,mem_size_grid);
			memcpy(h_store_Pred+(store_count*size_grid),h_Pred,mem_size_grid);			

			fprintf(stderr, "\r Current timestep: %1.0f, Storepoint %1.0d of %1.0d",time_elapsed,store_count, NumFrames);

			frame_count=0;
			store_count=store_count+1;		

		   }// if on writing one frame ends

		time_elapsed=time_elapsed+(float)dT;
		
	   }//while on time ends
    

	/*---------------------Report on time spending----------------------------*/
    gettimeofday(&Time_Measured, NULL);
    double Time_End=Time_Measured.tv_sec+(Time_Measured.tv_usec/1000000.0);
	printf("\r Processing time: %4.2f (s)                         \n\n",
	       Time_End-Time_Begin);

	/*---------------------Write to file now----------------------------------*/
	fp=fopen("PredPrey.dat","wb");
	fwrite(&width_matrix,sizeof(int),1,fp);	
	fwrite(&height_matrix,sizeof(int),1,fp);
	NumStored=store_count;
	float Length = dX*(float)WidthGrid;
    int EndTimeVal = EndTime;
	fwrite(&NumStored,sizeof(int),1,fp);	
	fwrite(&Length,sizeof(float),1,fp);
	fwrite(&EndTimeVal,sizeof(int),1,fp);	
	
	for(store_i=0;store_i<store_count;store_i++)
	{
	     fwrite(&h_store_Prey[store_i*size_grid],sizeof(float),size_grid,fp);
		 fwrite(&h_store_Pred[store_i*size_grid],sizeof(float),size_grid,fp);

         printf("\r Saving simulation results: %2.0f%%", (float)store_i/(float)store_count*100.0);
	}        

	printf("\r Saving simulation results: 100%%\n\n"); 
	     
	fclose(fp);     
	
    /*---------------------Clean up memory------------------------------------*/
	free(h_Prey);
	free(h_Pred);

	free(h_store_Pred);
	free(h_store_Prey);
    
	cudaFree(d_Prey);
	cudaFree(d_Pred);
   
	cudaThreadExit();
	cudaDeviceReset();

}
