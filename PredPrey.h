/* Rohit Gupta
parts of code borrowed/inspired/copied from cambridge course on cuda
http://www.many-core.group.cam.ac.uk/course/
*/
#ifndef _PredPrey_H_
#define _PredPrey_H_

// Thread block size
#define Block_Size_X	16		// 16	 
#define Block_Size_Y	16		// 16

// Number of blocks
/* I define the dimensions of the matrix as product of two numbers
Makes it easier to keep them a multiple of something (16, 32) when using CUDA*/
#define Block_Number_X	64		// 64
#define Block_Number_Y	64		// 64

// Matrix dimensions
// (chosen as multiples of the thread block size for simplicity)
#define WidthGrid (Block_Number_X * Block_Size_X)     // Matrix  width
#define HeightGrid (Block_Number_Y * Block_Size_Y)    // Matrix  height

// DIVIDE_INTO(x/y) for integers, used to determine # of blocks/warps etc.
#define DIVIDE_INTO(x,y) (((x) + (y) - 1)/(y))

// Definition of spatial parameters
#define dX			2		//  5     The size of each grid cell in X direction
#define dY			2		//  5     The size of each grid cell in Y direction 

// Process parameters            Original value    Explanation and Units
#define DifPrey		1		//  1        
#define DifPred		1		//  4        
    
#define A			3.5		//  3.5
#define B			1.2		//  1.2
#define C			4.9		//  4.9 / 6

#define dT	        0.05	// Timestep
#define Time		0		// Start time
#define NumFrames	1000	// The number of time the results are stored
#define	MAX_STORE	(NumFrames+2)	// Some more space than needed
#define EndTime		1000	// The time at which the simulation ends

// Name definitions
#define PREY		101
#define PREDATOR	102
#define HORIZONTAL	201
#define VERTICAL	202

#endif // _PredPrey_H_

