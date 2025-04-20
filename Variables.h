#ifndef VARIABLES
#define VARIABLES


typedef float DT;
typedef int16_t DT_INT; 


#define LARGE
#define MEDIUM_



//#define NUM_THREADS 12
#define NUM_THREADS 5


#ifdef LARGE

#define WIDEST_LAYER 32
#define SECOND_WIDEST_LAYER 32

#define NUM_LAYERS 8

#define NUM_OUTPUT_NODES  1
#define NUM_INPUT_NODES  4

#define NUM_CONNECTIONS 360
#define NUM_NODES       49

#elif defined(MEDIUM)


#define NUM_LAYERS 8
#define WIDEST_LAYER 10
#define SECOND_WIDEST_LAYER 10

#define NUM_OUTPUT_NODES 5
#define NUM_INPUT_NODES  6

#define NUM_CONNECTIONS 438
#define NUM_NODES       54

#else
#define NUM_LAYERS 4
#define WIDEST_LAYER 4
#define SECOND_WIDEST_LAYER 3

#define NUM_OUTPUT_NODES 2
#define NUM_INPUT_NODES  2

#define NUM_CONNECTIONS 26
#define NUM_NODES       9
#endif



#define MAX_FILE 1794000/4


typedef DT (*Activator) (DT);

/*
    contains the type of propagation functions
    0 -> no weight decay or dropout
    1 -> only weight decay and bias decay
    2 -> weight decay and dropout
*/
typedef void (*propagationType)(
    uint8_t tid, 
    uint32_t example_num, 
    const uint32_t *network_geometry, 
    const Activator *activationFunctions, 
    const Activator *activationDerivatives, 
    uint32_t *tmp_network_geometry,
    DT *average_error,
    const int numLayers);



#ifdef __linux__
                      
#define training_data "/media/pi/40F8EEB9F8EEAC7A/Users/jlaus/Documents/Programming/Data/Normalized/"

#else

#define training_data "C:/Users/jlaus/Documents/Programming/Data/TrainingData/Normalized/"

#endif



 
#include "Err_Check/Err_Check.h"
#include "Propagation/Propagation.h"
#include "SetupThreads/SetupThreads.h"
#include "Network_Usage/Network_Usage.h"
#include "Adjust_Network/Adjust_Network.h"
#include "Sum_Derivatives/Sum_Derivatives.h"
#include "SaveNetworkState/SaveNetworkState.h"
#include "Initialize_Network/Initialize_Network.h"
#include "Propagation_Dropout/Propagation_Dropout.h"
#include "Activation_Functions/Activation_Functions.h"
#include "Verify_Partial_Derivative/Verify_Partial_Derivative.h"
//#define DEBUGGING







// neural network geometry
// including input and output layers

extern uint32_t *bias_indexes   ;//[NUM_LAYERS  ];
extern uint32_t *weight_indexes ;//[NUM_LAYERS  ];








#define MAX_EXAMPLES    MAX_FILE

extern DT *avg_error;//[NUM_THREADS];

extern DT     **inputs_f ;//[MAX_EXAMPLES][NUM_INPUT_NODES];
extern DT     **outputs_f;//[MAX_EXAMPLES][NUM_OUTPUT_NODES];
// DT_INT inputs_i [MAX_EXAMPLES][NUM_INPUT_NODES];
// DT_INT outputs_i[MAX_EXAMPLES][NUM_INPUT_NODES];


extern uint32_t *array_using;//[NUM_NODES];

extern DT **offsets;//[MAX_EXAMPLES][NUM_OUTPUT_NODES];


extern DT *weights ;//                 [NUM_CONNECTIONS];
extern DT **weight_der ;//  [NUM_THREADS][NUM_CONNECTIONS];
extern DT *ema_weight_gradient ;//     [NUM_CONNECTIONS];
extern DT *ema_weight_sum_of_squares;//[NUM_CONNECTIONS];
extern DT *weight_der_sum   ;//        [NUM_CONNECTIONS];
 

extern DT *biases  ;//                   [NUM_NODES];
extern DT **bias_der  ;//    [NUM_THREADS][NUM_NODES];
extern DT *ema_bias_gradient  ;//        [NUM_NODES];
extern DT *ema_bias_sum_of_squares  ;//  [NUM_NODES];
extern DT **arr_activated ;//[NUM_THREADS][NUM_NODES-NUM_OUTPUT_NODES];
extern DT **arr          ;// [NUM_THREADS][NUM_NODES];
extern DT *bias_der_sum   ;//            [NUM_NODES];


extern DT num_examples_mult;
extern DT *costs;//[NUM_THREADS];

extern DT NORMALIZER_DER;
extern DT NORMALIZER_SUM;

#endif