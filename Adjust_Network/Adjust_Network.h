#ifndef ADJUST_NETWORK
#define ADJUST_NETWORK

#include "../Variables.h"





void adjustNetwork(void);


// use this to adjust the network if dropout is being used.
void adjustNetworkWithDropout(const uint32_t network_geometry[NUM_LAYERS], uint32_t tmp_network_geometry[NUM_LAYERS - 1]);





// run this one time at the start
void determineWhoDoesWhatAndArrayIndexes(const uint32_t network_geometry[NUM_LAYERS], const uint32_t num_examples);


#endif