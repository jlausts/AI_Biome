

#include "Variables.h"



/*

git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPO.git
git config --global user.name "Your Name"
git config --global user.email "your-email@example.com"

uint32_t bias_indexes   [NUM_LAYERS  ];
uint32_t weight_indexes [NUM_LAYERS  ];
uint32_t o_prime_indexes[NUM_LAYERS-1];



DT avg_error[NUM_THREADS];

DT     inputs_f [MAX_EXAMPLES][NUM_INPUT_NODES];
DT     outputs_f[MAX_EXAMPLES][NUM_OUTPUT_NODES];


uint32_t array_using[NUM_NODES];

DT offsets[MAX_EXAMPLES][NUM_OUTPUT_NODES];


DT weights                  [NUM_CONNECTIONS];
DT weight_der  [NUM_THREADS][NUM_CONNECTIONS];
DT ema_weight_gradient      [NUM_CONNECTIONS];
DT ema_weight_sum_of_squares[NUM_CONNECTIONS];
DT weight_der_sum           [NUM_CONNECTIONS];
 

DT biases                     [NUM_NODES];
DT bias_der      [NUM_THREADS][NUM_NODES];
DT ema_bias_gradient          [NUM_NODES];
DT ema_bias_sum_of_squares    [NUM_NODES];
DT arr_activated [NUM_THREADS][NUM_NODES-NUM_OUTPUT_NODES];
DT arr           [NUM_THREADS][NUM_NODES];
DT bias_der_sum               [NUM_NODES];


DT num_examples_mult;
DT costs[NUM_THREADS];


*/



uint32_t *bias_indexes;// [NUM_LAYERS  ];
uint32_t *weight_indexes;// [NUM_LAYERS  ];



DT *avg_error;//[NUM_THREADS];

DT     **inputs_f ;//[MAX_EXAMPLES][NUM_INPUT_NODES];
DT     **outputs_f;//[MAX_EXAMPLES][NUM_OUTPUT_NODES];


uint32_t *array_using;//[NUM_NODES];

DT **offsets;//[MAX_EXAMPLES][NUM_OUTPUT_NODES];


DT *weights;//                  [NUM_CONNECTIONS];
DT **weight_der;//  [NUM_THREADS][NUM_CONNECTIONS];
DT *ema_weight_gradient;//      [NUM_CONNECTIONS];
DT *ema_weight_sum_of_squares;//[NUM_CONNECTIONS];
DT *weight_der_sum;//          [NUM_CONNECTIONS];
 

DT *biases;//                     [NUM_NODES];
DT **bias_der;//      [NUM_THREADS][NUM_NODES];
DT *ema_bias_gradient;//          [NUM_NODES];
DT *ema_bias_sum_of_squares;//    [NUM_NODES];
DT **arr_activated;// [NUM_THREADS][NUM_NODES-NUM_OUTPUT_NODES];
DT **arr;//           [NUM_THREADS][NUM_NODES];
DT *bias_der_sum;//              [NUM_NODES];


DT num_examples_mult;
DT *costs;//[NUM_THREADS];

DT NORMALIZER_DER;
DT NORMALIZER_SUM;


uint32_t *network_geometry;//[NUM_LAYERS] = {NUM_INPUT_NODES, 8, 8, 8, 8, 8, 8, NUM_OUTPUT_NODES};


bool *threads_wait;//[NUM_THREADS];
bool exit_thread = false;
bool *thread_done;//[NUM_THREADS];
extern bool first_derivative_sum;
extern bool save_results;

uint32_t *tmp_network_geometry_threaded;//[NUM_LAYERS - 1];

Activator *activationFunctions ;// [NUM_LAYERS];
Activator *activationDerivatives;//[NUM_LAYERS];


void memsetAll(const int numConnections, const int numNodes, const int numThreads, const int numInputNodes, const int numExamples, const int numOutputNodes, const int numLayers)
{
    MALLOC_SAFE(tmp_network_geometry_threaded, numLayers - 1, uint32_t);

    MALLOC_SAFE(activationFunctions, numLayers, Activator);
    MALLOC_SAFE(activationDerivatives, numLayers, Activator);

    MALLOC_SAFE(threads_wait, numThreads, bool);
    MALLOC_SAFE(thread_done, numThreads, bool);

    MALLOC_SAFE(network_geometry, numLayers, uint32_t);

    MALLOC_SAFE(bias_indexes, numLayers, uint32_t);
    MALLOC_SAFE(weight_indexes, numLayers, uint32_t);

    MALLOC_SAFE(avg_error, numThreads, DT);

    MALLOC_SAFE(inputs_f, numExamples, DT*);
    MALLOC_SAFE(outputs_f, numExamples, DT*);
    MALLOC_SAFE(offsets, numExamples, DT*);
    for (int i = 0; i < numExamples; ++i)
    {
        MALLOC_SAFE(inputs_f[i], numInputNodes, DT);
        MALLOC_SAFE(outputs_f[i], numOutputNodes, DT);
        MALLOC_SAFE(offsets[i], numOutputNodes, DT);
    }

    
    MALLOC_SAFE(array_using, numNodes, uint32_t);
    
    MALLOC_SAFE(weights, numConnections, DT);
    MALLOC_SAFE(ema_weight_gradient, numConnections, DT);
    MALLOC_SAFE(ema_weight_sum_of_squares, numConnections, DT);
    MALLOC_SAFE(weight_der_sum, numConnections, DT);

    MALLOC_SAFE(biases, numNodes, DT);
    MALLOC_SAFE(ema_bias_gradient, numNodes, DT);
    MALLOC_SAFE(ema_bias_sum_of_squares, numNodes, DT);
    MALLOC_SAFE(bias_der_sum, numNodes, DT);

    MALLOC_SAFE(weight_der, numThreads, DT*);
    MALLOC_SAFE(bias_der, numThreads, DT*);
    MALLOC_SAFE(arr_activated, numThreads, DT*);
    MALLOC_SAFE(arr, numThreads, DT*);
    for (int i = 0; i < numThreads; ++i)
    {
        MALLOC_SAFE(weight_der[i], numConnections, DT);
        MALLOC_SAFE(bias_der[i], numNodes, DT);
        MALLOC_SAFE(arr_activated[i], (numNodes - numOutputNodes), DT);
        MALLOC_SAFE(arr[i], numNodes, DT);
    }

    MALLOC_SAFE(costs, numThreads, DT);
}

void freeAll(const int numThreads, const int numExamples)
{
    
    FREE_SAFE(tmp_network_geometry_threaded);
    FREE_SAFE(activationFunctions);
    FREE_SAFE(activationDerivatives);

    FREE_SAFE(threads_wait);
    FREE_SAFE(thread_done);

    FREE_SAFE(network_geometry);

    FREE_SAFE(bias_indexes);
    FREE_SAFE(weight_indexes);

    FREE_SAFE(avg_error);

    FREE_SAFE(inputs_f);
    FREE_SAFE(outputs_f);
    FREE_SAFE(offsets);
    for (int i = 0; i < numExamples; ++i)
    {
        FREE_SAFE(inputs_f[i]);
        FREE_SAFE(outputs_f[i]);
        FREE_SAFE(offsets[i]);
    }
    
    FREE_SAFE(array_using);
    
    FREE_SAFE(weights);
    FREE_SAFE(weight_der);
    FREE_SAFE(ema_weight_gradient);
    FREE_SAFE(ema_weight_sum_of_squares);
    FREE_SAFE(weight_der_sum);


    FREE_SAFE(weight_der);
    FREE_SAFE(bias_der);
    FREE_SAFE(arr_activated);
    FREE_SAFE(arr);

    FREE_SAFE(biases);
    FREE_SAFE(ema_bias_gradient);
    FREE_SAFE(ema_bias_sum_of_squares);
    FREE_SAFE(bias_der_sum);

    for (int i = 0; i < numThreads; ++i)
    {
        FREE_SAFE(weight_der[i]);
        FREE_SAFE(bias_der[i]);
        FREE_SAFE(arr_activated[i]);
        FREE_SAFE(arr[i]);
    }

    FREE_SAFE(costs);
}







void setActivationFunction()
{
    for (uint8_t i = 0; i < NUM_LAYERS; ++i)
    {
        activationFunctions  [i] = mySwoosh2;
        activationDerivatives[i] = mySwooshDerivative2;
    }
}

clock_t s;
void showTime(const int line)
{
    const clock_t now = clock();
    printf("%d %fs\n", line, (float)(now - s) / (float)CLOCKS_PER_SEC);
    s = now;
}




void calculateConnections()
{
    uint32_t num_connections = (uint32_t)network_geometry[NUM_LAYERS-1] * (uint32_t)network_geometry[NUM_LAYERS-2];
    uint32_t num_biases = (uint32_t) network_geometry[NUM_LAYERS-1];
    for (int i = 1; i < NUM_LAYERS-1; i++)
    {
        num_connections += (uint32_t)network_geometry[i] * (uint32_t)network_geometry[i - 1];
        num_biases = (uint32_t) network_geometry[i] + num_biases;
    }

    printf("\n#define NUM_CONNECTIONS %d", num_connections);
    printf("\n#define NUM_NODES       %d", num_biases);

    exit(0);
}
    



void oneEpoch(
    uint8_t tid, 
    const uint32_t num_examples, 
    const float dropout_rate, 
    const propagationType propagate, 
    uint32_t *tmp_network_geometry,
    const DT weight_normalizer,
    const bool using_dropout,
    const int numLayers,
    const int numThreads)
{
    struct timeval start, end;
    const struct timespec req = {0, 10000000L}; 


    // Get the start time
    gettimeofday(&start, NULL);

    avg_error[0] = 0.0f;

    costs[tid] = 0;

    // memset(bias_der_sum, 0, sizeof(bias_der_sum));
    // memset(weight_der_sum, 0, sizeof(weight_der_sum));

    NORMALIZER_DER = weight_normalizer;
    NORMALIZER_SUM = weight_normalizer * 0.5f;

    if (using_dropout)
        decideWhichNodesToUse(dropout_rate, tmp_network_geometry, network_geometry);

    setThreadLocalVariables(tmp_network_geometry, propagate, num_examples);

    // start the other threads
    memset(threads_wait, 0, (size_t)numThreads * sizeof(bool));
    // saveState(false);


    for (uint32_t i = 0; i < num_examples; i += NUM_THREADS)
    {
        // if ((i)%(NUM_THREADS*4000) == 0)
        // printf("%6.0d  of  %6.0d  %6.3f %6.3f %6.3f %6.3f %6.3f                \r", 
        //     i, num_examples, inputs_f[i][0], inputs_f[i][1], inputs_f[i][2], inputs_f[i][3], costs[0]),
        //     fflush(stdout); 

        propagate(tid, i, network_geometry, activationFunctions, activationDerivatives, tmp_network_geometry, &avg_error[tid], numLayers);
        LOOK_ERROR;
    }

    // wait for the other threads to finish before altering the weights and biases
    bool wait_for_threads = true;
    while (wait_for_threads)
    {
        wait_for_threads = false;
        for (int i = 1; i < NUM_THREADS; ++i)
        {
            if (thread_done[i] == false)
            {
                wait_for_threads = true;

                // wait for 10 ms
                nanosleep(&req, NULL);
                break;
            }
        }
    }

    // the first time the derivative get summed, the value 'sets' the array.
    // the next time, ithe derivatives get added onto that.
    first_derivative_sum = true;

    // tell the thread that is made to save the progress to save.
    save_results = true;

    // sum the average errors that were calculated over the threads.
    // and the costs.
    DT err_sum = avg_error[0];
    DT cost_sum = 0;
    for (int i = 0; i < NUM_THREADS; ++i)
        cost_sum += costs[i];

    LOOK_ERROR;
    
    gettimeofday(&end, NULL);
    static float preCost = 100;

    double elapsed = (double)(end.tv_sec - start.tv_sec) + ((double)(end.tv_usec - start.tv_usec)/1000000.0);
    printf("cost %15.10f err %15.10f  %2.5lfs     %f            \n", cost_sum / (DT)num_examples, err_sum / (DT)num_examples / NUM_OUTPUT_NODES, elapsed, ((cost_sum / (DT)num_examples) / preCost - 1) * 1e6);
    preCost = cost_sum / (DT)num_examples;

    LOOK_ERROR;
    changeLearningRate();
    LOOK_ERROR;

    if (using_dropout)
        adjustNetworkWithDropout(network_geometry, tmp_network_geometry);
    else
        adjustNetwork();
        
    LOOK_ERROR;
}




void doEpochs(const uint32_t epoch_count, const uint32_t num_examples, const int numLayers)
{
    const uint8_t 
        regular          __attribute__((unused)) = 0, 
        dropOut          __attribute__((unused)) = 3;

    float dropout_rate = 1.0f;
    uint32_t *tmp_network_geometry;//[NUM_LAYERS - 1];
    MALLOC_SAFE(tmp_network_geometry, numLayers - 1, uint32_t);

    propagationType propagations[2] = {
        cycleNetworkNoDecay, 
        cycleNetworkDropout, 
        };


    DT weight_normalizer = 0.0f;

    uint32_t i = 0;
    do
    {
        oneEpoch(0, num_examples, dropout_rate, propagations[regular], tmp_network_geometry, weight_normalizer, false, numLayers, NUM_THREADS);
        
    } while (++i < epoch_count);

    FREE_SAFE(tmp_network_geometry);
}

#define MAX_DATA_LEN 1794000/4

float absoluteDiff[128][MAX_DATA_LEN]; 
float lineDiff[128][MAX_DATA_LEN];
float slope[128][MAX_DATA_LEN];
float minute[MAX_DATA_LEN];
float answers[MAX_DATA_LEN];

int main()
{

    //calculateConnections();


    puts("Starting...");
    initializeThreads();
    

    srand(0);
    setActivationFunction();

    uint32_t num_examples;
    getFileData("TSLA", absoluteDiff, lineDiff, slope, minute, answers, &num_examples);
    num_examples /= 10;

    determineWhoDoesWhatAndArrayIndexes(network_geometry, num_examples);
    
    //initializeWeights(network_geometry);

    //getState();


    const uint32_t epochs = 100;
    PR2("%d", num_examples);

    doEpochs(epochs, num_examples, NUM_LAYERS);

    LOOK_ERROR;
    
    //saveNetworkState(biases      , "State/biases__.bin"  , NUM_NODES);
    //saveNetworkState(weights     , "State/weights__.bin" , NUM_CONNECTIONS);

}


