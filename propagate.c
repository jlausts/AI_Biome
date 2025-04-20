
#include "macros.h"





#define RELU(a) (a > 0 ? a : 0)
#define RELU_DER(a) (a > 0)
#define ABS(a) (a > 0 ? a : -a)







#define NUM_INPUTS 2
#define LAYER_WIDTH 2
#define NUM_LAYERS 4
#define NUM_OUTPUTS 2

#define NUM_WEIGHTS ((NUM_INPUTS * LAYER_WIDTH) + ((NUM_LAYERS-2 ) * LAYER_WIDTH * LAYER_WIDTH) + (NUM_OUTPUTS * LAYER_WIDTH))
#define NUM_NODES (NUM_INPUTS + ((NUM_LAYERS-2) * LAYER_WIDTH) + NUM_OUTPUTS)

#define NORMALIZER_DER 0.001f
#define NORMALIZER_SUM (NORMALIZER_DER * 0.5f)


#define NUM_EXAMPLES 10














#define B1 0.9f
#define B2 0.999f

const float beta1   = B1;
const float beta2   = B2;
const float beta1_1 = 1.0f - B1;
const float beta2_1 = 1.0f - B2;

const float epsilon = 1.0e-6f;
float learning_rate = 0.0001f;

typedef struct 
{
    const float beta1;
    const float beta2;
    const float beta1_1;
    const float beta2_1;
    const float epsilon;
    float learning_rate;
    int num_examples;
    float num_examples_mult;
}
Hyper;


typedef struct
{
    float der_sum;
    float ema_gradient;
    float ema_sum_of_squares;
}
Adam_parms;


typedef struct 
{
    Hyper hyper;
    Adam_parms bias[NUM_NODES];
    Adam_parms weight[NUM_WEIGHTS];
}
Adam;

typedef struct
{
    float nn[NUM_NODES];
    float activated_nodes[NUM_NODES];
    float bias_der[NUM_NODES * NUM_EXAMPLES];
    float weight_der[NUM_WEIGHTS * NUM_EXAMPLES];
}
Calculations;

typedef struct
{
    int bias[NUM_NODES];
    int weight[NUM_WEIGHTS];
}
Index;



typedef struct
{
    float cost;
    float bias[NUM_NODES];
    float weight[NUM_WEIGHTS];
    float input_data[NUM_INPUTS * NUM_EXAMPLES];
    float answers[NUM_OUTPUTS * NUM_EXAMPLES];
    Index indexer;
    Calculations results[NUM_EXAMPLES];
    Adam adam;
}
NN;


void printOutput(const float *const info)
{
    char things[NUM_OUTPUTS][16] = {0};
    char all[NUM_OUTPUTS*16 + 1] = {0};

    for (int i = 0, c = 0; i < NUM_OUTPUTS; ++i)
    {
        sprintf(things[i], "%5.2e", info[i]);
        for (int j = 0; j < 16 && things[i][j] != '\0'; ++j)
            all[c++] = things[i][j];
        all[c++] = ' ';
    }
    PR("OUTPUT: %s", all);
}



bool verifyFloats(const int line, const char *file)
{
    bool hasError = false;

    if (fetestexcept(FE_DIVBYZERO)) 
    {
        printf("%s:%d Error: Division by Zero.\n", file, line);
        hasError = true;
    }

    if (fetestexcept(FE_INVALID)) 
    {
        printf("%s:%d Error: Invalid Operation - Result of the operation is undefined .\n", file, line);
        hasError = true;
    }

    if (fetestexcept(FE_OVERFLOW)) 
    {
        printf("%s:%d Error: Overflow - Result of the operation is too large to be represented by the floating-point type.\n", file, line);
        hasError = true;
    }
    if (hasError)
    {
        feclearexcept(FE_ALL_EXCEPT);
        exit(1);
    }

    return hasError;
}

#define FLOATS verifyFloats(__LINE__, __FILE__)



void adamOptimization(float *const update_value, Adam_parms *const adam, Hyper *hyper) 
{
    // Update the exponential moving averages
    adam->der_sum *= hyper->num_examples_mult;
    adam->ema_gradient = hyper->beta1 * adam->ema_gradient + hyper->beta1_1 * adam->der_sum;
    adam->ema_sum_of_squares = hyper->beta2 * adam->ema_sum_of_squares + hyper->beta2_1 * adam->der_sum * adam->der_sum;

    // Update the parameter
    const float sum_gradient = adam->ema_gradient        / hyper->beta1_1;
    const float sum_squares  = adam->ema_sum_of_squares  / hyper->beta2_1;
    *update_value -= hyper->learning_rate * sum_gradient / sqrtf(sum_squares + hyper->epsilon);
}



void determineWhoDoesWhatAndArrayIndexes(Index *const index)
{
    index->bias[0] = 0;
    index->weight[0] = 0;
    index->bias[1] = NUM_INPUTS;
    index->weight[1] = NUM_INPUTS * LAYER_WIDTH;

    for (uint8_t i = 2; i < NUM_LAYERS-1; ++i)
    {
        index->weight[i] = index->weight[i - 1] + LAYER_WIDTH * LAYER_WIDTH;
        index->bias[i] = index->bias[i - 1] + LAYER_WIDTH;
    }

    index->weight[NUM_LAYERS - 1] = index->weight[NUM_LAYERS - -2] + LAYER_WIDTH * NUM_OUTPUTS;
    index->bias[NUM_LAYERS - 1] = index->bias[NUM_LAYERS - -2] + NUM_OUTPUTS;
}




void adjustNetwork(NN *nn, Adam *adam)
{
    for (uint8_t i = 0; i < NUM_NODES; i ++)
    {
        adamOptimization(&nn->bias[i], &adam->bias[i], &adam->hyper);
        adamOptimization(&nn->weight[i], &adam->weight[i], &adam->hyper);
    }
  
    for (uint16_t i = NUM_NODES; i < NUM_WEIGHTS; i ++) 
        adamOptimization(&nn->weight[i], &adam->weight[i], &adam->hyper);
}





int var;



void sumDerivatives(Calculations *calculated, Adam* adam, const float num_examples)
{
    for (uint8_t i = 0; i < NUM_NODES; ++i)
    {
        adam->bias[i].der_sum = calculated->bias_der[i];
        adam->weight[i].der_sum = calculated->weight_der[i];
    }
    
    for (uint16_t i = NUM_NODES; i < NUM_WEIGHTS; ++i)
        adam->weight[i].der_sum = calculated->weight_der[i];

    
    for (int j = 1; j < num_examples; ++j)
    {
        const float *const bd = &calculated->bias_der[j * NUM_NODES];
        const float *const wd = &calculated->weight_der[j * NUM_WEIGHTS];

        for (uint8_t i = 0; i < NUM_NODES; ++i)
        {
            adam->bias[i].der_sum += bd[i];
            adam->weight[i].der_sum += wd[i];
        }
        for (uint16_t i = NUM_NODES; i < NUM_WEIGHTS; ++i)
            adam->weight[i].der_sum += wd[i];
    }
}





void matrixForwardNoGrad(float *const arr1, float *const arr2, const float *const arr1_weights, const float *const arr1_biases)
{
    arr1[0] += arr1_biases[0];
    const float act = RELU(arr1[0]);
    
    for (uint8_t index2 = 0; index2 < LAYER_WIDTH; ++index2)
        arr2[index2] = act * arr1_weights[index2];

    for (uint8_t index1 = 1; index1 < LAYER_WIDTH; ++index1)
    {
        arr1[index1] += arr1_biases[index1];
        const float activated = RELU(arr1[index1]);
        const float *const weight_ptr = &arr1_weights[index1 * LAYER_WIDTH];

        for (uint8_t index2 = 0; index2 < LAYER_WIDTH; ++index2)
            arr2[index2] += activated * weight_ptr[index2];
    }
    for (int i = 0; i < LAYER_WIDTH; ++i)
        PR("%f %f", arr1[i], arr2[i]);
}


void finalLayerNoGrad(float *const arr1, float *const answers, const float *const arr1_biases)
{
    for (uint8_t i = 0; i < NUM_OUTPUTS; ++i)
    {
        const float answer = arr1[i] + arr1_biases[i];
        answers[i] = RELU(answer);
    }   
}


void matrixForwardFirstLayer(const float *const arr1, float *const arr2, const float *const arr1_weights)
{
    for (uint8_t index2 = 0; index2 < LAYER_WIDTH; ++index2)
        arr2[index2] = arr1[0] * arr1_weights[index2];

    for (uint8_t index1 = 1; index1 < NUM_INPUTS; ++index1)
    {
        const float *const weight_ptr = &arr1_weights[index1 * LAYER_WIDTH];
        for (uint8_t index2 = 0; index2 < LAYER_WIDTH; ++index2)
            arr2[index2] += arr1[index1] * weight_ptr[index2];
    }

    for (int i = 0; i < LAYER_WIDTH; ++i)
        PR("%f %f", arr1[i], arr2[i]);
}


void matrixForward(float *const arr1, float *const arr2, float *const arr1_activated, const float *const arr1_weights, const float *const arr1_biases)
{
    arr1[0] += arr1_biases[0];
    arr1_activated[0] = RELU(arr1[0]);

    for (uint8_t index2 = 0; index2 < LAYER_WIDTH; ++index2)
        arr2[index2] = arr1_activated[0] * arr1_weights[index2];

    for (uint8_t index1 = 1; index1 < LAYER_WIDTH; ++index1)
    {
        arr1[index1] += arr1_biases[index1];
        arr1_activated[index1] = RELU(arr1[index1]);
        const float *const weight_ptr = &arr1_weights[index1 * LAYER_WIDTH];

        for (uint8_t index2 = 0; index2 < LAYER_WIDTH; ++index2)
            arr2[index2] += arr1_activated[index1] * weight_ptr[index2];
    }


    for (int i = 0; i < LAYER_WIDTH; ++i)
        PR("%f %f", arr1[i], arr2[i]);
}


void finalLayerReversal(float *const arr1, const float *const answers, const float *const arr1_biases, float *const cost, float *const bias_der, float *const output)
{
    for (uint8_t i = 0; i < NUM_OUTPUTS; ++i)
    {
        arr1[i] += arr1_biases[i];
        output[i] = RELU(arr1[i]);
        const float offset = output[i] - answers[i];
        bias_der[i] = (arr1[i] > 0 ? 2.0f * offset : 0); // relu derivative
        *cost += offset * offset;
    }   
}


void backProp(const float *const arr1, const float *const arr1_activated,float *const bias_der1, const float *const bias_der2, float *const weight_der1, const float *const arr1_weights)
{
    for (uint8_t index1 = 0, weight_using = 0; index1 < LAYER_WIDTH; ++index1, weight_using += LAYER_WIDTH)
    { 
        bias_der1  [index1]       = arr1_weights[weight_using] * bias_der2[0];
        weight_der1[weight_using] = arr1_activated[index1]     * bias_der2[0];
    }
        
    for (uint8_t index1 = 0; index1 < LAYER_WIDTH; ++index1)
    {
        for (uint8_t index2 = 1, weight_using = (uint8_t)(index1 * LAYER_WIDTH + 1); index2 < LAYER_WIDTH; ++index2, ++weight_using)
        { 
            bias_der1  [index1]       += arr1_weights[weight_using] * bias_der2[index2];
            weight_der1[weight_using]  = arr1_activated[index1]     * bias_der2[index2];
        }
        bias_der1[index1] *= RELU_DER(arr1[index1]);
    }
}


void backPropFirstLayer(const float *const arr1, const float *const bias_der2, float *const weight_der1)
{
    for (uint8_t i = 0, input_i = 0; i < NUM_INPUTS * LAYER_WIDTH; ++input_i)
        for (uint8_t j = 0; j < LAYER_WIDTH; ++j, ++i)
            weight_der1[i] = arr1[input_i] * bias_der2[j];
}


void cycleNetwork(NN *net, const uint8_t example)
{
    const float *const weight = net->weight;
    const float *const bias = net->bias; 
    float *const nn = net->results[example].nn;
    const int *const bi = net->indexer.bias;
    const int *const wi = net->indexer.weight;
    float *const bias_der = net->results[example].bias_der;
    float *const weight_der = net->results[example].weight_der;

    static float activated[NUM_NODES];

    matrixForwardFirstLayer(net->input_data, nn, weight);

    for (uint8_t i = 1; i < NUM_LAYERS - 1; ++i)
        matrixForward(&nn[bi[i-1]], &nn[bi[i]], &activated[bi[i-1]], &weight[wi[i]], &bias[bi[i-1]]);

    finalLayerReversal(&nn[bi[NUM_LAYERS - 2]], net->answers, &bias[bi[NUM_LAYERS - 2]], &net->cost, &bias_der[bi[NUM_LAYERS - 2]], &nn[bi[NUM_LAYERS - 1]]);

    for (int i = NUM_LAYERS - 2; i > 0; --i)
        backProp(&nn[bi[i-1]], &activated[bi[i-1]], &bias_der[bi[i-1]], &bias_der[bi[i]], &weight_der[wi[i]], &weight[wi[i]]); 

    backPropFirstLayer(net->input_data, bias_der, weight_der);

    for (uint32_t i = 0; i < NUM_WEIGHTS; ++i)
    {
        weight_der[i] += weight[i] * NORMALIZER_DER;
        net->cost += weight[i] * weight[i] * NORMALIZER_SUM;
    }
}



void cycleNetworkNoGrad(NN *net, const uint8_t example)
{
    const float *const weight = net->weight;
    const float *const bias = net->bias; 
    float *const nn = net->results[example].nn;
    const int *const bi = net->indexer.bias;
    const int *const wi = net->indexer.weight;

    matrixForwardFirstLayer(net->input_data, nn, weight);

    for (uint8_t i = 1; i < NUM_LAYERS - 1; ++i)
        matrixForwardNoGrad(&nn[bi[i-1]], &nn[bi[i]], &weight[wi[i]], &bias[bi[i-1]]);

    finalLayerNoGrad(&nn[bi[NUM_LAYERS - 2]], net->answers, &bias[bi[NUM_LAYERS - 2]]);
}

void set(float *arr, int c, float val)
{
    for (int i = 0; i < c; ++i)
    {
        arr[i] = val;
    }
}

void printNNState(const NN *nn, const int example)
{
    printf("\n===== NN STATE (Example %d) =====\n", example);

    // Input Layer
    printf("\n--- Input Layer ---\n");
    for (int i = 0; i < NUM_INPUTS; ++i)
        printf("input[%d] = %.2e\n", i, nn->input_data[example * NUM_INPUTS + i]);

    int node_idx = 0;
    int weight_idx = 0;

    // Hidden Layers
    for (int layer = 0; layer < NUM_LAYERS - 1; ++layer)
    {
        int num_neurons = LAYER_WIDTH;
        int num_weights = num_neurons * LAYER_WIDTH;

        printf("\n--- Hidden Layer %d ---\n", layer + 1);

        printf("  Biases:\n");
        for (int i = 0; i < num_neurons; ++i)
            printf("    bias[%d] = %.2e\n", node_idx + i, nn->bias[node_idx + i]);

        printf("  Weights:\n");
        for (int i = 0; i < num_weights; ++i)
            printf("    weight[%d] = %.2e\n", weight_idx + i, nn->weight[weight_idx + i]);

        printf("  Bias Der:\n");
        for (int i = 0; i < num_neurons; ++i)
            printf("    bias[%d] = %.2e\n", node_idx + i, nn->results[example].bias_der[i]);

        printf("  Weight Der:\n");
        for (int i = 0; i < num_weights; ++i)
            printf("    weight[%d] = %.2e\n", weight_idx + i, nn->results[example].weight_der[i]);
    
        printf("  Activations:\n");
        for (int i = 0; i < num_neurons; ++i)
            printf("    nn[%d] = %.2e\n", node_idx + i, nn->results[example].nn[node_idx + i]);

        node_idx += num_neurons;
        weight_idx += num_weights;
    }
    printf("Cost = %.5e\n", nn->cost);

    printf("\n===== END OF NN STATE =====\n");
}


int main()
{
    static NN nn = {
        .adam.hyper.beta1=B1, 
        .adam.hyper.beta2=B2, 
        .adam.hyper.beta1_1=1-B1, 
        .adam.hyper.beta2_1=1-B2, 
        .adam.hyper.epsilon=1.0e-6f, 
        .adam.hyper.learning_rate=1e-4f
    };

    determineWhoDoesWhatAndArrayIndexes(&nn.indexer);

    set(nn.input_data, 20, 1);
    set(nn.weight, 16, 1);
    set(nn.bias, 8, 0);

    // nn.weight[0] = 0.999f;

    cycleNetwork(&nn, 0);

    printNNState(&nn, 0);

    return 0;
}