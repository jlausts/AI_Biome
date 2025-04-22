
#include "macros.h"





#define RELU(a) (a > 0 ? a : 0)
#define RELU_DER(a) (a > 0)
#define ABS(a) (a > 0 ? a : -a)







#define NUM_INPUTS 2
#define LAYER_WIDTH 2
#define NUM_LAYERS 4
#define NUM_OUTPUTS 2

#define NUM_WEIGHTS ((NUM_INPUTS * LAYER_WIDTH) + ((NUM_LAYERS-3 ) * LAYER_WIDTH * LAYER_WIDTH) + (NUM_OUTPUTS * LAYER_WIDTH))
#define NUM_NODES (NUM_INPUTS + ((NUM_LAYERS-2) * LAYER_WIDTH) + NUM_OUTPUTS)

#define NORMALIZER_DER 0.001f
#define NORMALIZER_SUM (NORMALIZER_DER * 0.5f)



#define B1 0.9f
#define B2 0.999f
#define NUM_EXAMPLES 10

#define FLOATS verifyFloats(__LINE__, __FILE__)


#define PRINTINGh
#define PR_ADAMd


// change the main datatype to "float", "double"
typedef double         f;
typedef f *const       fc_p;
typedef const f        cf;
typedef const f *const cfc_p;



typedef struct 
{
    cf beta1;
    cf beta2;
    cf beta1_1;
    cf beta2_1;
    cf epsilon;
    f learning_rate;
    int num_examples;
    f num_examples_mult;
}
Hyper;

typedef struct
{
    f der_sum;
    f ema_gradient;
    f ema_sum_of_squares;
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
    f nn[NUM_NODES];
    f activated_nodes[NUM_NODES];
    f bias_der[NUM_NODES * NUM_EXAMPLES];
    f weight_der[NUM_WEIGHTS * NUM_EXAMPLES];
}
Calculations;

typedef struct
{
    int bias[NUM_LAYERS];
    int weight[NUM_LAYERS];
}
Index;

typedef struct
{
    f cost;
    f offset;
    f bias[NUM_NODES];
    f weight[NUM_WEIGHTS];
    f input_data[NUM_INPUTS * NUM_EXAMPLES];
    f answers[NUM_OUTPUTS * NUM_EXAMPLES];
    Index indexer;
    Calculations results[NUM_EXAMPLES];
    Adam adam;
}
NN;






void fillArrayWithRandomGaussian(fc_p array, const uint32_t length, cf variance, cf mult) 
{
    for (uint32_t i = 0; i < length; i += 2) 
    {
        // Generate two uniform random numbers between 0 and 1
        cf u1 = sqrtf(-2.0f *  logf( ((f)rand() + 1.0f) / ((f)RAND_MAX + 3.0f) )) * variance;
        cf u2 =                      ((f)rand() + 1.0f) / ((f)RAND_MAX + 3.0f)    * 6.28318530717958f;

        // Scale by standard deviation and shift by mean
        array[i] = u1 * cosf(u2) * mult;

        if (array[i] > 1000.0f)
        {
            PR("Something wrong with network initialization\n%f %d %f %f %f", array[i], i, u1, u2, sqrtf(variance));
            exit(EXIT_FAILURE);
        }

        // Fill second number if we have space
        if (i + 1 < length) 
            array[i + 1] = u1 * sinf(u2) * mult;
    }

    // get max
    f max = array[0];
    for (uint32_t i = 1; i < length; ++i)
        if (array[i] > max)
            max = array[i];

    // re scale the weights to fit inside of the variance
    for (uint32_t i = 0; i < length; ++i)
        array[i] = array[i] * variance / max;
}


void initNetwork(NN *const net)
{
    cf var = sqrtf(6.0f / (f) (NUM_INPUTS + LAYER_WIDTH));
    fillArrayWithRandomGaussian(net->weight, NUM_INPUTS * LAYER_WIDTH, var, 1.0f);
    fillArrayWithRandomGaussian(net->bias, NUM_INPUTS, var, 0.1f);

    cf variance = sqrtf(6.0f / (f) (2 * LAYER_WIDTH));
    for (int layer = 1; layer < NUM_LAYERS; ++layer)
    {
        fillArrayWithRandomGaussian(&net->weight[net->indexer.weight[layer]], LAYER_WIDTH * LAYER_WIDTH, variance, 1.0f);
        fillArrayWithRandomGaussian(&net->bias[net->indexer.bias[layer]], LAYER_WIDTH, variance, 0.1f);
    }
}


void printOutput(cfc_p info)
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


void determineWhoDoesWhatAndArrayIndexes(NN *const net)
{
    Index *const index = &net->indexer;
    net->adam.hyper.num_examples_mult = 1.0f / (f)net->adam.hyper.num_examples;
    index->bias[0] = 0;
    index->weight[0] = 0;
    index->bias[1] = NUM_INPUTS;
    index->weight[1] = NUM_INPUTS * LAYER_WIDTH;

    for (uint8_t i = 2; i < NUM_LAYERS-1; ++i)
    {
        index->weight[i] = index->weight[i - 1] + LAYER_WIDTH * LAYER_WIDTH;
        index->bias[i] = index->bias[i - 1] + LAYER_WIDTH;
    }

    index->weight[NUM_LAYERS - 1] = index->weight[NUM_LAYERS - 2] + LAYER_WIDTH * NUM_OUTPUTS;
    index->bias[NUM_LAYERS - 1] = index->bias[NUM_LAYERS - 2] + NUM_OUTPUTS;

}


void adamOptimization(fc_p update_value, Adam_parms *const adam, Hyper *hyper) 
{
    // Update the exponential moving averages
    adam->der_sum *= hyper->num_examples_mult;
    adam->ema_gradient = hyper->beta1 * adam->ema_gradient + hyper->beta1_1 * adam->der_sum;
    adam->ema_sum_of_squares = hyper->beta2 * adam->ema_sum_of_squares + hyper->beta2_1 * adam->der_sum * adam->der_sum;

    // Compute bias-corrected terms
    cf sum_gradient = adam->ema_gradient        / hyper->beta1_1;
    cf sum_squares  = adam->ema_sum_of_squares  / hyper->beta2_1;

#ifdef PR_ADAM
    PR("\n--- ADAM OPTIMIZATION DEBUG ---");
    PR("update_value         = %.6e", *update_value);
    PR("der_sum              = %.6e", adam->der_sum);
    PR("ema_gradient         = %.6e", adam->ema_gradient);
    PR("ema_sum_of_squares   = %.6e", adam->ema_sum_of_squares);
    PR("sum_gradient         = %.6e", sum_gradient);
    PR("sum_squares          = %.6e", sum_squares);
    PR("num_examples_mult    = %.6e", hyper->num_examples_mult);
#endif

    // Apply the update
    *update_value -= hyper->learning_rate * sum_gradient / sqrtf(sum_squares + hyper->epsilon);
}


void adjustNetwork(NN *const net)
{
    Adam *const adam = &net->adam;
    for (uint8_t i = 0; i < NUM_NODES; i ++)
    {
 //       PR("%f %d", net->bias[i], i);
        adamOptimization(&net->bias[i], &adam->bias[i], &adam->hyper);
//        PR("%f %d", net->bias[i], i);
        adamOptimization(&net->weight[i], &adam->weight[i], &adam->hyper);
    }
  
    for (uint16_t i = NUM_NODES; i < NUM_WEIGHTS; i ++) 
        adamOptimization(&net->weight[i], &adam->weight[i], &adam->hyper);
}


void sumDerivatives(Calculations *const calculated, Adam* adam, const int num_examples)
{
    // set the arrays first
    cfc_p bd1 = calculated[0].bias_der;
    cfc_p wd1 = calculated[0].weight_der;
    for (uint8_t i = 0; i < NUM_NODES-2; ++i)
    {
        adam->bias[i].der_sum = bd1[i];
        adam->weight[i].der_sum = wd1[i];
    }
    
    for (uint16_t i = NUM_NODES-2; i < NUM_WEIGHTS; ++i)
        adam->weight[i].der_sum = wd1[i];

    
    // accumulate
    for (int j = 1; j < num_examples; ++j)
    {
        cfc_p bd = calculated[j].bias_der;
        cfc_p wd = calculated[j].weight_der;

        for (uint8_t i = 0; i < NUM_NODES-2; ++i)
        {
            adam->bias[i].der_sum += bd[i];
            adam->weight[i].der_sum += wd[i];
        }
        for (uint16_t i = NUM_NODES-2; i < NUM_WEIGHTS; ++i)
            adam->weight[i].der_sum += wd[i];
    }
}


void matrixForwardNoGrad(fc_p arr1, fc_p arr2, cfc_p arr1_weights, cfc_p arr1_biases)
{
    arr1[0] += arr1_biases[0];
    cf act = RELU(arr1[0]);
    
    for (uint8_t index2 = 0; index2 < LAYER_WIDTH; ++index2)
        arr2[index2] = act * arr1_weights[index2];

    for (uint8_t index1 = 1; index1 < LAYER_WIDTH; ++index1)
    {
        arr1[index1] += arr1_biases[index1];
        cf activated = RELU(arr1[index1]);
        cfc_p weight_ptr = &arr1_weights[index1 * LAYER_WIDTH];

        for (uint8_t index2 = 0; index2 < LAYER_WIDTH; ++index2)
            arr2[index2] += activated * weight_ptr[index2];
    }

#ifdef PRINTING
    for (int i = 0; i < LAYER_WIDTH; ++i)
        PR("%f %f", arr1[i], arr2[i]);
#endif
}


void finalLayerNoGrad(fc_p arr1, fc_p answers, cfc_p arr1_biases)
{
    for (uint8_t i = 0; i < NUM_OUTPUTS; ++i)
    {
        cf answer = arr1[i] + arr1_biases[i];
        answers[i] = RELU(answer);
    }   
}


void matrixForwardFirstLayer(cfc_p arr1, fc_p arr2, cfc_p arr1_weights)
{
    for (uint8_t index2 = 0; index2 < LAYER_WIDTH; ++index2)
        arr2[index2] = arr1[0] * arr1_weights[index2];

    for (uint8_t index1 = 1; index1 < NUM_INPUTS; ++index1)
    {
        cfc_p weight_ptr = &arr1_weights[index1 * LAYER_WIDTH];
        for (uint8_t index2 = 0; index2 < LAYER_WIDTH; ++index2)
            arr2[index2] += arr1[index1] * weight_ptr[index2];
    }

#ifdef PRINTING
    PR("\n=========\n--- matrixForwardFirstLayer ---");

    for (int i = 0; i < NUM_INPUTS; ++i)
        PR("Input arr1[%d]          = %.4f", i, arr1[i]);

    for (int i = 0; i < NUM_INPUTS * LAYER_WIDTH; ++i)
        PR("Weight arr1_weights[%d] = %.4f", i, arr1_weights[i]);

    for (int i = 0; i < LAYER_WIDTH; ++i)
        PR("Output arr2[%d]         = %.4f", i, arr2[i]);
#endif
}


void matrixForward(fc_p arr1, fc_p arr2, fc_p arr1_activated, cfc_p arr1_weights, cfc_p arr1_biases)
{
    arr1[0] += arr1_biases[0];
    arr1_activated[0] = RELU(arr1[0]);

    for (uint8_t index2 = 0; index2 < LAYER_WIDTH; ++index2)
        arr2[index2] = arr1_activated[0] * arr1_weights[index2];

    for (uint8_t index1 = 1; index1 < LAYER_WIDTH; ++index1)
    {
        arr1[index1] += arr1_biases[index1];
        arr1_activated[index1] = RELU(arr1[index1]);
        cfc_p weight_ptr = &arr1_weights[index1 * LAYER_WIDTH];

        for (uint8_t index2 = 0; index2 < LAYER_WIDTH; ++index2)
            arr2[index2] += arr1_activated[index1] * weight_ptr[index2];
    }

#ifdef PRINTING
    PR("\n--- matrixForward ---");

    for (int i = 0; i < LAYER_WIDTH; ++i)
        PR("Pre-activation arr1[%d]      = %.4f", i, arr1[i]);

    for (int i = 0; i < LAYER_WIDTH; ++i)
        PR("Bias arr1_biases[%d]         = %.4f", i, arr1_biases[i]);

    for (int i = 0; i < LAYER_WIDTH; ++i)
        PR("Activated arr1_activated[%d] = %.4f", i, arr1_activated[i]);

    for (int i = 0; i < LAYER_WIDTH * LAYER_WIDTH; ++i)
        PR("Weight arr1_weights[%d]      = %.4f", i, arr1_weights[i]);

    for (int i = 0; i < LAYER_WIDTH; ++i)
        PR("Output arr2[%d]              = %.4f", i, arr2[i]);
#endif
}


void finalLayerReversal(fc_p arr1, cfc_p answers, cfc_p arr1_biases, fc_p cost, fc_p bias_der, fc_p output)
{

#ifdef PRINTING
    PR("\n--- finalLayerReversal ---");
#endif
    for (uint8_t i = 0; i < NUM_OUTPUTS; ++i)
    {
        arr1[i] += arr1_biases[i];
        output[i] = RELU(arr1[i]);
        cf offset = output[i] - answers[i];
        bias_der[i] = (arr1[i] > 0 ? 2.0f * offset : 0); // relu derivative
        *cost += offset * offset;

#ifdef PRINTING
    PR("Offset[%d]                       = %.4f", i, offset);
    PR("Cost[%d]                         = %.4f", i, offset * offset);
#endif
    }

#ifdef PRINTING
    for (int i = 0; i < NUM_OUTPUTS; ++i)
        PR("Pre-activation arr1[%d]      = %.4f", i, arr1[i]);

    for (int i = 0; i < NUM_OUTPUTS; ++i)
        PR("Bias arr1_biases[%d]         = %.4f", i, arr1_biases[i]);

    for (int i = 0; i < NUM_OUTPUTS; ++i)
        PR("Output after RELU output[%d] = %.4f", i, output[i]);

    for (int i = 0; i < NUM_OUTPUTS; ++i)
        PR("Answer answers[%d]           = %.4f", i, answers[i]);

    for (int i = 0; i < NUM_OUTPUTS; ++i)
        PR("Bias Derivative bias_der[%d] = %.4f", i, bias_der[i]);

        PR("Total Cost                   = %.6f", *cost);
#endif
}


void backProp(cfc_p arr1, cfc_p arr1_activated, fc_p bias_der1, cfc_p bias_der2, fc_p weight_der1, cfc_p arr1_weights)
{
    for (uint8_t index1 = 0, weight_using = 0; index1 < LAYER_WIDTH; ++index1, weight_using += LAYER_WIDTH)
    {
        bias_der1[index1] = arr1_weights[weight_using] * bias_der2[0];
        weight_der1[weight_using] = arr1_activated[index1] * bias_der2[0];
    }

    for (uint8_t index1 = 0; index1 < LAYER_WIDTH; ++index1)
    {
        for (uint8_t index2 = 1, weight_using = (uint8_t)(index1 * LAYER_WIDTH + 1); index2 < LAYER_WIDTH; ++index2, ++weight_using)
        {
            bias_der1[index1] += arr1_weights[weight_using] * bias_der2[index2];
            weight_der1[weight_using] = arr1_activated[index1] * bias_der2[index2];
        }
        bias_der1[index1] *= RELU_DER(arr1[index1]);
    }

#ifdef PRINTING
    PR("\n--- backProp ---");
    for (int i = 0; i < LAYER_WIDTH; ++i)
        PR("Pre-activation arr1[%d]                = %.4f", i, arr1[i]);

    for (int i = 0; i < LAYER_WIDTH; ++i)
        PR("Activated arr1_activated[%d]           = %.4f", i, arr1_activated[i]);

    for (int i = 0; i < LAYER_WIDTH; ++i)
        PR("Bias Derivative bias_der1[%d]          = %.4f", i, bias_der1[i]);

    for (int i = 0; i < LAYER_WIDTH; ++i)
        PR("Incoming Bias Derivative bias_der2[%d] = %.4f", i, bias_der2[i]);

    for (int i = 0; i < LAYER_WIDTH * LAYER_WIDTH; ++i)
        PR("Weight arr1_weights[%d]                = %.4f", i, arr1_weights[i]);

    for (int i = 0; i < LAYER_WIDTH * LAYER_WIDTH; ++i)
        PR("Weight Derivative weight_der1[%d]      = %.4f", i, weight_der1[i]);
#endif
}


void backPropFirstLayer(cfc_p arr1, cfc_p bias_der2, fc_p weight_der1)
{
    for (uint8_t i = 0, input_i = 0; i < NUM_INPUTS * LAYER_WIDTH; ++input_i)
        for (uint8_t j = 0; j < LAYER_WIDTH; ++j, ++i)
            weight_der1[i] = arr1[input_i] * bias_der2[j];

#ifdef PRINTING
    PR("\n--- backPropFirstLayer ---");
    for (int i = 0; i < NUM_INPUTS; ++i)
        PR("Input arr1[%d]                    = %.4f", i, arr1[i]);

    for (int j = 0; j < LAYER_WIDTH; ++j)
        PR("Bias Derivative bias_der2[%d]     = %.4f", j, bias_der2[j]);

    for (int i = 0; i < NUM_INPUTS * LAYER_WIDTH; ++i)
        PR("Weight Derivative weight_der1[%d] = %.4f", i, weight_der1[i]);
#endif
}


void cycleNetwork(NN *net, const uint8_t example)
{
    cfc_p weight = net->weight;
    cfc_p bias = net->bias; 
    fc_p nn = net->results[example].nn;
    const int *const bi = net->indexer.bias;
    const int *const wi = net->indexer.weight;
    fc_p bias_der = net->results[example].bias_der;
    fc_p weight_der = net->results[example].weight_der;

    static f activated[NUM_NODES];

    matrixForwardFirstLayer(net->input_data, nn, weight);

    for (uint8_t i = 1; i < NUM_LAYERS - 1; ++i)
        matrixForward(&nn[bi[i-1]], &nn[bi[i]], &activated[bi[i-1]], &weight[wi[i]], &bias[bi[i-1]]);

    finalLayerReversal(&nn[bi[NUM_LAYERS - 2]], &net->answers[example * NUM_OUTPUTS], &bias[bi[NUM_LAYERS - 2]], &net->cost, &bias_der[bi[NUM_LAYERS - 2]], &nn[bi[NUM_LAYERS - 1]]);

    for (int i = NUM_LAYERS - 2; i > 0; --i)
        backProp(&nn[bi[i-1]], &activated[bi[i-1]], &bias_der[bi[i-1]], &bias_der[bi[i]], &weight_der[wi[i]], &weight[wi[i]]); 

    backPropFirstLayer(net->input_data, bias_der, weight_der);

    net->offset = net->cost;
    for (uint32_t i = 0; i < NUM_WEIGHTS; ++i)
    {
        weight_der[i] += weight[i] * NORMALIZER_DER;
        net->cost += weight[i] * weight[i] * NORMALIZER_SUM;
    }
}


void cycleNetworkNoGrad(NN *net, const uint8_t example)
{
    cfc_p weight = net->weight;
    cfc_p bias = net->bias; 
    fc_p nn = net->results[example].nn;
    const int *const bi = net->indexer.bias;
    const int *const wi = net->indexer.weight;

    matrixForwardFirstLayer(net->input_data, nn, weight);

    for (uint8_t i = 1; i < NUM_LAYERS - 1; ++i)
        matrixForwardNoGrad(&nn[bi[i-1]], &nn[bi[i]], &weight[wi[i]], &bias[bi[i-1]]);

    finalLayerNoGrad(&nn[bi[NUM_LAYERS - 2]], net->answers, &bias[bi[NUM_LAYERS - 2]]);
}


void set(f *arr, int c, f val)
{
    for (int i = 0; i < c; ++i)
    {
        arr[i] = val;
    }
}


void printNNState(const NN *nn, const int example)
{
    PR("\n\n=====\n===== NN STATE (Example %d) =====", example);

    // Input Layer
    PR("\n--- Input Layer ---");
    for (int i = 0; i < NUM_INPUTS; ++i)
        PR("input[%d] = %f", i, nn->input_data[example * NUM_INPUTS + i]);

    int node_idx = 0;
    int weight_idx = 0;

    // Hidden + Output Layers
    for (int layer = 0; layer < NUM_LAYERS; ++layer)
    {
        // Skip biases for the first layer
        if (layer > 0)
        {
            PR("\n--- Layer %d ---", layer + 1);
            PR("  Biases:");
            for (int i = 0; i < LAYER_WIDTH; ++i)
                PR("    bias[%d] = %f", node_idx + i, nn->bias[node_idx + i]);

            PR("  Bias Der:");
            for (int i = 0; i < LAYER_WIDTH; ++i)
                PR("    bias[%d] = %f", node_idx + i, nn->results[example].bias_der[node_idx + i]);
        }

        // Skip weights for the last layer
        if (layer < NUM_LAYERS - 1)
        {
            int num_weights = LAYER_WIDTH * LAYER_WIDTH;

            PR("  Weights:");
            for (int i = 0; i < num_weights; ++i)
                PR("    weight[%d] = %f", weight_idx + i, nn->weight[weight_idx + i]);

            PR("  Weight Der:");
            for (int i = 0; i < num_weights; ++i)
                PR("    weight[%d] = %f", weight_idx + i, nn->results[example].weight_der[weight_idx + i]);

            weight_idx += num_weights;
        }

        PR("  Activations:");
        for (int i = 0; i < LAYER_WIDTH; ++i)
            PR("    nn[%d] = %f", node_idx + i, nn->results[example].nn[node_idx + i]);

        node_idx += LAYER_WIDTH;
    }

    for (int i = 0; i < NUM_OUTPUTS; ++i)
        PR("Answer[%d] = %f", i, nn->answers[example * NUM_OUTPUTS + i]);

    PR("%s", "\n--- Cost ---");    
    PR("Cost = %f", nn->cost);
    PR("Offset = %f", nn->offset);

    PR("\n===== END OF NN STATE =====");
}


void trainNetwork(NN *const net)
{
    net->cost = 0;
    static f pre = 0;
    for (uint8_t i = 0; i < NUM_EXAMPLES; ++i)
        cycleNetwork(net, i);
    sumDerivatives(net->results, &net->adam, 1);
    adjustNetwork(net);
    PR("Cost: %f Offset: %f", net->cost, net->offset);

    pre = net->cost;
}


void verifyDerivatives(NN *net)
{
    int num_examples = NUM_EXAMPLES;

    for (int i = 0; i < num_examples; ++i)
    {
        net->input_data[i*NUM_INPUTS] = (f)rand() / (f)RAND_MAX;
        net->input_data[i*NUM_INPUTS+1] = (f)rand() / (f)RAND_MAX;
        net->answers[i*NUM_OUTPUTS] = (f)rand() / (f)RAND_MAX;
        net->answers[i*NUM_OUTPUTS+1] = (f)rand() / (f)RAND_MAX;
    }


    const f delta = 0.000001f;
    for (int i = 0; i < NUM_NODES-NUM_OUTPUTS; ++i)
    {
        net->cost = 0;
        for (uint8_t j = 0; j < num_examples; ++j)
            cycleNetwork(net, j);
        sumDerivatives(net->results, &net->adam, num_examples);

        f original_cost = net->cost;
        net->bias[i] += delta;
        net->cost = 0;

        for (uint8_t j = 0; j < num_examples; ++j)
            cycleNetwork(net, j);
        sumDerivatives(net->results, &net->adam, num_examples);

        f diff = (net->cost - original_cost) / delta;
        PR("BIAS:    der: %13.6e calc der: %13.6e diff: %13.6e %s", 
            net->adam.bias[i].der_sum, diff, 
            fabs(net->adam.bias[i].der_sum - diff), 
            (fabs(net->adam.bias[i].der_sum - diff) < 0.000001f ? "⭐" : ""));
    }

    for (int i = 0; i < NUM_WEIGHTS; ++i)
    {
        net->cost = 0;
        for (uint8_t j = 0; j < num_examples; ++j)
            cycleNetwork(net, j);
        sumDerivatives(net->results, &net->adam, num_examples);

        f original_cost = net->cost;
        net->weight[i] += delta;
        net->cost = 0;

        for (uint8_t j = 0; j < num_examples; ++j)
            cycleNetwork(net, j);
        sumDerivatives(net->results, &net->adam, num_examples);

        f diff = (net->cost - original_cost) / delta;
        PR("WEIGHT:  der: %13.6e calc der: %13.6e diff: %13.6e %s", 
            net->adam.weight[i].der_sum, diff, 
            fabs(net->adam.weight[i].der_sum - diff), 
            (fabs(net->adam.weight[i].der_sum - diff) < 0.000001f ? "⭐" : ""));
    }

    // printNNState(net, 0);
    exit(0);
}


int main()
{
    srand(40);

    static NN net = {
        .adam.hyper.beta1=B1, 
        .adam.hyper.beta2=B2, 
        .adam.hyper.beta1_1=1-B1, 
        .adam.hyper.beta2_1=1-B2, 
        .adam.hyper.epsilon=1.0e-6f, 
        .adam.hyper.learning_rate=1e-1f,
        .adam.hyper.num_examples = NUM_EXAMPLES
    };

    determineWhoDoesWhatAndArrayIndexes(&net);
    initNetwork(&net);

    // verifyDerivatives(&net);

    for (int i = 0; i < NUM_NODES; ++i)
        net.bias[i] = i;

    for (int i = 0; i < NUM_WEIGHTS; ++i)
        net.weight[i] = i;

    for (int i = 0; i < NUM_EXAMPLES; ++i)
    {
        net.input_data[i*NUM_INPUTS] = (f)i;
        net.input_data[i*NUM_INPUTS+1] = (f)i;

        net.answers[i*NUM_OUTPUTS] = (f)-i;
        net.answers[i*NUM_OUTPUTS+1] = (f)-i;
    }

    for (int i = 0; i < 1000; ++i)
        trainNetwork(&net);

    printNNState(&net, 0);
    return 0;
}

