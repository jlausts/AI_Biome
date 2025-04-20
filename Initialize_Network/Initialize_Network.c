

#include "Initialize_Network.h"

#ifndef MY_PI
#define MY_PI 6.28318530717958f
#endif

void fillArrayWithRandomGaussian(DT *array, const uint32_t length, const float variance, const DT mult) 
{
    for (uint32_t i = 0; i < length; i += 2) 
    {
        // Generate two uniform random numbers between 0 and 1
        const DT u1 = sqrtf(-2.0f *  logf( ((DT)rand() + 1.0f) / (DT)((float)RAND_MAX + 3.0f) )) * sqrtf(variance);
        const DT u2 =                      ((DT)rand() + 1.0f) / (DT)((float)RAND_MAX + 3.0f)    * MY_PI;

        // Scale by standard deviation and shift by mean
        array[i] = u1 * cosf(u2) * mult;

        if (array[i] > 1000.0f)
        {
            printf("%s:%d\nSomething wrong with network initialization\n%f %d %f %f %f\n", __FILE__, __LINE__, array[i], i, u1, u2, sqrtf(variance));
            exit(EXIT_FAILURE);
        }

        // PRINTLN(array[i]);

        // Fill second number if we have space
        if (i + 1 < length) 
            array[i + 1] = u1 * sinf(u2) * mult;
    }
}



void initializeWeights(const uint32_t network_geometry[NUM_LAYERS])
{
    const DT mults[NUM_LAYERS - 1] = {1.08f, 2.85f, 2.8f, 2.57f, 2.88f, 2.68f, 3.36f};

    for (uint32_t layer = 0; layer < NUM_LAYERS - 1; ++layer)
    {
        const DT variance = (1.0f / (DT) (network_geometry[layer] + network_geometry[layer + 1]) + 1.0f / (DT)network_geometry[layer]);
        fillArrayWithRandomGaussian(&weights[weight_indexes[layer]], weight_indexes[layer+1] - weight_indexes[layer], variance, mults[layer]);
        fillArrayWithRandomGaussian(&biases[bias_indexes[layer]], bias_indexes[layer+1] - bias_indexes[layer], variance, 0.1f);
    }
}

void fillArrayWithRandomGaussian2(DT *array, const uint32_t length, const float variance, const DT mult) 
{
    for (uint32_t i = 0; i < length; i += 2) 
    {
        // Generate two uniform random numbers between 0 and 1
        const DT u1 = sqrtf(-2.0f *  logf( ((DT)rand() + 1.0f) / (DT)((float)RAND_MAX + 3.0f) )) * variance;
        const DT u2 =                      ((DT)rand() + 1.0f) / (DT)((float)RAND_MAX + 3.0f)    * MY_PI;

        // Scale by standard deviation and shift by mean
        array[i] = u1 * cosf(u2) * mult;

        if (array[i] > 1000.0f)
        {
            printf("%s:%d\nSomething wrong with network initialization\n%f %d %f %f %f\n", __FILE__, __LINE__, array[i], i, u1, u2, sqrtf(variance));
            exit(EXIT_FAILURE);
        }

        // Fill second number if we have space
        if (i + 1 < length) 
            array[i + 1] = u1 * sinf(u2) * mult;
    }

    // get average
    DT avg = array[0];
    for (uint32_t i = 1; i < length; ++i)
        avg += array[i];
    avg /= (DT) length;


    // stdev
    DT std = 0;
    for (uint32_t i = 0; i < length; ++i)
    {
        const DT diff = array[i] - avg;
        std += diff * diff;
    }
    std = sqrtf(std / (DT)length);


    // normalize the array to a different variance
    for (uint32_t i = 0; i < length; ++i)
        array[i] = (array[i] - avg) / std * variance;
}

void initializeWeights2(const uint32_t network_geometry[NUM_LAYERS])
{
    for (uint32_t layer = 0; layer < NUM_LAYERS - 1; ++layer)
    {
        const DT variance = sqrtf(2.0f / (DT) (network_geometry[layer] + network_geometry[layer + 1]));
        fillArrayWithRandomGaussian2(&weights[weight_indexes[layer]], weight_indexes[layer+1] - weight_indexes[layer], variance, 1.0f);
        fillArrayWithRandomGaussian2(&biases[bias_indexes[layer]], bias_indexes[layer+1] - bias_indexes[layer], variance, 0.1f);
    }
}

void fillArrayWithRandomGaussian3(DT *array, const uint32_t length, const float variance, const DT mult) 
{
    for (uint32_t i = 0; i < length; i += 2) 
    {
        // Generate two uniform random numbers between 0 and 1
        const DT u1 = sqrtf(-2.0f *  logf( ((DT)rand() + 1.0f) / (DT)((float)RAND_MAX + 3.0f) )) * variance;
        const DT u2 =                      ((DT)rand() + 1.0f) / (DT)((float)RAND_MAX + 3.0f)    * MY_PI;

        // Scale by standard deviation and shift by mean
        array[i] = u1 * cosf(u2) * mult;

        if (array[i] > 1000.0f)
        {
            printf("%s:%d\nSomething wrong with network initialization\n%f %d %f %f %f\n", __FILE__, __LINE__, array[i], i, u1, u2, sqrtf(variance));
            exit(EXIT_FAILURE);
        }

        // Fill second number if we have space
        if (i + 1 < length) 
            array[i + 1] = u1 * sinf(u2) * mult;
    }

    // get max
    DT max = array[0];
    for (uint32_t i = 1; i < length; ++i)
        if (array[i] > max)
            max = array[i];


    // re scale the weights to fit inside of the variance
    for (uint32_t i = 0; i < length; ++i)
        array[i] = array[i] * variance / max;
}

void initializeWeights3(const uint32_t network_geometry[NUM_LAYERS])
{
    for (uint32_t layer = 0; layer < NUM_LAYERS - 1; ++layer)
    {
        const DT variance = sqrtf(6.0f / (DT) (network_geometry[layer] + network_geometry[layer + 1]));
        fillArrayWithRandomGaussian3(&weights[weight_indexes[layer]], weight_indexes[layer+1] - weight_indexes[layer], variance, 1.0f);
        fillArrayWithRandomGaussian3(&biases[bias_indexes[layer]], bias_indexes[layer+1] - bias_indexes[layer], variance, 0.1f);
    }
}


#ifdef TESTING
int main()
{

    srand(0);
    const uint32_t network_geometry[NUM_LAYERS] = {10, 10, 10, 10, 10, 10, 10, 10};


    bias_indexes[0] = 0;
    weight_indexes[0] = 0;
    for (uint8_t i = 1; i < NUM_LAYERS; ++i)
    {
        weight_indexes[i] = weight_indexes[i - 1] + (uint32_t) (network_geometry[i - 1] * network_geometry[i]);
        bias_indexes[i] = (uint32_t) (bias_indexes[i - 1] + network_geometry[i]);
    }


    initialize(network_geometry);

    PRINT_SIZE(weights, 0, 193);
}
#endif