#ifndef DEBUGGING__
#define DEBUGGING__


#include "Variables.h"



#ifdef VERIFY_ISNORMAL

void verifyIsNormal(DT number, int line, char *file, char *variable)
{

    if (!isnan(number)  && isinf(number))
    {
        printf("%s:%d %s not normal:%f\n", file, line, variable, number);
        exit(EXIT_FAILURE);
    }


}
#endif

#endif
