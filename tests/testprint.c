#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <omp.h>

#include "TinyDFT.h"



int main(int argc, char **argv)
{
    double x = -0.002;
    double y =fabs(x);
    double lgx=log10(y);
    printf("%f,%f",y,lgx);
    return 0;
}
