#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#ifndef M_PI
#define M_PI (3.14159265358979323846)
#endif
const double A1 = 0.0075;
const double A2 = 0.0300;
const double B1 = 8. * M_PI;
const double B2 = 22. * M_PI;
const double C2 = 0.0625;
const double kappa = 0.0004;

void manufactured_shift(const double x, const double t, double* a)
{
	/* Equation 3 */
	*a = 0.25 + A1 * t * sin(B1 * x) + A2 * sin(B2 * x + C2 * t);
}

void source_dewtt(const double x, const double y, const double t, double* S)
{
    double alpha = 0.;
    manufactured_shift(x, t, &alpha);
    const double dadx = A1 * B1 * t * cos(B1 * x) + A2 * B2 * cos(B2 * x + C2 * t);
    const double d2adx2 = -A1 * B1 * B1 * t * sin(B1 * x) - A2 * B2 * B2 * sin(B2 * x + C2 * t);
    const double dadt = A1 * sin(B1 * x) + A2 * C2 * cos(B2 * x + C2 * t);
    const double Q = (y - alpha) / sqrt(2. * kappa);
    const double sech = 1.0 / cosh(Q);
    const double sum = - sqrt(4. * kappa) * tanh(Q) * dadx * dadx
                       + sqrt(2.) * (dadt - kappa * d2adx2);
    *S = (sech * sech  * sum) / sqrt(16. * kappa);
}

void source_sympy(const double x, const double y, const double t, double* S)
{
    const double dadx = A1*B1*t*cos(x*B1) + A2*B2*cos(x*B2 + C2*t);
    const double d2adx2 = -A1*pow(B1, 2)*t*sin(x*B1) - A2*pow(B2, 2)*sin(x*B2 + C2*t);
    const double alpha  = A1*t*sin(x*B1) + A2*sin(x*B2 + C2*t) + 0.25;
    const double dadt   = A1*sin(x*B1) + A2*C2*cos(x*B2 + C2*t);
    const double hyper = tanh(0.5*sqrt(2)*(-y + alpha)/sqrt(kappa));
    *S = -(sqrt(kappa) * (0.25*sqrt(2*kappa) * d2adx2
                          + 0.5*pow(dadx, 2) * tanh(0.5*sqrt(2)*(-y + alpha) / sqrt(kappa))
               ) + 0.25*sqrt(2) * (dadt)
        ) * (hyper*hyper - 1)/sqrt(kappa);
}

int main()
{

    FILE* fp;
    int i, j, N=10000;
    double rms = 0.;

    srand(time(0));
    fp = fopen("source-check.txt", "w");

    fprintf(fp, "x\ty\tt\tS0\tS1\tError\n");

    for (k=0; k<2; k+=3) {
        for (j=0; j < sqrt(N); j++) {
            for (i = 0; i < sqrt(N); i++) {
                /*
                const double x = (double)rand() / (double)RAND_MAX;
                const double y = (double)rand() / (double)RAND_MAX / 2.000;
                const double t = (double)rand() / (double)RAND_MAX / 0.125;
                */

                double Ss = 0.;
                double Sp = 0.;
                double er = 0.;

                source_dewtt(x, y, t, &Ss);
                source_sympy(x, y, t, &Sp);
                er = (Ss - Sp) / Ss;
                rms += er * er;

                fprintf(fp, "%f\t%f\t%f\t%f\t%f\t%f\n", x, y, t, Ss, Sp, er);
            }
        }
    }

    printf("rms error for %i runs was %f\n", N, sqrt(rms / N));

    fclose(fp);

    return 0;
}
