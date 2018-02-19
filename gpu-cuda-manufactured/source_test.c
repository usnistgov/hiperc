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

void source_dewtt(const double x, const double y, const double t, double* S)
{
    /* Equation 3 */
    const double alpha = 0.25 + A1 * t * sin(B1 *x) + A2* sin(B2 * x + C2 * t);
    /* Equation 4 */
    const double dadx = A1 * B1 *t * cos(B1 * x)+ A2 * B2 * cos(B2 * x + C2 * t);
    const double d2adx2 = -A1 * B1 * B1 * t * sin(B1 * x) - A2 * B2 * B2 * sin(B2 * x + C2 * t);
    const double dadt = A1 * sin(B1 * x) + A2 * C2 * cos(B2 * x +C2 * t);
    const double Q = (y -alpha) / sqrt(2. * kappa);
    const double sech = 1. / cosh(Q);
    const double sum = -sqrt(4. *kappa) * tanh(Q) * dadx* dadx + sqrt(2.) * (dadt - kappa * d2adx2);
    *S = sech * sech / sqrt(16. * kappa) * sum;

}

void source_sympy(const double x, const double y, const double t, double* S)
{
    const double sq2 = sqrt(2.);
    const double sqK = sqrt(kappa);
    const double Q = 0.5*sq2*(-y + A1*t*sin(B1*x) + A2*sin(B2*x + C2*t) + 0.25)/sqK;
    const double sech2 = 1. - tanh(Q)*tanh(Q);
    *S = (1. - tanh(Q)*tanh(Q))/sqrt(16.*kappa) * (2.0*sqK*pow(A1*B1*t*cos(B1*x) + A2*B2*cos(B2*x + C2*t), 2)*(-sech2)*tanh(Q)
                                                   + sq2*kappa*(A1*pow(B1, 2)*t*sin(B1*x) + A2*pow(B2, 2)*sin(B2*x + C2*t))*(-sech2)
                                                   + sq2*(A1*sin(B1*x) + A2*C2*cos(B2*x + C2*t)));
}

int main()
{

    FILE* fp;
    int i, j, N=10000;
    double rms = 0.;

    srand(time(0));
    fp = fopen("source-check.txt", "w");

    fprintf(fp, "x\ty\tt\tS0\tS1\tError\n");

    for (j=0; j < sqrt(N); j++) {
        for (i = 0; i < sqrt(N); i++) {
            const double x = (double)rand() / (double)RAND_MAX;
            const double y = (double)rand() / (double)RAND_MAX / 2.000;
            const double t = (double)rand() / (double)RAND_MAX / 0.125;

            double Ss = 0.;
            double Sp = 0.;
            double er = 0.;

            source_dewtt(x, y, t, &Ss);
            source_sympy(x, y, t, &Sp);
            er = fabs(Ss - Sp);
            rms += er * er;

            fprintf(fp, "%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n", x, y, t, Ss, Sp, er);
        }
    }

    printf("rms error for %i runs was %.4f\n", N, sqrt(rms / N));

    fclose(fp);

    return 0;
}
