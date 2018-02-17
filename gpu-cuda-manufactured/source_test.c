#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

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
    const double Q = tanh((1.0L/2.0L)*sqrt(2)*(A1*t*sin(B1*x) + A2*sin(B2*x + C2*t) - y + 0.25)/sqrt(kappa));
    *S = (  0.5  * sqrt(kappa)*tanh((1.0L/2.0L)*sqrt(2)*(A1*t*sin(B1*x) + A2*sin(B2*x + C2*t) - y + 0.25)/sqrt(kappa))
            - 0.25 * sqrt(2)*(A1*sin(B1*x) + A2*C2*cos(B2*x + C2*t))
         ) * (Q * Q - 1)/sqrt(kappa);
}

int main()
{

    FILE* fp;
    int i;
    double rms = 0.;

    srand(time(0));
    fp = fopen("source-check.txt", "w");

    fprintf(fp, "x\ty\tt\tS0\tS1\tError\n");

    for (i = 0; i < 1000; i++) {
        const double x = (double)rand() / (double)RAND_MAX;
        const double y = (double)rand() / (double)RAND_MAX / 2.000;
        const double t = (double)rand() / (double)RAND_MAX / 0.125;

        double Ss = 0.;
        double Sp = 0.;
        double er = 0.;

        source_dewtt(x, y, t, &Ss);
        source_sympy(x, y, t, &Sp);
        er = (Ss - Sp) / Ss;
        rms += er * er;

        fprintf(fp, "%f\t%f\t%f\t%f\t%f\t%f\n", x, y, t, Ss, Sp, er);
    }

    printf("rms error for 1000 runs was %f\n", sqrt(rms / 1000.0));

    fclose(fp);

    return 0;
}
