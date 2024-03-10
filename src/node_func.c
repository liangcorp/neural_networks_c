#include <math.h>
#include <stdlib.h>

#include "machine_learning.h"

double *node_func(double **X, double *theta, int num_feat, int num_train)
{
	double sum = 0.0L;
	double *h_x = calloc(num_train, sizeof(double));

	for (int i = 0; i < num_train; i++) {
		sum = 0.0L;
		for (int j = 0; j < num_feat; j++) {
			sum += theta[j] * X[i][j];
		}
		h_x[i] = pow(M_E, sum) / (1 + pow(M_E, sum));
	}

	return h_x;
}
