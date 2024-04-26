//
// Created by root on 4/9/24.
//

#pragma once

void add(float *x, float *y, float *z, int n) {
    for (int i = 0; i < n; i++) {
        z[i] = x[i] + y[i];
    }
}

void matrix_add(float **x, float **y, float **z, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            z[i][j] = x[i][j] + y[i][j];
        }
    }
}

void histograms_add(float *x, float **y, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            x[i] = x[i] + y[i][j];
        }
    }
}
