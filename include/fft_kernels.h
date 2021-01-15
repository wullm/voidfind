/*******************************************************************************
 * This file is part of Mitos.
 * Copyright (c) 2020 Willem Elbers (whe@willemelbers.com)
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/

#ifndef FFT_KERNELS_H
#define FFT_KERNELS_H

typedef void (*kernel_func)(struct kernel *the_kernel);

static inline void kernel_lowpass(struct kernel *the_kernel) {
    double k = the_kernel->k;
    double k_max = *((double*) the_kernel->params);
    the_kernel->kern = (k < k_max) ? 1.0 : 0.0;
}

static inline void kernel_hipass(struct kernel *the_kernel) {
    double k = the_kernel->k;
    double k_min = *((double*) the_kernel->params);
    the_kernel->kern = (k > k_min) ? 1.0 : 0.0;
}

static inline void kernel_tophat(struct kernel *the_kernel) {
    double k = the_kernel->k;
    double *param = (double *) the_kernel->params;
    double k_min = param[0];
    double k_max = param[1];
    the_kernel->kern = (k >= k_min && k <= k_max) ? 1.0 : 0.0;
}


static inline void kernel_elliptic_tophat(struct kernel *the_kernel) {
    double kx = the_kernel->kx;
    double ky = the_kernel->ky;
    double kz = the_kernel->kz;
    double *param = (double *) the_kernel->params;
    double rx = param[0];
    double ry = param[1];
    double rz = param[2];
    double eta2 = param[3];
    double kr = kx*rx + ky*ry + kz*rz;
    double kk = kx*kx + ky*ky + kz*kz;
    double rr = rx*rx + ry*ry + rz*rz;
    double theta = kk*rr + (eta2 - 1)*kr*kr;
    the_kernel->kern = (theta <= 4*M_PI*M_PI) ? 1.0 : 0.0;
}

static inline void kernel_translate(struct kernel *the_kernel) {
    double kx = the_kernel->kx;
    double ky = the_kernel->ky;
    double kz = the_kernel->kz;
    double *param = (double *) the_kernel->params;
    double rx = param[0];
    double ry = param[1];
    double rz = param[2];
    the_kernel->kern = cexp(I*(rx*kx + ry*ky + rz*kz));
}

static inline void kernel_gaussian(struct kernel *the_kernel) {
    double k = the_kernel->k;
    double R = *((double*) the_kernel->params);
    double kR = k * R;
    the_kernel->kern = exp(-kR * kR);
}

static inline void kernel_gaussian_inv(struct kernel *the_kernel) {
    double k = the_kernel->k;
    double R = *((double*) the_kernel->params);
    double kR = k * R;
    the_kernel->kern = exp(kR * kR);
}


static inline void kernel_inv_poisson(struct kernel *the_kernel) {
    double k = the_kernel->k;
    the_kernel->kern = (k > 0) ? -1.0/k/k : 1.0;
}

static inline void kernel_dx(struct kernel *the_kernel) {
    double kx = the_kernel->kx;
    the_kernel->kern = I*kx;
}

static inline void kernel_dy(struct kernel *the_kernel) {
    double ky = the_kernel->ky;
    the_kernel->kern = I*ky;
}

static inline void kernel_dz(struct kernel *the_kernel) {
    double kz = the_kernel->kz;
    the_kernel->kern = I*kz;
}


#endif
