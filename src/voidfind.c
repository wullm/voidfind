/*******************************************************************************
 * This file is part of Voidfind.
 * Copyright (c) 2021 Willem Elbers (whe@willemelbers.com)
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <assert.h>
#include <complex.h>

#include "../include/voidfind.h"

int main(int argc, char *argv[]) {
    if (argc == 1) {
        printf("No parameter file specified.\n");
        return 0;
    }

    /* Read options */
    const char *fname = argv[1];
    printf("Void Finder\n");
    printf("The parameter file is '%s'\n", fname);

    /* Voidfind structures */
    struct params pars;
    struct units us;

    /* Read parameter file for parameters, units, and cosmological values */
    readParams(&pars, fname);
    readUnits(&us, fname);

    /* Read the density file */
    double *density;
    double boxlen;
    int N;
    readFieldFile(&density, &N, &boxlen, pars.InputFilename);

    /* Read the secondary density file */
    double *density2;
    int N2 = 0;
    if (strcmp(pars.SecondaryInputFilename, "") != 0) {
        double boxlen2;
        readFieldFile(&density2, &N2, &boxlen2, pars.SecondaryInputFilename);

        assert(N == N2);
        assert(boxlen == boxlen2);

        /* Add the secondary density to the primary density */
        double fac = pars.SecondaryFactor;
        printf("Factor for the secondary input is %f\n", fac);
        for (int i=0; i<N*N*N; i++) {
            density[i] = (density[i] + density2[i] * fac) / (1.0 + fac);
        }
    }

    /* Compute the Fourier transform */
    fftw_complex *fbox = malloc(N*N*(N/2+1) * sizeof(fftw_complex));
    fftw_plan r2c = fftw_plan_dft_r2c_3d(N, N, N, density, fbox, FFTW_ESTIMATE);
    fft_execute(r2c);
    fftw_destroy_plan(r2c);
    fft_normalize_r2c(fbox, N, boxlen);

    printf("Computed Fourier transform.\n");

    /* Smooth the grid */
    double R_smooth = pars.SmoothingRadius;
    fft_apply_kernel(fbox, fbox, N, boxlen, kernel_gaussian, &R_smooth);

    /* Compute the gradient */
    // fftw_complex *fgrad_x = malloc(N*N*(N/2+1) * sizeof(fftw_complex));
    // fftw_complex *fgrad_y = malloc(N*N*(N/2+1) * sizeof(fftw_complex));
    // fftw_complex *fgrad_z = malloc(N*N*(N/2+1) * sizeof(fftw_complex));
    // fft_apply_kernel(fgrad_x, fbox, N, boxlen, kernel_dx, NULL);
    // fft_apply_kernel(fgrad_y, fbox, N, boxlen, kernel_dy, NULL);
    // fft_apply_kernel(fgrad_z, fbox, N, boxlen, kernel_dz, NULL);
    //
    // /* Compute the inverse Fourier transform of the gradient and Hessian */
    // double *grad_x = malloc(N*N*N*sizeof(double));
    // double *grad_y = malloc(N*N*N*sizeof(double));
    // double *grad_z = malloc(N*N*N*sizeof(double));
    // fftw_plan c2r_x = fftw_plan_dft_c2r_3d(N, N, N, fgrad_x, grad_x, FFTW_ESTIMATE);
    // fftw_plan c2r_y = fftw_plan_dft_c2r_3d(N, N, N, fgrad_y, grad_y, FFTW_ESTIMATE);
    // fftw_plan c2r_z = fftw_plan_dft_c2r_3d(N, N, N, fgrad_z, grad_z, FFTW_ESTIMATE);
    // fft_execute(c2r_x);
    // fft_execute(c2r_y);
    // fft_execute(c2r_z);
    // fftw_destroy_plan(c2r_x);
    // fftw_destroy_plan(c2r_y);
    // fftw_destroy_plan(c2r_z);
    // fft_normalize_c2r(grad_x, N, boxlen);
    // fft_normalize_c2r(grad_y, N, boxlen);
    // fft_normalize_c2r(grad_z, N, boxlen);
    // free(fgrad_x);
    // free(fgrad_y);
    // free(fgrad_z);

    /* Compute the inverse Fourier transform of the smoothed grid */
    fftw_plan c2r = fftw_plan_dft_c2r_3d(N, N, N, fbox, density, FFTW_ESTIMATE);
    fft_execute(c2r);
    fftw_destroy_plan(c2r);
    fft_normalize_c2r(density, N, boxlen);
    free(fbox);

    printf("Computed inverse Fourier transform.\n");

    int void_num = 0;

    char *is_minimum = calloc(N*N*N, sizeof(char));

    /* Find local minima that are below the point threshold */
    #pragma omp parallel for reduction(+: void_num)
    for (int x=0; x<N; x++) {
        for (int y=0; y<N; y++) {
            for (int z=0; z<N; z++) {
                // int xa = sgn(grad_x[row_major(x,y,z,N)]);
                // int xb = sgn(grad_x[row_major(x+1,y,z,N)]);
                // int ya = sgn(grad_x[row_major(x,y,z,N)]);
                // int yb = sgn(grad_x[row_major(x,y+1,z,N)]);
                // int za = sgn(grad_x[row_major(x,y,z,N)]);
                // int zb = sgn(grad_x[row_major(x,y,z+1,N)]);

                // if (xa != xb && ya != yb && za != zb) {

                double val = density[row_major(x,y,z,N)];
                int local_minimum = 1;
                int r = 1;
                for (int i=-r; i<r+1; i++) {
                    for (int j=-r; j<r+1; j++) {
                        for (int k=-r; k<r+1; k++) {
                            if (i==0 && j==0 && k==0) continue;
                            if (val >= density[row_major(x+i,y+j,z+k,N)]) {
                                local_minimum = 0;
                                goto end_check;
                            }
                        }
                    }
                }

                end_check:

                if (local_minimum) {
                    is_minimum[row_major(x,y,z,N)] = 1;
                    void_num++;
                }
            }
        }
    }

    printf("We found %d local minima.\n", void_num);

    /* Make a sorted list of local minima */
    struct minimum *minima = malloc(void_num * sizeof(struct minimum));
    int indx = 0;

    /* Insert the critical points into the array */
    for (int x=0; x<N; x++) {
        for (int y=0; y<N; y++) {
            for (int z=0; z<N; z++) {
                if (is_minimum[row_major(x,y,z,N)]) {
                    minima[indx].val = density[row_major(x,y,z,N)];
                    minima[indx].y = y;
                    minima[indx].x = x;
                    minima[indx].z = z;
                    indx++;
                }
            }
        }
    }

    /* We are done with the is_minimum field */
    free(is_minimum);

    /* We are done with the gradient field */
    // free(grad_x);
    // free(grad_y);
    // free(grad_z);

    /* Sort the minima by value */
    qsort(minima, void_num, sizeof(struct minimum), compareByVal);

    /* Discard minima that are too close to a lower minimum */
    double Rx = pars.ExclusionRadius / boxlen * N;
    double Rx2 = Rx * Rx;
    int discard = 0;
    for (int i=1; i<void_num; i++) {
        for (int j=0; j<i; j++) {
            double d2 = dist2(&minima[i], &minima[j], N);
            if (d2 < Rx2 && minima[j].val < minima[i].val) {
                minima[i].val = pars.MaximumPointDensity + 1; //disable
                discard++;
                break;
            }
        }
    }

    /* Sort the minima again, to fish out the discards */
    qsort(minima, void_num, sizeof(struct minimum), compareByVal);

    /* Resize to remove the discards */
    minima = realloc(minima, (void_num - discard) * sizeof(struct minimum));
    void_num -= discard;

    printf("Discarded %d voids that are too close to another.\n", discard);

    /* Find maximum radius of voids */
    discard = 0;
    #pragma omp parallel for reduction(+: discard)
    for (int i=0; i<void_num; i++) {
        int R = 0;
        double avg = minima[i].val;
        while(R<N && avg < pars.DensityThreshold) {
            R++;
            struct minimum *m = &minima[i];
            double sum = 0;
            int count = 0;
            for (int x=-R; x<R+1; x++) {
                for (int y=-R; y<R+1; y++) {
                    for (int z=-R; z<R+1; z++) {
                        if (x*x + y*y + z*z > R*R) continue;
                        sum += density[row_major(m->x + x, m->y + y, m->z + z, N)];
                        count++;
                    }
                }
            }
            avg = sum/count;
        }

        minima[i].R = R - 1;

        /* Discard voids with negative radius */
        if (minima[i].R <= 0) {
            discard++;
        }
    }

    /* Sort the minima by void radius */
    qsort(minima, void_num, sizeof(struct minimum), compareByRadius);

    /* The largest void in the sample */
    int Rmax = minima[0].R;

    /* Resize to remove the discards */
    minima = realloc(minima, (void_num - discard) * sizeof(struct minimum));
    void_num -= discard;

    printf("Discarded %d voids with zero or negative radius.\n", discard);

    /* Discard voids that are entirely inside other voids  */
    discard = 0;
    for (int i=1; i<void_num; i++) {
        for (int j=0; j<i; j++) {
            double d = sqrt(dist2(&minima[i], &minima[j], N));
            if (d + minima[i].R < minima[j].R && minima[j].R >= minima[i].R) {
                minima[i].val = pars.MaximumPointDensity + 1; //disable
                discard++;
                break;
            }
        }
    }

    /* Sort the minima by value */
    qsort(minima, void_num, sizeof(struct minimum), compareByVal);

    /* Resize to remove the discards */
    minima = realloc(minima, (void_num - discard) * sizeof(struct minimum));
    void_num -= discard;

    printf("Discarded %d voids that are contained inside another.\n", discard);
    printf("There are %d remaining voids.\n", void_num);

    /* Sort the minima by void radius */
    qsort(minima, void_num, sizeof(struct minimum), compareByRadius);

    printf("\n\nHistogram\n");

    /* Compute a void size histogram */
    int *histogram = calloc(Rmax+1, sizeof(int));
    for (int i=0; i<void_num; i++) {
        int R = minima[i].R;
        if (R >= 0 && R <= Rmax) {
            histogram[R]++;
        }
    }

    for (int r=0; r<Rmax+1; r++) {
        printf("%d %d\n", r, histogram[r]);
    }


    printf("\n\nAverage profile\n");

    /* Compute average profile */
    int bin_R_min = (int) (pars.MinRadius * N / boxlen);
    int bin_R_max = (int) (pars.MaxRadius * N / boxlen + 1);
    int profile_size = (int) (pars.ProfileMaxRadius * N / boxlen + 1);;

    if (pars.MaxRadius == -1) {
        bin_R_max = Rmax + 1;
    }
    printf("Computing average profile for voids with R in (%d, %d)\n", bin_R_min, bin_R_max);

    /* Start computing the average profile */
    int voids_counted = 0;
    double *profile = calloc(profile_size, sizeof(double));
    double *profile2 = calloc(profile_size, sizeof(double));

    /* Reload the original density field */
    free(density);
    readFieldFile(&density, &N, &boxlen, pars.InputFilename);

    for (int i=0; i<10; i++) {
        struct minimum *m = &minima[i];
        int R = m->R;
        if (R <= bin_R_min || R >= bin_R_max) continue;

        #pragma omp parallel for
        for (int r=0; r<profile_size; r++) {
            double sum = 0, sum2 = 0;
            int count = 0;
            for (int x=-r; x<r+1; x++) {
                for (int y=-r; y<r+1; y++) {
                    for (int z=-r; z<r+1; z++) {
                        if (x*x + y*y + z*z > r*r) continue;
                        sum += density[row_major(m->x + x, m->y + y, m->z + z, N)];
                        if (N2 > 0) sum2 += density2[row_major(m->x + x, m->y + y, m->z + z, N)];
                        count++;
                    }
                }
            }
            profile[r] += sum/count;
            profile2[r] += sum2/count;
        }
        voids_counted++;
    }

    printf("We counted %d voids\n", voids_counted);

    for (int r=0; r<profile_size; r++) {
        profile[r] /= voids_counted;
        profile2[r] /= voids_counted;
    }

    for (int r=0; r<profile_size; r++) {
        printf("%d %f %f\n", r, profile[r], profile2[r]);
    }

    // /* Colour the density grid */
    // for (int i=0; i<void_num; i++) {
    //     struct minimum *m = &minima[i];
    //     int R = m->R;
    //
    //     for (int x=-R; x<R+1; x++) {
    //         for (int y=-R; y<R+1; y++) {
    //             for (int z=-R; z<R+1; z++) {
    //                 if (x*x + y*y + z*z > R*R) continue;
    //                 density[row_major(m->x + x, m->y + y, m->z + z, N)] = -1;
    //             }
    //         }
    //     }
    // }
    //
    // /* Export the coloured grid */
    // writeFieldFile(density, N, boxlen, "out.hdf5");

    /* Free arrays that are no longer needed */
    free(minima);
    free(profile);
    free(profile2);
    free(density);
    if (N2 > 0) free(density2);
    free(histogram);

    /* Clean up */
    cleanParams(&pars);

    return 0;
}
