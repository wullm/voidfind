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

#ifndef VOIDFIND_H
#define VOIDFIND_H

#include "input.h"
#include "output.h"
#include "fft.h"
#include "fft_kernels.h"


#define sgn(x) (x > 0) ? 1 : ((x < 0) ? -1 : 0)

struct minimum {
    double val;
    int x,y,z;
    int R;
};

/* Sort by value, lowest first */
static inline int compareByVal(const void *a, const void *b) {
    struct minimum *ma = (struct minimum*) a;
    struct minimum *mb = (struct minimum*) b;
    return ma->val >= mb->val;
}

/* Sort by radius, largest first */
static inline int compareByRadius(const void *a, const void *b) {
    struct minimum *ma = (struct minimum*) a;
    struct minimum *mb = (struct minimum*) b;
    return ma->R <= mb->R;
}

static inline double dist2(struct minimum *a, struct minimum *b, int N) {
    /* Compute the distances */
    double dx = a->x - b->x;
    double dy = a->y - b->y;
    double dz = a->z - b->z;

    /* Account for periodic boundaries */
    dx = (dx > N) ? dx - N : dx;
    dy = (dy >= N) ? dy - N : dy;
    dz = (dz >= N) ? dz - N : dz;

    /* Return the squared distance */
    return (dx * dx + dy * dy + dz * dz);
}

#endif
