/*******************************************************************************
 * This file is part of Voidfind.
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

#ifndef INPUT_H
#define INPUT_H

#define DEFAULT_STRING_LENGTH 150

#define KM_METRES 1000
#define MPC_METRES 3.085677581282e22

#define SPEED_OF_LIGHT_METRES_SECONDS 2.99792e8
#define GRAVITY_G_SI_UNITS 6.67428e-11 // m^3 / kg / s^2
#define PLANCK_CONST_SI_UNITS 6.62607015e-34 //J s
#define BOLTZMANN_CONST_SI_UNITS 1.380649e-23 //J / K
#define ELECTRONVOLT_SI_UNITS 1.602176634e-19 // J

/* The .ini parser library is minIni */
#include "../parser/minIni.h"

struct params {
    /* Input parameters */
    char *InputFilename;
    double SmoothingRadius;
    double ExclusionRadius;
    double MaximumPointDensity;
    double DensityThreshold;

    double MinRadius;
    double MaxRadius;
    double ProfileMaxRadius;
};

struct units {
    double UnitLengthMetres;
    double UnitTimeSeconds;
    double UnitMassKilogram;
    double UnitTemperatureKelvin;
    double UnitCurrentAmpere;

    /* Physical constants in internal units */
    double SpeedOfLight;
    double GravityG;
    double hPlanck;
    double kBoltzmann;
    double ElectronVolt;
};

int readParams(struct params *parser, const char *fname);
int readUnits(struct units *us, const char *fname);

int cleanParams(struct params *parser);

int readFieldFile(double **box, int *N, double *box_len, const char *fname);

#endif
