#!/bin/bash
#PBS -S /bin/bash
#PBS -l nodes=1:ppn=8
#PBS -l walltime=00:05:00
#PBS -j operation
#PBS -N calcpi_dfranco
#PBS -m bae
#PBS -M dfranco24@unm.edu
module load gcc/11
module load openmpi
module load cmake
mpic++ -o calc_pi calc_pi.c
mpirun -n 1 ./calc_pi 1048576

#!/bin/bash
#PBS -S /bin/bash
#PBS -l nodes=1:ppn=8
#PBS -l walltime=00:05:00
#PBS -j operation
#PBS -N calcpi_dfranco
#PBS -m bae
#PBS -M dfranco24@unm.edu
module load gcc/11
module load openmpi
module load cmake
mpic++ -o calc_pi calc_pi.c
mpirun -n 8 ./calc_pi 1048576

#!/bin/bash
#PBS -S /bin/bash
#PBS -l nodes=2:ppn=8
#PBS -l walltime=00:05:00
#PBS -j operation
#PBS -N calcpi_dfranco
#PBS -m bae
#PBS -M dfranco24@unm.edu
module load gcc/11
module load openmpi
module load cmake
mpic++ -o calc_pi calc_pi.c
mpirun -n 16 -N 8 ./calc_pi 1048576

#!/bin/bash
#PBS -S /bin/bash
#PBS -l nodes=4:ppn=8
#PBS -l walltime=00:05:00
#PBS -j operation
#PBS -N calcpi_dfranco
#PBS -m bae
#PBS -M dfranco24@unm.edu
module load gcc/11
module load openmpi
module load cmake
mpic++ -o calc_pi calc_pi.c
mpirun -n 32 -N 8 ./calc_pi 1048576