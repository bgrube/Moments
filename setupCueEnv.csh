#!/bin/tcsh

# load the required modules
module load gcc/10.2.0
module load python/3.9.5

# source the ROOT setup script
source /apps/root/6.28.06/setROOT_CUE-gcc10.2.0.csh

# add gcc includes to search path;
# this ensures that ACLiC can find omp.h
setenv CPATH "/apps/gcc/10.2.0/lib/gcc/x86_64-pc-linux-gnu/10.2.0/include/:${CPATH}"
