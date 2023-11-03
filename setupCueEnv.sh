module load gcc/10.2.0
module load python/3.9.5
source /apps/root/6.28.06/setROOT_CUE-gcc10.2.0.sh
# add gcc includes to search path; otherwise ACLiC cannot find omp.h
export CPATH="/apps/gcc/10.2.0/lib/gcc/x86_64-pc-linux-gnu/10.2.0/include/:${CPATH}"
