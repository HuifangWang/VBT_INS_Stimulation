#! /bin/bash
# activate conda env with openmeeg installed 
__conda_setup="$('/Users/pault/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
eval "$__conda_setup"
conda activate openmeeg

export OMP_NUM_THREADS=7
proc_dir=$1

# calculate forward model with openmeeg
om_assemble -HeadMat ${proc_dir}/subject.geom ${proc_dir}/subject.cond ${proc_dir}/HeadMat.mat
om_minverser ${proc_dir}/HeadMat.mat ${proc_dir}/HeadMatInv.mat
om_assemble -Head2InternalPotMat ${proc_dir}/subject.geom ${proc_dir}/subject.cond ${proc_dir}/SEEG_sensors.txt ${proc_dir}/Head2IPMat.mat
om_assemble -DipSource2InternalPotMat ${proc_dir}/subject.geom ${proc_dir}/subject.cond ${proc_dir}/source_space.txt ${proc_dir}/SEEG_sensors.txt ${proc_dir}/Source2IPMat.mat
om_assemble -DipSourceMat ${proc_dir}/subject.geom ${proc_dir}/subject.cond ${proc_dir}/source_space.txt ${proc_dir}/SourceMat.mat
om_gain -InternalPotential ${proc_dir}/HeadMatInv.mat ${proc_dir}/SourceMat.mat ${proc_dir}/Head2IPMat.mat ${proc_dir}/Source2IPMat.mat ${proc_dir}/GainMat.mat

conda deactivate