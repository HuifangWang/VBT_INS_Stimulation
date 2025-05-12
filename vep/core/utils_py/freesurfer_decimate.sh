#! /bin/bash

# make sure to use correct Freesurfer version
#export FREESURFER_HOME=/home/prior/vep_pipeline/freesurfer_v7
#source $FREESURFER_HOME/SetUpFreeSurfer.sh

# downsample the cortical mesh using freesurfer surface registration tool
# it will register and resample the cortical surface onto 
# that of fsaverage5 with lower resolution
# source /Users/pault/.bash_profile

sub=$1

echo "sub : "${sub}
sub_dir=$2
parc=$3
surf=$4
order=$5
for hemi in rh lh
do 
    echo $sub
    echo $sub_dir
    echo $hemi
    echo $parc
    # surface gemoetry
    mri_surf2surf \
        --sd ${sub_dir} --srcsubject ${sub} \
        --trgsubject ico --trgicoorder ${order} \
        --hemi ${hemi} \
        --sval-xyz ${surf}  \
        --tval ${sub_dir}/${sub}/surf/${hemi}.${surf}_ico${order}  \
        --tval-xyz ${sub_dir}/${sub}/mri/T1.mgz

    # annotation/parcellation
    mri_surf2surf \
        --sd ${sub_dir} --srcsubject ${sub} \
        --trgsubject ico --trgicoorder ${order} \
        --hemi ${hemi} \
        --sval-annot ${sub_dir}/${sub}/label/${hemi}.${parc}.annot \
        --tval ${sub_dir}/${sub}/label/${hemi}.${parc}_ico${order}.annot
done
