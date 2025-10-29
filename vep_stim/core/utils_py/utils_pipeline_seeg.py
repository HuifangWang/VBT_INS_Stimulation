import nibabel as nib
import numpy as np
from scipy import ndimage

def create_seeg_nifti(seeg_xyz, T1_path, elec_image_path, use_affine, dilate_iters=3):
    dilate_iters = int(dilate_iters)
    xyz = np.genfromtxt(seeg_xyz, usecols=[0,1,2])
    n_elec = xyz.shape[0]
    img = nib.load(T1_path)
    if use_affine=="True":
        ijk = np.dot(np.linalg.inv(img.affine), np.hstack((xyz, np.ones((n_elec,1)))).T).astype(int)
        ijk = ijk[:3,:].T
    elif use_affine=="False":
        ijk = xyz.astype(int)

    elec_image_data = np.zeros_like(img.get_fdata())
    elec_image_data[ijk[:,0],ijk[:,1],ijk[:,2]] = 1
    elec_image_data = ndimage.binary_dilation(elec_image_data, iterations=dilate_iters).astype(int)
    elec_image = nib.Nifti1Image(elec_image_data, img.affine)
    nib.save(elec_image, elec_image_path)



if __name__ == '__main__':
    import sys
    cmd = sys.argv[1]
    eval(cmd)(*sys.argv[2:])