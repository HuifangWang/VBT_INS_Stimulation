#! python
#%%
import nibabel as nib
import numpy as np
from scipy import ndimage
from nibabel.freesurfer.io import read_geometry, read_annot, write_annot, write_geometry
from collections import Counter
from numpy.linalg import inv

#%%

aseg_wm_val   = {"rh":41, "lh":2} # WM values for aseg
filled_wm_val = {"rh":127, "lh":255} # WM values for filled.mgz
aseg_gm_val   = {"rh":42, "lh":3} # GM values for aseg
aseg_HC_val   = {"rh":53, "lh":17} # HC values for aseg
hires_hc_gm_LUT_idx = [203,204,205,206,208,209,211,226] # Freesurfer LUT values for hires HC grey matter,
    
def adjust_WM_GM(aseg_path, filled_path, brain_path, wm_path, 
            rh_HC_seg_path, lh_HC_seg_path, rh_stats_path, lh_stats_path):

    # get GM mean value from freesurfer stats file
    brain_gm_val = {}
    for hemi, stats_path in zip(["rh","lh"],[rh_stats_path, lh_stats_path]):
        GM_mean = dict(np.genfromtxt(stats_path, dtype=str))["gray_mean"]
        brain_gm_val[hemi] = float(GM_mean)

    print(">>> load image data")
    aseg_img  = nib.load(aseg_path)
    aseg_data = aseg_img.get_fdata()
    aseg_data_orig = np.copy(aseg_data)

    filled_img  = nib.load(filled_path)
    filled_data = filled_img.get_fdata()

    brain_img  = nib.load(brain_path)
    brain_data = brain_img.get_fdata()

    wm_img  = nib.load(wm_path)
    wm_data = wm_img.get_fdata()

    print(">>> turn inferior lateral ventricle, choroid plexus and amygdala into WM")
    # such that the white surface touches the HC GM at every point
    for idx, hemi in zip([44,63,54,5,31,18],["rh","rh","rh","lh","lh","lh"]):
        mask = np.zeros_like(aseg_data_orig, dtype=bool)
        mask[aseg_data_orig==idx] = 1 

        aseg_data[mask]   = aseg_wm_val[hemi]
        filled_data[mask] = filled_wm_val[hemi]

        # wm.mgz and brain.mgz are not segmented images, i.e. not only "int" values, but smooth varying voxel intensities
        # dilate the WM mask into the WM of aseg to cover possible breaks in intensity 
        # (is this dilation step necessary ???)
        mask_dil = ndimage.binary_dilation(mask, iterations=3).astype(bool)
        mask_dil[aseg_data_orig!=aseg_wm_val[hemi]] = 0
        mask += mask_dil

        # TODO : maybe implement some smoothing ??
        # nib.processing.smooth_image()
        brain_data[mask] = 110 # freesurfer norms images wm to 110
        wm_data[mask]    = 110 
    
    print(">>> turn HC into 0")
    for idx in aseg_HC_val.values():
        mask = np.zeros_like(aseg_data_orig, dtype=bool)
        mask[aseg_data_orig==idx] = 1 
        aseg_data[mask]   = 0
        brain_data[mask]  = 0 
        filled_data[mask] = 0 # necessary ?
        wm_data[mask]     = 0 # necessary ?

    print(">>> dilate aseg GM and WM into former HC ONLY, to fill gap between GM or WM and hires HC")
    # hires HC seg is usually a little "smaller" than the lowres HC seg in aseg.presurf
    # the gap would otherwise cause problems in correct pial surface placement later on 
    # possibly adjust dilation iteration for optimal performance
    
    for hemi, HC_idx in aseg_HC_val.items(): # rh/lh
        print("first dilate GM")
        mask = np.zeros_like(aseg_data_orig, dtype=bool) 
        mask[aseg_data_orig==aseg_gm_val[hemi]] = 1
        mask = ndimage.binary_dilation(mask, iterations=5).astype(bool)
        mask[aseg_data_orig!=HC_idx] = 0
        aseg_data[mask]  = aseg_gm_val[hemi]
        brain_data[mask] = brain_gm_val[hemi]

        print("than dilate WM")
        mask = np.zeros_like(aseg_data, dtype=bool) 
        mask[aseg_data==aseg_wm_val[hemi]] = 1
        mask = ndimage.binary_dilation(mask, iterations=15).astype(bool)
        mask[aseg_data_orig!=HC_idx] = 0
        aseg_data[mask]   = aseg_wm_val[hemi] 
        filled_data[mask] = filled_wm_val[hemi]
        brain_data[mask]  = 110 # freesurfer norms images wm to 110
        wm_data[mask]     = 110

        print("dilate CSF to correct for some WM outliers")
        mask = np.zeros_like(aseg_data, dtype=bool) 
        mask[aseg_data==0] = 1
        mask = ndimage.binary_dilation(mask, iterations=6).astype(bool)
        mask[aseg_data_orig!=HC_idx] = 0
        aseg_data[mask]   = 0 
        filled_data[mask] = 0
        brain_data[mask]  = 0 
        wm_data[mask]     = 0


    print(">>> set dilated WM voxels which leak into hires HC to 0, ")
    # such that later the white surface ins't intersecting hires HC, but touches it
    # also create a hires HC GM from subiculum, CA and HC-tail and insert into aseg and brain
    for hires_HC_path, hemi in zip([rh_HC_seg_path, lh_HC_seg_path],["rh","lh"]):

        hires_HC_img  = nib.load(hires_HC_path)
        hires_HC_data = hires_HC_img.get_fdata()

        # set dilated WM voxels which leak into hires HC to 0
        # dont use fimbria part of hires HC [212], cause this might introduce a gap between HC GM and brain WM again
        mask = np.zeros_like(hires_HC_data, dtype=bool)
        for idx in [203,204,205,206,207,208,209,210,211,214,215,226]:
            if idx not in hires_hc_gm_LUT_idx:
                mask[hires_HC_data==idx]=1

        aseg_data[mask]   = 0
        filled_data[mask] = 0
        brain_data[mask]  = 0 
        wm_data[mask]     = 0

        # insert hires HC GM into aseg and brain
        mask = np.zeros_like(hires_HC_data, dtype=bool)
        for idx in hires_hc_gm_LUT_idx:
            mask[hires_HC_data==idx] = 1    
        
        aseg_data[mask]   = aseg_wm_val[hemi]
        filled_data[mask] = filled_wm_val[hemi]
        brain_data[mask]  = 110 
        wm_data[mask]     = 110

    print(">>> save images")
    new_aseg_img = nib.freesurfer.mghformat.MGHImage(aseg_data, aseg_img.affine, header=aseg_img.header)
    nib.save(new_aseg_img, aseg_path)

    new_filled_img = nib.freesurfer.mghformat.MGHImage(filled_data, filled_img.affine, header=filled_img.header)
    nib.save(new_filled_img, filled_path)

    new_brain_img = nib.freesurfer.mghformat.MGHImage(brain_data, brain_img.affine, header=brain_img.header)
    nib.save(new_brain_img, brain_path)

    new_wm_img = nib.freesurfer.mghformat.MGHImage(wm_data, wm_img.affine, header=wm_img.header)
    nib.save(new_wm_img, wm_path)
    
        
def adjust_cortex_HC_label(hemi, hiresHC_path, cortex_label_path, 
                            HC_label_path, surf_path, dist_thr):
    """
    The usual Freesufer command 'mri_label2label', which generates the ${hemi}.cortex.label 
    and the ${hemi}.cortex+hipamyg.label, doesn't work well with the new hires HC.
    I.e. some parts are not labeled HC when they should be, this causes problem in later pial surface placement.
    Therefore this function will set all vertices that are close to HC, but leave out Amygdala because we don't 
    need it for pial surface placement.
    """
    dist_thr = float(dist_thr)
    hiresHC_img    = nib.load(hiresHC_path)
    hiresHC_data   = hiresHC_img.get_fdata()
    hiresHC_affine = hiresHC_img.affine

    # read surface
    vertices, tris, volume_info = read_geometry(surf_path, read_metadata=True)
    vertices[:,0:3]  += volume_info["cras"] # include the offset for freesurfer surfaces

    # read current labels
    with open(cortex_label_path,"r") as f:
        first_line = f.readline()
    current_labels = np.genfromtxt(cortex_label_path, skip_header=2, usecols=0, dtype=int)

    # label new vertices
    new_label = np.zeros((vertices.shape[0]), dtype=bool)
    new_label[current_labels] = True

    for idx in hires_hc_gm_LUT_idx:
        print("hires HC idx : "+str(idx))
        r,c,s = np.where(hiresHC_data==idx)
        print("nvox : "+str(len(r)))
        hiresHC_vox_coords = np.vstack((r,c,s,np.ones((r.shape))))
        hiresHC_ras_coords = hiresHC_affine.dot(hiresHC_vox_coords)[:3,:].T
        
        # set all not yet labelled vertices that are close to hiresHC_ras_coords to 1
        vert_not_labeled = vertices[new_label==False,:]
        difference = vert_not_labeled - hiresHC_ras_coords[:,np.newaxis,:] # use numpy array broadcasting 
        distance   = np.sqrt(np.sum(difference**2,axis=2))
        new_label[np.where(new_label==False)[0][np.any(distance<dist_thr,axis=0)]] = True 
    
    #new_label[current_labels] = False # use this if you want to oupt ONLY the HC labels, i.e. w/o cortex

    # save new label file
    with open(HC_label_path,"w") as f:
        f.write(first_line)
        f.write(str(np.sum(new_label))+"\n")

        data = np.hstack((np.where(new_label)[0][:,np.newaxis], vertices[new_label,:], np.zeros((np.sum(new_label),1))))
        np.savetxt(f,data, fmt=["%d"]+ 3 * ["%.3f"] + ["%.10f"])

def gii_shape_to_FSlabel(surf_path, gii_path, label_path, old_label_path):
    # read surface
    vertices, tris, volume_info = read_geometry(surf_path, read_metadata=True)
    vertices[:,0:3]  += volume_info["cras"] # include the offset for freesurfer surfaces
    
    # read gii label file
    gii_img = nib.load(gii_path)
    gii_data = gii_img.agg_data().astype(int)
    r = np.where(gii_img.agg_data()==1)[0]
    new_label = np.zeros((vertices.shape[0]), dtype=bool)
    new_label[r] = True
    
    # get obligatory first line of old label file
    with open(old_label_path,"r") as f:
        first_line = f.readline()

    # save new label file
    with open(label_path,"w") as f:
        f.write(first_line)
        f.write(str(np.sum(new_label))+"\n")
        data = np.hstack((np.where(new_label)[0][:,np.newaxis], vertices[new_label,:], np.zeros((np.sum(new_label),1))))
        np.savetxt(f,data, fmt=["%d"]+ 3 * ["%.3f"] + ["%.10f"])


def adjust_aparc_aseg(aseg_path, aseg_vep_path, aseg_vep_wHC_path, rh_hiresHC_path, lh_hiresHC_path):
    """
    Insert hiresHC into aseg image
    """
    aseg_img  = nib.load(aseg_path)
    aseg_data = aseg_img.get_fdata()

    aseg_vep_img    = nib.load(aseg_vep_path)
    aseg_vep_data   = aseg_vep_img.get_fdata()

    # include offset to match VepHCFreeSurferColorLut.txt
    for hemi, offset, hiresHC_path in zip(["rh","lh"],[72073, 71073], [rh_hiresHC_path, lh_hiresHC_path]):
        hiresHC_data = nib.load(hiresHC_path).get_fdata()
        hiresHC_mask = np.ones_like(hiresHC_data, dtype=bool)
        hiresHC_gm_mask = np.zeros_like(hiresHC_data)

        for i, HC_idx in enumerate(hires_hc_gm_LUT_idx):
            print(HC_idx)
            mask = hiresHC_data==HC_idx
            hiresHC_gm_mask[mask] = offset + i
            hiresHC_mask[mask] = False

            mask[aseg_vep_data!=aseg_wm_val[hemi]] = False # insert only into WM of aseg
            aseg_vep_data[mask] = offset + i
            
        # WM has changed from aseg.presurf to final aparc+aseg.vep, because of the surface smoothing and topo correction
        # so some WM voxel might not be changed from the above step 
        # get WM voxel in hires HC and assign nearest neighbour from hires_HC_GM
        for i, HC_idx in enumerate([210,212,214,215]): # other hires HC indices next to GM
            print(HC_idx)
            mask = hiresHC_data==HC_idx
            hiresHC_mask[mask] = False
        
        aseg_vep_data_tmp = np.copy(aseg_vep_data)
        aseg_vep_data_tmp[hiresHC_mask] = 0
        r,c,s = np.where(aseg_vep_data_tmp==aseg_wm_val[hemi])
        xyz_wm = np.vstack((r,c,s)).T

        r,c,s = np.where(hiresHC_gm_mask>0)
        xyz_hc = np.vstack((r,c,s)).T

        difference = xyz_wm - xyz_hc[:,np.newaxis,:] # use numpy array broadcasting 
        distance   = np.sqrt(np.sum(difference**2,axis=2))
        argmin = np.argmin(distance,axis=0)
        aseg_vep_data[xyz_wm[:,0],xyz_wm[:,1],xyz_wm[:,2]] = hiresHC_gm_mask[xyz_hc[argmin,0],xyz_hc[argmin,1],xyz_hc[argmin,2]]

    # insert Amygdala and ventricles back into image, important for 5tt image and connectome generation
    # only insert structure back into WM, i.e. not into region where we now have hiresHC 
    for idx, hemi in zip([44,63,54,5,31,18],["rh","rh","rh","lh","lh","lh"]):
        mask = aseg_data==idx
        mask[aseg_vep_data!=aseg_wm_val[hemi]] = False
        aseg_vep_data[mask] = idx

    aseg_vep_wHC_img = nib.freesurfer.mghformat.MGHImage(aseg_vep_data, aseg_vep_img.affine, header=aseg_vep_img.header)
    nib.save(aseg_vep_wHC_img, aseg_vep_wHC_path)

def adjust_vepHC_annot(white_surf, white_normals, cortex_hip_filled_label, vepHC_aseg, vep_annot, vepHC_annot,
                        vepHC_lut):
    """
    Caution, some hardcoded numbers for labels in the code below !
    """
    # read vepHC_lut
    vepHC_lut_names = np.genfromtxt(vepHC_lut,dtype=str, usecols=1)
    vepHC_lut_index = np.genfromtxt(vepHC_lut,dtype=int, usecols=0)
    vepHC_lut_color = np.genfromtxt(vepHC_lut,dtype=int, usecols=[2,3,4,5])

    # read annot
    labels, ctab, names = read_annot(vep_annot)
    names = [n.decode('UTF-8') for n in names]
    ctab = ctab[:,:4]

    # read surface
    vertices, tris, volume_info = read_geometry(white_surf, read_metadata=True)
    vertices[:,0:3]  += volume_info["cras"] # include the offset for freesurfer surfaces

    # read surface normals 
    n_vert = vertices.shape[0]
    normals = np.genfromtxt(white_normals, skip_header=2, usecols=[0,1,2])[:n_vert]

    # read label
    corthip_labels = np.genfromtxt(cortex_hip_filled_label, skip_header=2, usecols=0, dtype=int)

    # read vepHC_aseg
    aseg_vepHC_img    = nib.load(vepHC_aseg)
    aseg_vepHC_data   = aseg_vepHC_img.get_fdata()

    # get vertices which are in cortex/hipp but have label -1
    # project from those vertices along the normal into space and get the label of that voxel
    mask = np.zeros_like(labels, dtype=bool)
    mask[corthip_labels] = True
    points = vertices[(labels==-1)*mask,:] + normals[(labels==-1)*mask,:] * (-0.5)
    points = np.hstack((points,np.ones((points.shape[0],1))))
    voxels = inv(aseg_vepHC_img.affine).dot(points.T)[:3,:].T # use inverse affine to go from XYZ to IJK
    voxels = np.round(voxels).astype(int)
    aseg_labels = (aseg_vepHC_data[voxels[:,0],voxels[:,1],voxels[:,2]]).astype(int)
    aseg_labels[aseg_labels<71001] = -1 # don't assign anything else but VepHC labels, i.e. don't assign WM or 0
    labels[(labels==-1)*mask] = aseg_labels

    # some vertices might still have a -1 after the above step, 
    # use most common label from its neighbours
    vert_wo_label = np.where((labels==-1)*mask)[0]
    n_vert_wo_label = len(vert_wo_label)
    last_n_vert_wo_label = -100
    while n_vert_wo_label!=0: 
        print("Number of vertices w/o label : " + str(n_vert_wo_label))
        for v_ind in vert_wo_label:
            vv_ind = tris[np.any(tris==v_ind,axis=1)].flatten()
            vv_ind = vv_ind[vv_ind!=v_ind]
            neighb_label = labels[vv_ind]
            neighb_label = neighb_label[neighb_label!=-1]
            if neighb_label.size != 0:
                mc = Counter(neighb_label).most_common()[0][0] 
                #print(mc)
                labels[v_ind] = mc
        
        vert_wo_label = np.where((labels==-1)*mask)[0]
        last_n_vert_wo_label = n_vert_wo_label
        n_vert_wo_label = len(vert_wo_label)

        if last_n_vert_wo_label == n_vert_wo_label:
            print("Warning : Not all vertices within cortex/hip mask could be labelled. %d vertices remain unlabeled" %n_vert_wo_label)
            break
    
    # annot labels must be incremental from -1, 1 to n cortical regions
    # aseg labels are from the VepHCFreesuferLut thus have values >71001
    # adjust labels, ctab and names to account for this issue
    unique_labels = np.unique(labels)
    new_label_counter = 72
    for l in unique_labels:
        if l >= 71001:
            print(l)
            idx = np.where(vepHC_lut_index==l)[0]
            l_name = vepHC_lut_names[idx][0].strip("Right-").strip("Left-")
            print(l_name)

            if l_name in names:
                i = names.index(l_name)
                labels[labels==l] = i 
                print(i)
            else:
                new_label_counter += 1
                print(new_label_counter)
                labels[labels==l] = new_label_counter
                names += [l_name]
                ctab = np.vstack((ctab, vepHC_lut_color[idx]))
    write_annot(vepHC_annot, labels, ctab, names, fill_ctab=True)


#adjust_aparc_aseg("/data/retrospective/neural_field_simulation/id039_mra/mri/aseg.presurf_orig_hires.mgz",
#                        "/data/retrospective/neural_field_simulation/id039_mra/mri/aparc+aseg.vep.mgz",
#                        "/data/retrospective/neural_field_simulation/id039_mra/mri/aparc+aseg.vepHC.mgz",
#                        "/data/retrospective/neural_field_simulation/id039_mra/mri/rh.hippoAmygLabels-T1.v21.FS60_fullsize.mgz",
#                        "/data/retrospective/neural_field_simulation/id039_mra/mri/lh.hippoAmygLabels-T1.v21.FS60_fullsize.mgz")

if __name__ == '__main__':
    import sys
    cmd = sys.argv[1]
    eval(cmd)(*sys.argv[2:])
