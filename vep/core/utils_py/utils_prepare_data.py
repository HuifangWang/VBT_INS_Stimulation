import mne
import pyvista as pv
import numpy as np
import zipfile 
import os
import shutil
import vtk
from vtk.util import numpy_support
from matplotlib.colors import ListedColormap
from nibabel.freesurfer.io import read_annot, read_geometry
import nibabel as nib
from pathlib import Path
from scipy import ndimage
from multiprocessing import Pool
from functools import partial
from utils_py.calc_centroid_and_volume import calc_centroid_and_volume
import utils_py.gain_matrix_seeg as gms

def save_connectivity(atlas, save_dir, nodes_image, mrtrix_lut_names, 
                        weights, lengths, region_map, nthreads):
    """
    Save a tvb format connectivity.zip file. 
    Create the centroids from the nodes image.
    Add "Unknown" / medial wall region to the connectome, to represent the vertices on the cortical surface.
    """
    # compute centers
    img = nib.load(nodes_image)
    voxel_size = np.prod(img.header.get_zooms())
    data = img.get_fdata()
    labels = np.unique(data)[1:] # leave out the 0 label
    
    centroids = np.zeros((len(labels),3))
    volumes   = np.zeros((len(labels),1))
    for i,label in enumerate(labels):
        r,c,s = np.where(data==label)
        vox_coords = np.vstack((r,c,s,np.ones((r.shape))))
        ras_coords = img.affine.dot(vox_coords)[:3,:].T
        centroids[i,:] = ras_coords.mean(axis=0)
        volumes[i]   = voxel_size * len(r)

        print(f"label : {label}")
        print(f"nvox  : {len(r)}")

    # with Pool(nthreads) as pool:
    #     f = partial(calc_centroid_and_volume, data, img.affine, voxel_size)
    #     output = pool.map(f, labels)
    # centroids = np.array([i[0] for i in output])
    # volumes   = np.array([i[1] for i in output])

    # add Unknown region, with 0 entries 
    volumes   = np.vstack(([0], volumes))   
    centroids = np.vstack(([0,0,0],centroids)) 
    names     = mrtrix_lut_names # contains "Unknown" already
    centres   = np.column_stack((names,centroids))
    weights_new = np.zeros((weights.shape[0]+1,weights.shape[1]+1 ))
    weights_new[1:, 1:] = weights
    lengths_new = np.zeros((lengths.shape[0]+1,lengths.shape[1]+1 ))
    lengths_new[1:, 1:] = lengths

    # create cortical
    cortical = np.zeros((len(names)))
    unique_labels = np.unique(region_map)
    for i in range(len(names)):
        if i in unique_labels: # if cortical
            cortical[i] = 1 

    # save and zip
    connectivity_dir = save_dir/f"connectivity.{atlas}"
    connectivity_dir.mkdir(exist_ok=True)
    np.savetxt(str(connectivity_dir/"weights.txt"),      weights_new, fmt="%f")
    np.savetxt(str(connectivity_dir/"tract_lengths.txt"),lengths_new, fmt="%f")
    np.savetxt(str(connectivity_dir/"volumes.txt"),      volumes, fmt="%f")
    np.savetxt(str(connectivity_dir/"centres.txt"),      centres, delimiter=" ", fmt="%s")
    np.savetxt(str(connectivity_dir/"cortical.txt"),     cortical, fmt="%d")
    #np.savetxt(str(connectivity_dir/"areas.txt"),        np.hstack(([0],a)), fmt="%f")
    #np.savetxt(str(connectivity_dir/"average_orientations.txt"), np.vstack(([0,0,1],avg_ori)), fmt="%f")
    shutil.make_archive(connectivity_dir, 'zip', connectivity_dir)
    shutil.rmtree(connectivity_dir)

    return centres

def modify_and_save_connectivity(sub_dir, save_dir):
    """
    Load old SC from vep pipeline and add "Unknown"/medial wall region for the neural field simulation.
    Modify the SC by adding a region "Unknown" to the top of the matix, for vetices with regionmapping == 0.
    """

    # load old SC
    with zipfile.ZipFile(sub_dir/"tvb"/"connectivity.vep.zip", mode="r") as sczip:
        print(sczip.namelist())
        with sczip.open('areas.txt',mode='r') as areas:
            a = np.genfromtxt(areas)
        with sczip.open('average_orientations.txt',mode='r') as average_orientations:
            avg_ori = np.genfromtxt(average_orientations)
        with sczip.open('centres.txt',mode='r') as centres:
            c = np.genfromtxt(centres, usecols=[1,2,3])
        with sczip.open('centres.txt',mode='r') as centres:
            names = np.genfromtxt(centres, usecols=[0], dtype="str")
        with sczip.open('cortical.txt',mode='r') as cortical:
            cort = np.genfromtxt(cortical)
        with sczip.open('weights.txt',mode='r') as weights:
            w = np.genfromtxt(weights)
        with sczip.open('tract_lengths.txt',mode='r') as tract_lengths:
            tl = np.genfromtxt(tract_lengths)
        with sczip.open('volumes.txt',mode='r') as volumes:
            vol = np.genfromtxt(volumes)

    # modify and save
    connectivity_new_dir = save_dir/"connectivity.vep_new"
    connectivity_new_dir.mkdir(exist_ok=True)

    c_new     = np.vstack(([0,0,0],c))
    names_new = np.hstack(("Unkown",names))
    centres_new = np.column_stack((names_new,c_new))

    # make 0 entries in SC for "Unkown"/ medial wall region
    w_new = np.zeros((w.shape[0]+1,w.shape[1]+1 ))
    w_new[1:, 1:] = w
    tl_new = np.zeros((tl.shape[0]+1,tl.shape[1]+1 ))
    tl_new[1:, 1:] = tl

    np.savetxt(str(connectivity_new_dir/"weights.txt"),      w_new, fmt="%f")
    np.savetxt(str(connectivity_new_dir/"tract_lengths.txt"),tl_new, fmt="%f")
    np.savetxt(str(connectivity_new_dir/"volumes.txt"),      np.hstack(([0],vol)), fmt="%f")
    np.savetxt(str(connectivity_new_dir/"centres.txt"),      centres_new, delimiter=" ", fmt="%s")
    np.savetxt(str(connectivity_new_dir/"cortical.txt"),     np.hstack((1,cort)), fmt="%d")
    np.savetxt(str(connectivity_new_dir/"areas.txt"),        np.hstack(([0],a)), fmt="%f")
    np.savetxt(str(connectivity_new_dir/"average_orientations.txt"), np.vstack(([0,0,1],avg_ori)), fmt="%f")

    shutil.make_archive(connectivity_new_dir, 'zip', connectivity_new_dir)
    shutil.rmtree(connectivity_new_dir)

def get_vert_tris(mesh, do_normals=True):
    vertices  = np.asarray(mesh.points,dtype=np.float64)
    triangles = np.asarray(mesh.faces, dtype=np.int32)
    triangles = np.delete(triangles, np.arange(0, triangles.size, 4)).reshape((-1,3))
    if do_normals:
        normals   = mesh["Normals"]
        return vertices, triangles, normals
    else :
        return vertices, triangles

def get_vert_tris_normals_from_pyvista_mesh(py_vista_mesh, mesh_name, do_normals=True):
    """Extract vertices, triangles and normals form the py_vista format."""
    return get_vert_tris(py_vista_mesh[mesh_name], do_normals)

def save_surfaces(py_vista_mesh, save_dir, surf="pial", resolution="_ico5"):
    """
    Save surfaces in TVB format.
    """
    # extract data from pyvista format
    lh_vertices, lh_triangles, lh_normals = get_vert_tris_normals_from_pyvista_mesh(py_vista_mesh, mesh_name="lh_"+surf+resolution)
    rh_vertices, rh_triangles, rh_normals = get_vert_tris_normals_from_pyvista_mesh(py_vista_mesh, mesh_name="rh_"+surf+resolution)

    # SC is ordered first left than right, so concatenate data from hemispheres correctly
    vertices  = np.vstack((lh_vertices, rh_vertices))
    triangles = np.vstack((lh_triangles, rh_triangles + lh_vertices.shape[0]))
    normals   = np.vstack((lh_normals, rh_normals))

    # save txt files and zip
    cort_surf_path = save_dir/("Cortex_"+surf+resolution)
    cort_surf_path.mkdir(exist_ok=True)

    np.savetxt(str(cort_surf_path/"triangles.txt"), triangles, fmt="%i")
    np.savetxt(str(cort_surf_path/"vertices.txt"), vertices, fmt="%f")
    np.savetxt(str(cort_surf_path/"normals.txt"), normals, fmt="%f")

    shutil.make_archive(cort_surf_path, 'zip', cort_surf_path)
    shutil.rmtree(cort_surf_path)

def get_cortical_regionmap(py_vista_mesh, surf="pial", resolution="_ico5", rh_offset=87):
    """
    Create a regionmap txt file for TVB.
    """  
    # save region mapping
    lh_labels = py_vista_mesh["lh_"+surf+resolution]["labels"]
    rh_labels = np.copy(py_vista_mesh["rh_"+surf+resolution]["labels"])
    rh_labels[rh_labels>0] += rh_offset
    labels    = np.hstack((lh_labels, rh_labels))
    return labels


def create_source_space(py_vista_mesh, surf, resolution, subcort_aseg, cortical_region_map,  
        freesurf_lut_aseg, freesurf_lut_names, mrtrix_lut_names, sub_dir):

    # cortical source space
    # combine all vertices and triangles and compute areas to create source space
    lh_vertices, lh_triangles, lh_normals = get_vert_tris_normals_from_pyvista_mesh(py_vista_mesh, mesh_name="lh_"+surf+resolution)
    rh_vertices, rh_triangles, rh_normals = get_vert_tris_normals_from_pyvista_mesh(py_vista_mesh, mesh_name="rh_"+surf+resolution)
    vertices   = np.vstack((lh_vertices, rh_vertices))
    triangles  = np.vstack((lh_triangles, rh_triangles + lh_vertices.shape[0]))
    normals    = np.vstack((lh_normals, rh_normals))
    tri_areas  = gms.compute_triangle_areas(vertices, triangles)
    vert_areas = gms.compute_vertex_areas(vertices, triangles)
    
    # subcortial source space
    # combine cortical and subcortical source space and create corresponding region map, to map sources to the connectome
    gain_region_map = cortical_region_map 
    if True:
        # add subcortical vertices and normals from subcortical surfaces
        py_vista_mesh = read_subcortical_surfaces(py_vista_mesh, subcort_aseg, sub_dir, reduction_rate = 0.8)
        for aseg in subcort_aseg:
            aseg_vertices, aseg_triangles, aseg_normals = get_vert_tris_normals_from_pyvista_mesh(py_vista_mesh, mesh_name=str(aseg)+"_lowres")
            aseg_tri_areas  = gms.compute_triangle_areas(aseg_vertices, aseg_triangles)
            aseg_vert_areas = gms.compute_vertex_areas(aseg_vertices, aseg_triangles)
            vertices   = np.vstack((vertices, aseg_vertices))
            triangles  = np.vstack((triangles, aseg_triangles + triangles.shape[0]))
            vert_areas = np.hstack((vert_areas, aseg_vert_areas))
            normals    = np.vstack((normals, aseg_normals))
    
            # add to regionmap
            # get position of aseg structure in connectome
            idx = np.where(freesurf_lut_aseg==aseg)[0]
            idx = np.where(mrtrix_lut_names == freesurf_lut_names[idx])[0]
            aseg_region_map = np.repeat([idx], len(aseg_vertices))
            gain_region_map = np.hstack((gain_region_map, aseg_region_map))
            print(len(aseg_region_map))
    
    else: # TODO: volumetric grid
        pass 
        # create volumetric grid in subcortical areas
        # vol_rc = create_subcortical_vol_grid(sub, sub_dir, subcort_aseg, freesurf_lut_names, freesurf_lut_aseg)
        # for src in vol_src:
        #     name = src["seg_name"]
        #     # apply transformation matrix
        #     points = src['rr'][src['inuse'].astype(bool)]
        #     # convert from Freesurfer surface RAS to RAS to align with the other surfaces and electrodes
        #     # also convert from units "m" to "mm"
        #     points = apply_trans(src["mri_ras_t"]["trans"], points) * 1000 
    
        #     # create random dipole orientation and normalise to magnitude = 1
        #     dipole_orientation = np.random.uniform(-1,1,size=(src["nuse"],3))
        #     norm = np.linalg.norm(dipole_orientation,axis=1)
        #     dipole_orientation = (dipole_orientation.T / norm).T
            
        #     name = src["seg_name"]
        #     points = py_vista_mesh[name].points
        #     np.savetxt(f, np.hstack((points,dipole_orientation)), fmt="%.10f")
    return gain_region_map, vertices, triangles, vert_areas, normals


def read_surfaces(sub_dir, surfs=["pial"], hemis=["rh", "lh"], resolutions=["", "_ico5"], parc="aparc.vepHC"):

    """
    Read surfaces, potentially downsampled, from Freesurfer SUBJECTDIR.
    sub_dir should be a Path object from pathlib.
    """
    cort_parc     = {} # parcellation
    py_vista_mesh = {}

    for surf in surfs:
        for hemi in hemis:
            for res in resolutions:
                # read vertices and tris
                surf_file = sub_dir / "surf" / (hemi + "." + surf + res)
                vertices, tris, volume_info = read_geometry(str(surf_file), read_metadata=True)
                vertices[:,0:3]  += volume_info["cras"] # include the offset for freesurfer surfaces
                if surf == "inflated" and hemi == "rh":
                    vertices[:,0] += 45
                elif surf == "inflated" and hemi == "lh":
                    vertices[:,0] -= 45
                py_vista_mesh[hemi+"_"+surf+res] = convert_to_pyvista_mesh(vertices, tris)

                # read parcellation
                cort_parc[hemi+'_labels'+res], color_lut, cort_parc[hemi+'_names'+res] = read_annot(str(sub_dir/"label"/(hemi+'.'+parc+res+'.annot')))
                cort_parc[hemi+'_names'+res] = [i.decode('UTF-8') for i in cort_parc[hemi+'_names'+res]]

                # create rgb array form colot_lut for view of parcellation
                color_lut = color_lut[:,:4]/255
                color_lut[0] = [0.5, 0.5, 0.5, 1] # change color of "subcortical" (i.e. medial wall) vertices
                color_lut[:,3] = 1 # set alpha to 1 
                cmap_lut = ListedColormap(color_lut)
                rgba = np.zeros((cort_parc[hemi+"_labels"+res].shape[0], 4))
                tmp_cort_parc = cort_parc[hemi+"_labels"+res]
                tmp_cort_parc[tmp_cort_parc==-1] = 0 # set subcoritcal voxels to 0
                for i in np.unique(tmp_cort_parc):
                    rgba[tmp_cort_parc==i] = color_lut[i]

                py_vista_mesh[hemi+"_"+surf+res]["rgba"]  = rgba
                py_vista_mesh[hemi+"_"+surf+res]["labels"] = tmp_cort_parc
                
    return py_vista_mesh, cort_parc


def read_subcortical_surfaces(py_vista_mesh, aseg_numbers, sub_dir, parc="aparc.vepHC", reduction_rate = 0.8, use_ascii=False):

    if py_vista_mesh == None: # create one
        py_vista_mesh = {}

    if parc == "aparc.vepHC":
        aseg2srf_dname = "vepHC"
    elif parc == "aparc.vep":
        aseg2srf_dname = "vep"
    else:
        raise Exception('Unexpected parc')

    for aseg in aseg_numbers:
        if use_ascii: # when using ascii files
            aseg_file = sub_dir/"aseg2srf"/parc/("aseg_"+str(aseg).zfill(5)+".srf")
            with aseg_file.open("r") as f:
                    f.readline()
                    nverts, ntriangs = [int(n) for n in f.readline().strip().split(' ')]
            vert = np.genfromtxt(str(aseg_file), dtype=float, skip_header=2, skip_footer=ntriangs, usecols=(0, 1, 2))
            nverts = vert.shape[0]
            tris = np.genfromtxt(str(aseg_file), dtype=int, skip_header=2 + nverts, usecols=(0, 1, 2))
        else :
            # when using freesurfer files
            aseg_file = sub_dir/"aseg2srf"/parc/("aseg_"+str(aseg).zfill(5))
            vert, tris = read_geometry(aseg_file)
        
        py_vista_mesh[aseg] = convert_to_pyvista_mesh(vert,tris)
        py_vista_mesh[str(aseg)+"_lowres"] = py_vista_mesh[aseg].decimate(reduction_rate)
        py_vista_mesh[str(aseg)+"_lowres"].compute_normals(cell_normals=False, point_normals=True, inplace=True)
    
    return py_vista_mesh


def convert_to_pyvista_mesh(vertices, triangles):
    """
    Take triangles and vertices as [nx3] and [mx3] arrays.
    To triangles add one row with number 3, to indicate for pyvista that face is a triangle
    Return a pyvista mesh
    """
    faces_ = np.ones((triangles.shape[0],triangles.shape[1]+1),dtype=int)
    faces_[:,:1] *= 3
    faces_[:,1:]  = triangles
    return pv.PolyData(vertices,faces_.flatten())


def write_BEM_to_BrainVisa(sub, sub_dir, openmeeg_proc_dir, reduction_rate = 0.8,
                            py_vista_mesh=None):
    """
    Read the freesurfer BEM surfaces and save them in BrainVisa format for OpenMEEG.
    Surfaces get slightly downsampled in order to work with OpenMEEG.
    """
    if py_vista_mesh == None:
        py_vista_mesh = {}

    for bem in ["brain", "inner_skull","outer_skull", "outer_skin"]:
        vertices, tris, volume_info = read_geometry(str(sub_dir/sub/"bem"/(bem+".surf")), read_metadata=True)
        vertices[:,0:3]  += volume_info["cras"]  # include the offset for freesurfer surfaces, tp align with SEEG
        py_vista_mesh[bem] = convert_to_pyvista_mesh(vertices, tris)
        
        # decimate and compute surface normals
        py_vista_mesh[bem+"_lowres"] = py_vista_mesh[bem].decimate(reduction_rate)#, preserve_topology=True)
        py_vista_mesh[bem+"_lowres"].compute_normals(cell_normals=False, point_normals=True, inplace=True)
        vertices, triangles, normals = get_vert_tris_normals_from_pyvista_mesh(py_vista_mesh, bem+"_lowres")
        n_tris = triangles.shape[0]
        n_vertices = vertices.shape[0]

        # write to BrainVisa format
        bem_tri_file = openmeeg_proc_dir/(bem+".tri")
        bem_tri_file.unlink(missing_ok=True)
        with bem_tri_file.open(mode="ab") as f:
            #vertices and normals
            f.write(b"- %d \n" %n_vertices )
            np.savetxt(f, np.hstack((vertices, normals)), fmt="%.10f")
            
            # triangles
            f.write(b"- %d %d %d \n" %(n_tris, n_tris, n_tris))
            np.savetxt(f,triangles, fmt="%d")
    
    return py_vista_mesh

def create_subcortical_vol_grid(sub, sub_dir, vep_subcort_aseg, 
                        vep_freesurf_lut_names, vep_freesurf_lut_aseg):
    
    # create dict for mne to extract volume grids from structures
    labels_vol = {}
    for aseg in vep_subcort_aseg:
        name = vep_freesurf_lut_names[vep_freesurf_lut_aseg == aseg][0]
        labels_vol[name] = aseg
        
    # get volume grid
    vol_src = mne.setup_volume_source_space(subject=sub, mri=str(sub_dir/sub/"mri"/"aparc+aseg.vep.mgz"), pos=5,  
                                            volume_label=labels_vol, subjects_dir=str(sub_dir), bem=None, surface=None,
                                            add_interpolator=True, verbose=True)
    return vol_src

    # for src in vol_src:
    #     name = src["seg_name"]
    #     # apply transformation matrix
    #     points = src['rr'][src['inuse'].astype(bool)]
    #     # convert from Freesurfer surface RAS to RAS to align with the other surfaces and electrodes
    #     # also convert from units "m" to "mm"
    #     points = apply_trans(src["mri_ras_t"]["trans"], points) * 1000 


    #     # create random dipole orientation and normalise to magnitude = 1
    #     dipole_orientation = np.random.uniform(-1,1,size=(src["nuse"],3))
    #     norm = np.linalg.norm(dipole_orientation,axis=1)
    #     dipole_orientation = (dipole_orientation.T / norm).T
        
    #     name = src["seg_name"]
    #     points = py_vista_mesh[name].points
    #     np.savetxt(f, np.hstack((points,dipole_orientation)), fmt="%.10f")
        
def normals_from_mesh(mesh,flip_normals=True, auto_orient_normals=True, consistent_normals=True):
    """
    Return pyvista mesh of arrows, being the extracted normal vectors of the input mesh
    """
    mesh.compute_normals(cell_normals=False, point_normals=True, inplace=True, auto_orient_normals=auto_orient_normals, consistent_normals=consistent_normals, flip_normals=flip_normals)
    mesh_vert = pv.PolyData(mesh.points)
    mesh_vert["Normals"] = mesh["Normals"]
    return mesh_vert.glyph(orient="Normals", geom=pv.Arrow(), scale=False)


if __name__ == '__main__':
    import sys
    cmd = sys.argv[1]
    eval(cmd)(*sys.argv[2:])
