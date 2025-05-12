# This code contains the functions with activated tissue area, distance, showing in supp fig. A9.
# HuifangWang@ins, Jan, 2nd, 2025
def area_dynamic(results_dir, fname, vertex_areas,tavg_threshold,ts,tf):
    npz_fname = Path.joinpath(results_dir, fname + '.npz')
    #mp4_fname = Path.joinpath(results_dir, fname + '.mp4')
    #npz = np.load(npz_fname)
    tavg_raw = np.squeeze(np.load(npz_fname)['tavg'])
    '''
    if "time" in npz.keys():
        time = np.load(npz_fname)['time']
    else:
        time = np.load(npz_fname)['time_steps']
    '''
    tavg = tavg_raw[:,0,:-18]-tavg_raw[:,1,:-18]
    Area_calculation = []
    for it in np.arange(ts,tf):
        SEEG_actived = np.where(tavg[it]> tavg_threshold)
        Area_calculation.append(np.sum(vertex_areas[SEEG_actived]))
    return Area_calculation


def area_activetissue_dynamic(results_dir, fname, thd, vertex_areas, ts, tf):
    npz_fname = Path.joinpath(results_dir, fname + '.npz')
    # for each vertices, find the onset
    tavg_raw = np.squeeze(np.load(npz_fname)['tavg'])

    tavg = tavg_raw[:, 0, :-18] - tavg_raw[:, 1, :-18]

    focus_data = tavg[ts:tf, :]
    onset_v = np.where(np.max(focus_data, axis=0) > thd)[0]
    vidlist = []
    for vid in onset_v:
        indices = np.where(focus_data[:, vid] > thd)[0]
        tso_id = indices[0]
        tsf_id = indices[-1]
        vidlist.append([vid, tso_id, tsf_id])

    Area_calculation = []
    for its in np.arange(0, tf - ts):
        ntt = []
        for vid, tso_id, tsf_id in vidlist:
            if tso_id <= its <= tsf_id:
                ntt.append(vid)
        Area_calculation.append(np.sum(vertex_areas[ntt]))
    return Area_calculation


from scipy.spatial.distance import pdist


# def distance_activetissue_dynamic(results_dir, fname, thd, vertices,ts,tf):
def computate_ntt(results_dir, fname, thd, ts, tf):
    npz_fname = Path.joinpath(results_dir, fname + '.npz')
    # for each vertices, find the onset
    tavg_raw = np.squeeze(np.load(npz_fname)['tavg'])

    tavg = tavg_raw[:, 0, :-18] - tavg_raw[:, 1, :-18]

    focus_data = tavg[ts:tf, :]
    onset_v = np.where(np.max(focus_data, axis=0) > thd)[0]
    vidlist = []
    for vid in onset_v:
        indices = np.where(focus_data[:, vid] > thd)[0]
        tso_id = indices[0]
        tsf_id = indices[-1]
        vidlist.append([vid, tso_id, tsf_id])

    dist_calculation = []
    for its in np.arange(0, tf - ts):
        ntt = []
        for vid, tso_id, tsf_id in vidlist:
            if tso_id <= its <= tsf_id:
                ntt.append(vid)
        if len(ntt) < -1:
            dist_calculation.append(0)
        else:
            dist_calculation.append(ntt)
            # dist_calculation.append(np.max(pdist(vertices[ntt], metric='euclidean')))
        # print(its,ntt)
    return dist_calculation


def distance_activetissue_dynamic(ntt_v, vertices):
    dist_calculation = []
    for ntt in ntt_v:
        if len(ntt) < 2:
            dist_calculation.append(0)
        else:
            dist_calculation.append(np.max(pdist(merged_vertices[ntt], metric='euclidean')))
    return dist_calculation

def amp_dynamic(results_dir, fname,ts,tf):
    npz_fname = Path.joinpath(results_dir, fname + '.npz')
    #mp4_fname = Path.joinpath(results_dir, fname + '.mp4')
    #npz = np.load(npz_fname)
    tavg_raw = np.squeeze(np.load(npz_fname)['tavg'])
    tavg = tavg_raw[:,0,:-18]-tavg_raw[:,1,:-18]
    Amp_calculation = []
    for it in np.arange(ts,tf):
        Amp_calculation.append(np.mean(sorted(tavg[it], reverse=True)[:10]))
    return Amp_calculation

def compute_triangle_areas(vertices, triangles):
    """Calculates the area of triangles making up a surface."""
    tri_u = vertices[triangles[:, 1], :] - vertices[triangles[:, 0], :]
    tri_v = vertices[triangles[:, 2], :] - vertices[triangles[:, 0], :]
    tri_norm = np.cross(tri_u, tri_v)
    triangle_areas = np.sqrt(np.sum(tri_norm ** 2, axis=1)) / 2.0
    triangle_areas = triangle_areas[:, np.newaxis]
    return triangle_areas


def compute_vertex_areas(vertices, triangles):
    triangle_areas = compute_triangle_areas(vertices, triangles)
    vertex_areas = np.zeros((vertices.shape[0]))
    for triang, vertices in enumerate(triangles):
        for i in range(3):
            vertex_areas[vertices[i]] += 1./3. * triangle_areas[triang]

    return vertex_areas

''' 
First, read the surf files of the patients, then read the source-stimulated data. 
After that, you can obtain Supplementary Figure A9.

Due to space limitations, we are only sharing the code here. 
The source-stimulated data can be generated using Sim_SEEG_nf_dk.ipynb, Sim_SEEG_nf_lc.ipynb, Sim_SEEG_nf.ipynb, and Sim_TI_nf.ipynb.


'''