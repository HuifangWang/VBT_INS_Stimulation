import numpy as np
import struct
import math
import pyvista as pv
import matplotlib.pyplot as plt
import scipy as scp

def mean_dist(orig, v0, v1, v2):
    dist = np.linalg.norm(v0-orig) + np.linalg.norm(v1-orig) + np.linalg.norm(v2-orig)
    return dist/3

def read_fibres(tck_fname, n_fibres="all"):
    with open(tck_fname, "rb") as f:
        l = ""
        while l != "END":
            l = f.readline().decode('UTF-8').rstrip()
            if "datatype: " in l:
                datatype = l.strip("datatype: ")
            if "file: . " in l:
                offset = int(l.strip("file: . "))
            if "count: " in l:
                if n_fibres == "all":
                    n_fibres = int(l.strip("count: "))
        print(f"Datatype : {datatype}")
        print(f"Offset in bytes to get to fibres : {offset}")
        print(f"Number of fibres in tck file : {n_fibres}")
        
        f.seek(offset,0) # jump to byte which starts the vertex data
        fibres = []
        for i in range(n_fibres):
            fibres.append(read_fibre(f, format="<fff"))
    return fibres

def read_fibre(tck_file, format):
    fibre=[]
    v = list(struct.unpack(format, tck_file.read(12)))
    while not (math.isnan(v[0]) and math.isnan(v[1]) and math.isnan(v[2])):
        fibre.append(v)
        v = list(struct.unpack(format, tck_file.read(12)))
    return fibre

def polyline_from_points(points):
    poly = pv.PolyData()
    poly.points = np.array(points)
    the_cell = np.arange(0, len(points), dtype=np.int_)
    the_cell = np.insert(the_cell, 0, len(points))
    poly.lines = the_cell
    return poly

def create_fibre_mesh(fibres):
    points = []
    point_counter = 0
    cells = []
    for fibre in fibres:
        for v_idx, v in enumerate(fibre):
            points.append(v)
            if v_idx < len(fibre)-1:
                cells.append([2, point_counter, point_counter+1])
            point_counter += 1
    
    fibre_mesh = pv.PolyData()
    fibre_mesh.points = np.array(points)
    fibre_mesh.lines  = np.array(cells)
    return fibre_mesh

def apply_xfm_to_mesh(xfm, points):
    ones = np.ones((points.shape[0],1))
    points = np.hstack((points,ones))
    points  = xfm.dot(points.T)
    return points[:3,:].T

def ray_triangle_intersection(orig, dir, v0, v1, v2, epsilon=0.000001):
    """
    Ray-triangle-intersection-finding algorithm from Möller and Trumbore 1997.
    orig and dir are the origin and direction of the ray.
    v0, v1, v2 are the vertices which make up the triangle.
    """
    intersection_point = np.array([0.0,0.0,0.0])
    t, u, v = 0.0, 0.0, 0.0
    edge1 = v1-v0
    edge2 = v2-v0
    pvec = np.cross(dir, edge2)
    det  = edge1.dot(pvec) # if det is near zero the ray is parallel to the triangle
    # skip branch of the algorithm which culls backfacing triangles#
    if (det > -epsilon and det < epsilon):
        return False # no intersection possible
    inv_det = 1.0/det
    tvec = orig - v0
    u = tvec.dot(pvec) * inv_det
    if (u < 0 or u > 1):
        return False # intersection of ray with triangle plane outside of triangle
    
    qvec = np.cross(tvec,edge1)
    v = dir.dot(qvec) * inv_det
    if (v < 0 or u + v > 1):
        return False
    t = edge2.dot(qvec) * inv_det
    intersection_point = orig + t * dir

    return [intersection_point, t, u, v]

def get_begin_end_fibre_mesh_intersection(fibre, vertices, triangles):
    intersection_point_begin, t_begin, u_begin, v_begin, tri_ind_begin = None, None, None, None, None
    intersection_point_end, t_end, u_end, v_end, tri_ind_end = None, None, None, None, None
    # begin of fibre
    orig = fibre[1,:]
    dir  = fibre[0,:] - fibre[1,:]
    dir /= np.linalg.norm(dir)
    result_begin = fibre_mesh_intersection(vertices, triangles, orig, dir)
    if result_begin != False:
        intersection_point_begin, t_begin, u_begin, v_begin, tri_ind_begin = result_begin

    # end of fibre
    orig = fibre[-2,:]
    dir  = fibre[-1,:] - fibre[-2,:]
    dir /= np.linalg.norm(dir)
    result_end = fibre_mesh_intersection(vertices, triangles, orig, dir)
    if result_end != False:
        intersection_point_end, t_end, u_end, v_end, tri_ind_end = result_end
    
    return (intersection_point_begin, t_begin, u_begin, v_begin, tri_ind_begin,
            intersection_point_end,   t_end,   u_end,   v_end,   tri_ind_end)
        
    


def fibre_mesh_intersection(verts, tris, orig, dir ):
    result = False
    i = 0
    while (result == False and i<tris.shape[0]):
        tri = tris[i]
        v0, v1, v2 = verts[tri[0]], verts[tri[1]], verts[tri[2]]
        if mean_dist(orig, v0, v1, v2) < 10: # only test triangle if mean distance of triangle vertices to ray origin is less than some distance
            result = ray_triangle_intersection(orig, dir, v0, v1, v2)
            if (result != False):
                result.append(i)
                return result
        i += 1
    return result

def plot_square_sparse_mat(W, imsize=2**11, norm=None, save_fname=None):
    """ 
    Input 
    Wcoo - a scipy sparse square matrix
    imsize - resolution of plot in pixels

    Output a matplotlib imshow figure """
    plt.switch_backend('Qt5Agg')
    Wcoo = W.tocoo()
    assert Wcoo.shape[0] == Wcoo.shape[1] 

    zfac = (imsize-1)/Wcoo.shape[0]

    rows = np.rint(zfac*Wcoo.row).astype(int)
    cols = np.rint(zfac*Wcoo.col).astype(int)
    imdata = np.zeros((imsize,imsize))
    for r,c, _  in zip(rows, cols, Wcoo.data):
        imdata[r,c] += 1

    fig, ax = plt.subplots(figsize=(10,10))
    im = ax.imshow(imdata, interpolation='none', extent=[0,Wcoo.shape[0],0,Wcoo.shape[1]],norm=norm)
    fig.colorbar(im, label="Weights")
    fig.show()
    
    if not save_fname == None:
        fig.savefig(save_fname)


def construc_vert2vert_SC(vertices, triangles, intsec_points_begin, tri_idx_begin, intsec_points_end, tri_idx_end, weights):
    data = []
    row = []
    col = []
    # loop over fibres
    for ip_b, ti_b, ip_e, ti_e, w in zip(intsec_points_begin, tri_idx_begin, intsec_points_end, tri_idx_end, weights):
        # only assign a fibre to the connectome if both fibre ends are assigned
        if ti_b > -1 and ti_e > -1 : 
            v_idx = [] # --> store vertices which are connected by fibre
            # loop over both ends of the fibre
            for ip, ti in zip([ip_b, ip_e],[ti_b, ti_e]):
                if ti < triangles.shape[0]: # fibre is assigned to vertex
                    # assign fibre intersection point to closest vertex of the intersected triangle
                    v = vertices[triangles[ti]]
                    d = np.sum((v - ip)**2,axis=1)**(1/2)
                    v_idx.append(triangles[ti][np.argmin(d)])
                else: # else fibre is assigned to subcortical structure
                    # subcortical areas are appended to the end of the connectome
                    tmp = ti - triangles.shape[0] + vertices.shape[0]
                    v_idx.append(tmp)
            
            # append fibre twice to connectome, because it needs to be symmetric
            for rc, cr in zip([row,col],[col,row]):
                rc.append(v_idx[0])
                cr.append(v_idx[1])
                data.append(w)
    vert_conn_mat = scp.sparse.coo_matrix((data,(row, col)))
    return vert_conn_mat