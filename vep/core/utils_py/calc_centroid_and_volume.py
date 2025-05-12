def calc_centroid_and_volume(data, affine, voxel_size, label):
        r,c,s = np.where(data==label)
        vox_coords = np.vstack((r,c,s,np.ones((r.shape))))
        ras_coords = affine.dot(vox_coords)[:3,:].T
        centroid = ras_coords.mean(axis=0)
        volume   = voxel_size * len(r)

        print(label)
        print("nvox : "+str(len(r)))

        return centroid, volume 