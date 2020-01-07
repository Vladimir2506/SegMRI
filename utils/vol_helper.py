import numpy as np
import copy
# calculate the cube information
def fit_cube_param(vol_dim, cube_size, ita):
    dim = np.asarray(vol_dim)
    # cube number and overlap along 3 dimensions
    fold = dim / cube_size + ita
    ovlap = np.ceil(np.true_divide((fold * cube_size - dim), (fold - 1)))
    ovlap = ovlap.astype('int')

    fold = np.ceil(np.true_divide((dim + (fold - 1)*ovlap), cube_size))
    fold = fold.astype('int')

    return fold, ovlap


# decompose volume into list of cubes
def decompose_vol2cube(vol_data, batch_size, cube_size, n_chn, ita):
    cube_list = []
    # get parameters for decompose
    fold, ovlap = fit_cube_param(vol_data.shape, cube_size, ita)
    dim = np.asarray(vol_data.shape)
    # decompose
    r_s_list = []
    r_e_list = []
    c_s_list = []
    c_e_list = []
    h_s_list = []
    h_e_list = []
    for R in range(0, fold[0]):
        r_s = R*cube_size - R*ovlap[0]
        r_e = r_s + cube_size
        if r_e >= dim[0]:
            r_s = dim[0] - cube_size
            r_e = r_s + cube_size
        for C in range(0, fold[1]):
            c_s = C*cube_size - C*ovlap[1]
            c_e = c_s + cube_size
            if c_e >= dim[1]:
                c_s = dim[1] - cube_size
                c_e = c_s + cube_size
            for H in range(0, fold[2]):
                h_s = H*cube_size - H*ovlap[2]
                h_e = h_s + cube_size
                if h_e >= dim[2]:
                    h_s = dim[2] - cube_size
                    h_e = h_s + cube_size
                r_s_list.append(r_s)
                r_e_list.append(r_e)
                c_s_list.append(c_s)
                c_e_list.append(c_e)
                h_s_list.append(h_s)
                h_e_list.append(h_e)

    # print len(r_s_list)
    for ind in range(0,len(r_s_list),batch_size):
        cube_batch = np.zeros([batch_size, cube_size, cube_size, cube_size, n_chn]).astype('float32')
        for cnt in range(0,batch_size):
            if ind+cnt==len(r_s_list):
                break
            r_s = r_s_list[ind+cnt]
            r_e = r_e_list[ind+cnt]
            c_s = c_s_list[ind+cnt]
            c_e = c_e_list[ind+cnt]
            h_s = h_s_list[ind+cnt]
            h_e = h_e_list[ind+cnt]
            cube_temp = vol_data[r_s:r_e, c_s:c_e, h_s:h_e]
            cube_batch[cnt,:,:,:,0] = copy.deepcopy(cube_temp)
        cube_list.append(cube_batch)

    return cube_list


# compose list of label cubes into a label volume
def compose_label_cube2vol(cube_list, vol_dim, batch_size, cube_size, ita, class_n):
    # get parameters for compose
    fold, ovlap = fit_cube_param(vol_dim, cube_size, ita)
    # create label volume for all classes
    label_classes_mat = (np.zeros([vol_dim[0], vol_dim[1], vol_dim[2], class_n])).astype('int32')
    idx_classes_mat = (np.zeros([cube_size, cube_size, cube_size, class_n])).astype('int32')

    r_s_list = []
    r_e_list = []
    c_s_list = []
    c_e_list = []
    h_s_list = []
    h_e_list = []

    p_count = 0
    for R in range(0, fold[0]):
        r_s = R*cube_size - R*ovlap[0]
        r_e = r_s + cube_size
        if r_e >= vol_dim[0]:
            r_s = vol_dim[0] - cube_size
            r_e = r_s + cube_size
        for C in range(0, fold[1]):
            c_s = C*cube_size - C*ovlap[1]
            c_e = c_s + cube_size
            if c_e >= vol_dim[1]:
                c_s = vol_dim[1] - cube_size
                c_e = c_s + cube_size
            for H in range(0, fold[2]):
                h_s = H*cube_size - H*ovlap[2]
                h_e = h_s + cube_size
                if h_e >= vol_dim[2]:
                    h_s = vol_dim[2] - cube_size
                    h_e = h_s + cube_size
                r_s_list.append(r_s)
                r_e_list.append(r_e)
                c_s_list.append(c_s)
                c_e_list.append(c_e)
                h_s_list.append(h_s)
                h_e_list.append(h_e)

    for ind in range(0,len(r_s_list),batch_size):
        cube_batch = cube_list[p_count]
        for cnt in range(0,batch_size):
            if ind+cnt==len(r_s_list):
                break
            r_s = r_s_list[ind+cnt]
            r_e = r_e_list[ind+cnt]
            c_s = c_s_list[ind+cnt]
            c_e = c_e_list[ind+cnt]
            h_s = h_s_list[ind+cnt]
            h_e = h_e_list[ind+cnt]
            for k in range(class_n):
               idx_classes_mat[:,:,:,k] = (cube_batch[cnt,...]==k)
            label_classes_mat[r_s:r_e, c_s:c_e, h_s:h_e, :] = label_classes_mat[r_s:r_e, c_s:c_e, h_s:h_e, :] + idx_classes_mat
        p_count += 1

    # print 'label mat unique:'
    # print np.unique(label_mat)

    compose_vol = np.argmax(label_classes_mat, axis=3)
    # print np.unique(label_mat)

    return compose_vol


# compose list of probability cubes into a probability volumes
def compose_prob_cube2vol(cube_list, vol_dim, batch_size, cube_size, ita, class_n):
    # vol_dim = vol_dim.astype('int')
    cube_size = int(cube_size)

    # get parameters for compose
    fold, ovlap = fit_cube_param(vol_dim, cube_size, ita)
    # create label volume for all classes
    map_classes_mat = (np.zeros([vol_dim[0], vol_dim[1], vol_dim[2], class_n])).astype('float32')
    cnt_classes_mat = (np.zeros([vol_dim[0], vol_dim[1], vol_dim[2], class_n])).astype('float32')

    r_s_list = []
    r_e_list = []
    c_s_list = []
    c_e_list = []
    h_s_list = []
    h_e_list = []

    p_count = 0
    for R in range(0, fold[0]):
        r_s = R*cube_size - R*ovlap[0]
        r_e = r_s + cube_size
        if r_e >= vol_dim[0]:
            r_s = vol_dim[0] - cube_size
            r_e = r_s + cube_size
        for C in range(0, fold[1]):
            c_s = C*cube_size - C*ovlap[1]
            c_e = c_s + cube_size
            if c_e >= vol_dim[1]:
                c_s = vol_dim[1] - cube_size
                c_e = c_s + cube_size
            for H in range(0, fold[2]):
                h_s = H*cube_size - H*ovlap[2]
                h_e = h_s + cube_size
                if h_e >= vol_dim[2]:
                    h_s = vol_dim[2] - cube_size
                    h_e = h_s + cube_size
                r_s_list.append(r_s)
                r_e_list.append(r_e)
                c_s_list.append(c_s)
                c_e_list.append(c_e)
                h_s_list.append(h_s)
                h_e_list.append(h_e)

    for ind in range(0,len(r_s_list),batch_size):
        cube_batch = cube_list[p_count]
        for cnt in range(0,batch_size):
            if ind+cnt==len(r_s_list):
                break
            r_s = r_s_list[ind+cnt]
            r_e = r_e_list[ind+cnt]
            c_s = c_s_list[ind+cnt]
            c_e = c_e_list[ind+cnt]
            h_s = h_s_list[ind+cnt]
            h_e = h_e_list[ind+cnt]
            map_classes_mat[r_s:r_e, c_s:c_e, h_s:h_e, :] = map_classes_mat[r_s:r_e, c_s:c_e, h_s:h_e, :] + cube_batch[cnt,...]
            cnt_classes_mat[r_s:r_e, c_s:c_e, h_s:h_e, :] = cnt_classes_mat[r_s:r_e, c_s:c_e, h_s:h_e, :] + 1.0
        p_count += 1

    # elinimate NaN
    nan_idx = (cnt_classes_mat == 0)
    cnt_classes_mat[nan_idx] = 1.0
    # average
    compose_vol = map_classes_mat / cnt_classes_mat

    return compose_vol