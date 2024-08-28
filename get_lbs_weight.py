import numpy as np
from os.path import join
from submodules import smplx
from utils.general_utils import load_masks, load_barycentric_coords, gen_lbs_weight_from_ori

assets_path = '/home/omj/GaussianAvatar'
data_path = '/home/omj/GaussianAvatar/assets'
resolution = 256

##smpl
#smpl_model = smplx.SMPL(model_path='/home/omj/GaussianAvatar/assets/smpl_files/smpl', batch_size = 1)

#ori_lbs_weight = smpl_model.lbs_weights

#flist_uv, valid_idx, uv_coord_map = load_masks(assets_path, resolution, body_model='smpl')
#bary_coords = load_barycentric_coords(assets_path, resolution, body_model='smpl')
#map_lbs = gen_lbs_weight_from_ori(ori_lbs_weight, bary_coords.cpu(), flist_uv.cpu()) #[, uvsize, uvsize, 24]
#np.save( join(data_path, "lbs_map_smpl_{}".format(str(resolution))), map_lbs.numpy())

##smplx
smpl_model = smplx.SMPLX(model_path='/home/omj/GaussianAvatar/assets/smpl_files/smplx', use_pca=False,num_pca_comps=45, flat_hand_mean=True, batch_size=1)

ori_lbs_weight = smpl_model.lbs_weights

flist_uv, valid_idx, uv_coord_map = load_masks(assets_path, resolution, body_model='smplx')
bary_coords = load_barycentric_coords(assets_path, resolution, body_model='smplx')
map_lbs = gen_lbs_weight_from_ori(ori_lbs_weight, bary_coords.cpu(), flist_uv.cpu()) #[, uvsize, uvsize, 24]
np.save( join(data_path, "lbs_map_smplx_{}".format(str(resolution))), map_lbs.numpy())