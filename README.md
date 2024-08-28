# GaussianAvatar_mmai
- [Files to download for executing](#files-to-download-for-executing)
- [Needed files for executing](#needed-files-for-executing)
- [Code for change](#code-for-change)

## Files to download for executing
- [SMPL](https://smpl.is.tue.mpg.de/) & [SMPL-X](https://smpl-x.is.tue.mpg.de/) model

  다운 받은 파일은 `./assets/smpl_files`에 아래와 같은 형식으로 저장 
  ```
  smpl_files
   └── smpl
     ├── SMPL_FEMALE.pkl
     ├── SMPL_MALE.pkl
     └── SMPL_NEUTRAL.pkl
   └── smplx
     ├── SMPLX_FEMALE.npz
     ├── SMPLX_MALE.npz
     └── SMPLX_NEUTRAL.npz
  ```

- Data
  [OneDrive](https://hiteducn0-my.sharepoint.com/:f:/g/personal/lx_hu_hit_edu_cn/EsGcL5JGKhVGnaAtJ-rb1sQBR4MwkdJ9EWqJBIdd2mpi2w?e=KnloBM)

  `assets`: `./assets`에 저장

  `data_checkpoints_4peoplesnapshot`: `./`에 `data` 폴더 만들어서 저장

  `gs_data`: `data` 폴더에 저장

  `pretrainded_models`: `./`에 `output`폴더 만들어서 저장
  

- transforms.py

  `./`에 저장
  
- scripts/custom

  `./scripts/custom`에 저장
  
- sam_vit_h_4b8939.pth

  [segment-anything](https://github.com/facebookresearch/segment-anything/tree/main) 에서 제공하는 ViT-H SAM model

  `$HOME/third-party/segment-anything/ckpts`에 파일 저장
  
- get_lbs_weight.py

  `./`에 저장


## Needed files for executing
### train.py
- given data
  ```
  m4c_processed
   └── train
     └── images
        ├── 00000000.png
        ├── ...
     └── masks
        ├── 00000000.png
        ├── ...
     ├── cam_parms.npz
     ├── smpl_parms.pth
     ├── query_posemap_512_cano_smpl.npz
     └── smpl_cano_joint_mat.pth
   └── test
     └── images
        ├── 00000660.png
        ├── ...
     └── masks
        ├── 00000660.png
        ├── ...
     ├── cam_parms.npz
     ├── smpl_parms.pth
     ├── query_posemap_512_cano_smpl.npz
     └── smpl_cano_joint_mat.pth
  ```
- with new data

  - images들 혹은 `ffmpeg`을 이용하여 frame cut한 images들만 존재하는 경우
  
     [InstantAvatar](https://github.com/tijiang13/InstantAvatar) 에서 제공하는 `scripts/custom/process-sequence.sh`를 사용.
      ```
      data/subject
       └── images
      ```

  - `bash scripts/custom/process-sequence.sh`을 실행
  
    `keypoints.npy`, `masks`, `poses.npz`,`cameras.npz`, `poses_optimized.npz`, `output.mp4`를 얻게 됨. 그 중 mask와 pose 정보사용.
    
    `run-openpose-bin.sh`의 `line 15` 경로 설정 필요.
    ```
    data/subject
     ├── images
     ├── masks                //run_sam.py & extract-largest-connected-components.py
     ├── cameras.npz          //run_romp.py
     └── poses_optimized.npz  //refine-smpl.py
    ```
  
  - `cd scripts & python sample_romp2gsavatar.py`을 실행
  
    `train`과 `test` 데이터에서 `cam_parms.npz`와 `smpl_parms.pth`를 얻게 됨.
  
    `sample_romp2gsavatar.py`의 `line 50 & 51` 경로 설정 필요.
      ```
      data/subject
       ├── images
       ├── masks
       ├── cam_parms.npz
       └── smpl_parms.pth
      ```

  - `python gen_pose_map_cano_smpl.py`을 실행

    `query_posemap_%%_cano_smpl.npz`와 `smpl_cano_joint_mat.pth`를 얻게 됨.
  
    `gen_pose_map_cano_smpl.py`의 `line 103 & 106` 경로 설정 필요.
     ```
      data/subject
       ├── images
       ├── masks
       ├── cam_parms.npz
       ├── smpl_parms.pth
       ├── query_posemap_%%_cano_smpl.npz
       └── smpl_cano_joint_mat.pth
      ```

   - data의 preprocessing이 끝났으면 train 시키기 가능.
      ```
      python train.py -s $path_to_data/$subject -m output/$subject --train_stage 1 
      ```
  
### eval.py
  ```
  python eval.py -s $path_to_data/$subject -m output/$subject --epoch 200
  ```
  
### render_novel_pose.py
  ```
  python render_novel_pose.py -s $path_to_data/$subject -m output/$subject--epoch 200
  ```

  

## Code for change
- `arguments/__init__.py` `line 8`
  
  `pytorch3d` 설치 이슈로 transforms와 관련한 함수들을 가져와 `transforms.py`에 만들고 상위폴더로부터 불러와 import 하도록 코드 수정.
  ```
  # from pytorch3d import transforms   

  sys.path.append(os.path.join(os.path.dirname(__file__), ".."))    
  import transforms
  ```

- `model/avatar_model.py` `line 76 & 87`
  
  `RuntimeError : indices should be either on cpu or on the same device as the indexed tensor (cpu)`와 같은 에러가 발생하게 된다면 아래와 같이 코드 수정.
  ```
  #query_points = query_map[valid_idx, :].cuda().contiguous()
  query_points = query_map[valid_idx.cpu(), :].cuda().contiguous()    
  
  #self.query_lbs = query_lbs[valid_idx, :][None].expand(self.batch_size, -1, -1).cuda().contiguous()
  self.query_lbs = query_lbs[valid_idx.cpu(), :][None].expand(self.batch_size, -1, -1).cuda().contiguous()   
  ```

- `gaussian_renderer/__init__.py` `line 40`
  ```
  rendered_image = rasterizer(        
        means3D = points,
        means2D = screenspace_points,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    
  return rendered_image[0]   
  ```



