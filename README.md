# Pixel_aligned_VLM

this repogetory is list of Pixel aligend VLM.

1. Openseg
2. Lseg
3. CLIP + SAM (Todo)

## Openseg

### requirements

- download pretraining model

https://drive.google.com/drive/folders/1IgHP6Xe-az3GOqMSvh7oDsIapPJkEmlS?usp=sharing

```
- openseg
    - openseg_exported_clip
        - variables
        - saved_model.pb
        - graph_def.pbtxt
    - openseg_test.py
```

- install tensorflow
```
pip install tensorflow[and-cuda]
```
### run
```
cd openseg
python openseg_test.py
```


This implementation is based on [opennerf](https://github.com/opennerf/opennerf), [openseg](https://github.com/tensorflow/tpu/tree/c3186a4386eb090f8f13bb07cd4bae0b149b4e01/models/official/detection/projects/openseg).


## Lseg

### requirements


- download pretraining model

https://drive.google.com/file/d/1ayk6NXURI_vIPlym16f_RG3ffxBWHxvb/view

```
- Lseg
    - checkpoints
        -demo_e200.ckpt
    - additional_utils
    - data
    - modules
    - ...
```


- gcc 9버전, cuda 11.3 환경 준비 (cuda 설치 화면에서 오직 cuda 만 설치할 것. 그래야 22.04환경에서도 사용 가능)
```
sudo apt -y install gcc-9 g++-9
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 9
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 9
sudo update-alternatives --config gcc   # 이후 나오는 화면에서 gcc-9에 해당하는 selection 번호 입력
sudo update-alternatives --config g++   # 이후 나오는 화면에서 g++-9에 해당하는 selection 번호 입력
```
```
wget https://developer.download.nvidia.com/compute/cuda/11.3.0/local_installers/cuda_11.3.0_465.19.01_linux.run
sudo sh cuda_11.3.0_465.19.01_linux.run
export PATH="/usr/local/cuda-11.3/bin:$PATH" && export LD_LIBRARY_PATH="/usr/local/cuda-11.3/lib64:$LD_LIBRARY_PATH"
```
- Lseg 환경 설치
```
cd Lseg
conda create -n lseg python=3.8
conda activate lseg

pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install git+https://github.com/zhanghang1989/PyTorch-Encoding/@331ecdd5306104614cb414b16fbcd9d1a8d40e1e  # this step takes >5 minutes
pip install pytorch_lightning==1.4.9

pip install git+https://github.com/openai/CLIP.git
pip install timm==0.5.4
pip install torchmetrics==0.6.0
pip install setuptools==59.5.0
pip install imageio matplotlib pandas six
```
만약 오류가 발생하는 경우 pytorch_lightning이 설치되면서 torch가 최신버전으로 다시 설치되었을 수 있음
torch를 1.9.1로 다시 설치하고 PyTorch-Encoding 설치

- ade20k 데이터셋 설치
```
python prepare_ade20k.py
unzip ../datasets/ADEChallengeData2016.zip
```
설혹 ade20k을 사용하지 않더라도 작동을 위해서는 받아야 함.


### run
If you want to extract LSeg per-pixel features and save locally, please check lseg_feature_extraction.py.

```bash
python lseg_feature_extraction.py --data_dir data/example/ --output_dir data/example_output/ --img_long_side 320
```
where 
- `data_dir` is the folder where contains RGB images
- `output_dir` is the folder where saves the corresponding LSeg features
- `img_long_side` is the length of the long side of your image. For example, for an image with a resolution of [640, 480], `img_long_side` is 640.

This implementation is based on [lseg_feature_extraction](https://github.com/pengsongyou/lseg_feature_extraction/tree/master), [Lseg](https://github.com/isl-org/lang-seg).


## CLIP + SAM

update later


## Pytorch model Time Measurement (mean of 1000 iterations)

CLIP model: 15.050ms
Lseg model (Vit): 63.520ms
Lseg model (Resnet): 21.677ms

### run
```
python lseg_zs_practice.py
```
설치가 안된 패키지가 있으면 pip로 설치하고 실행하면 된다.
