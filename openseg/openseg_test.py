import numpy as np
import tensorflow as tf
import os
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import onnx
import tf2onnx


def read_bytes(path):
    '''Read bytes for OpenSeg model running.'''
    with tf.io.gfile.GFile(path, 'rb') as f:
        file_bytes = f.read()
    return file_bytes

def extract_openseg_img_feature(img_dir, openseg_model, img_size=None, regional_pool=True):
    '''Extract per-pixel OpenSeg features.'''

    text_emb = tf.zeros([1, 1, 768])
    # load RGB image
    np_image_string = read_bytes(img_dir)
    # run OpenSeg

    results = openseg_model.signatures['serving_default'](
        inp_image_bytes=tf.convert_to_tensor(np_image_string),
        inp_text_emb=text_emb
    )
    img_info = results['image_info']
    crop_sz = [
        int(img_info[0, 0] * img_info[2, 0]),
        int(img_info[0, 1] * img_info[2, 1])
    ]
    if regional_pool:
        image_embedding_feat = results['ppixel_ave_feat'][:, :crop_sz[0], :crop_sz[1]]
    else:
        image_embedding_feat = results['image_embedding_feat'][:, :crop_sz[0], :crop_sz[1]]
    
    if img_size is not None:
        feat_2d = tf.image.resize(image_embedding_feat, img_size, method='nearest')[0]
        feat_2d = tf.cast(feat_2d, dtype=tf.float16).numpy()
    else:
        feat_2d = tf.cast(image_embedding_feat[0], dtype=tf.float16).numpy()

    del results
    del image_embedding_feat

    return feat_2d

def convert_to_grayscale(image):
    grayscale_image = np.mean(image, axis=-1)
    return grayscale_image


import time

if __name__ == "__main__" :
    # tf.compat.v1.disable_eager_execution()
    saved_model_path = '/home/jihoon-epitone/Downloads/Pixel_aligned_VLM/openseg/openseg_exported_clip/' # openseg 모델 경로
    saved_model_path = os.path.realpath(os.path.expanduser(saved_model_path))
    print(saved_model_path)
    openseg_model = tf.saved_model.load(saved_model_path)
    output_names = ['image_embedding_feat', 'pixel_prediction']

    image_path = '/home/jihoon-epitone/Pictures/road.jpeg'    # 이미지 경로
    h = 183
    w = 275

    iter_num = 1000
    time_list = np.zeros([iter_num])

    for i in range(iter_num):
        start_time = time.time()
        a = extract_openseg_img_feature(image_path, openseg_model, img_size=[h, w], regional_pool=True)
        end_time = time.time()
        # print(a.shape)
        # 소요 시간 계산
        execution_time = end_time - start_time
        print(f"Execution Time: {execution_time:.4f} seconds")
        time_list[i] = execution_time
    
    print(np.mean(time_list))
    # # .pb 파일 로드
    # # pb_path = saved_model_path + '/saved_model.pb'
    # converter = trt.TrtGraphConverterV2(input_saved_model_dir=saved_model_path)

    
    # # TensorRT로 변환
    # print('--------------------------start converting-------------------------')
    # converter.convert()
    # print('-------------------------finish converting-------------------------')
    # trt_model_dir = "./openseg/openseg.trt"
    # converter.save(trt_model_dir)

    # ONNX 파일로 변환
    spec = (tf.TensorSpec((None, 5, 5, 3), tf.float32, name="input"),)  # 입력 텐서 스펙 정의
    model_proto, _ = tf2onnx.convert.from_tensorflow(openseg_model, input_signature=spec, output_path='output_model.onnx')





