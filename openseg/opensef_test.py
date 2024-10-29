import numpy as np
import tensorflow as tf
import os


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


if __name__ is "__main__" :
    saved_model_path = 'openseg_exported_clip' # openseg 모델 경로
    saved_model_path = os.path.realpath(os.path.expanduser(saved_model_path))
    print(saved_model_path)
    openseg_model = tf.saved_model.load(saved_model_path)

    image_path = '/home/user/data/dataset/Replica/room0/results/frame001083.jpg'    # 이미지 경로
    h = 680
    w = 1200

    a = extract_openseg_img_feature(image_path, openseg_model, img_size=[h, w], regional_pool=True)





