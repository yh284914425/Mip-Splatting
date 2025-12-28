# 复制自 https://github.com/hjxwhy/mipnerf_pl/blob/dev/datasets/convert_blender_data.py
# 此文件修改自官方 mipnerf
import json
import os
import argparse
from os import path
import numpy as np
from PIL import Image


def load_renderings(data_dir, split):
    """从磁盘加载图像和元数据。"""
    f = 'transforms_{}.json'.format(split)
    with open(path.join(data_dir, f), 'r') as fp:
        meta = json.load(fp)
    images = []
    cams = []
    print('加载图像中')
    for frame in meta['frames']:
        fname = os.path.join(data_dir, frame['file_path'] + '.png')
        with open(fname, 'rb') as imgin:
            image = np.array(Image.open(imgin), dtype=np.float32) / 255.
        cams.append(frame['transform_matrix'])
        images.append(image)
    ret = {}
    ret['images'] = np.stack(images, axis=0)
    print('已加载所有图像，形状为', ret['images'].shape)
    ret['camtoworlds'] = np.stack(cams, axis=0)
    w = ret['images'].shape[2]
    camera_angle_x = float(meta['camera_angle_x'])
    ret['focal'] = .5 * w / np.tan(.5 * camera_angle_x)
    return ret


def down2(img):
    """盒式下采样"""
    sh = img.shape
    return np.mean(np.reshape(img, [sh[0] // 2, 2, sh[1] // 2, 2, -1]), (1, 3))


def convert_to_nerfdata(basedir, newdir, n_down):
    """将 Blender 数据转换为多尺度。"""
    if not os.path.exists(newdir):
        os.makedirs(newdir)
    splits = ['train', 'val', 'test']
    bigmeta = {}
    # 遍历数据集中的每个分割
    for split in splits:
        print('分割', split)
        # 加载所有内容
        data = load_renderings(basedir, split)

        # 保存所有图像
        imgdir = 'images_{}'.format(split)
        os.makedirs(os.path.join(newdir, imgdir), exist_ok=True)
        fnames = []
        widths = []
        heights = []
        focals = []
        cam2worlds = []
        lossmults = []
        labels = []
        nears, fars = [], []
        f = data['focal']
        print('保存图像中')
        for i, img in enumerate(data['images']):
            for j in range(n_down):
                fname = '{}/{:03d}_d{}.png'.format(imgdir, i, j)
                fnames.append(fname)
                fname = os.path.join(newdir, fname)
                with open(fname, 'wb') as imgout:
                    img8 = Image.fromarray(np.uint8(img * 255))
                    img8.save(imgout)
                widths.append(img.shape[1])
                heights.append(img.shape[0])
                focals.append(f / 2 ** j)
                cam2worlds.append(data['camtoworlds'][i].tolist())
                lossmults.append(4. ** j)
                labels.append(j)
                nears.append(2.)
                fars.append(6.)
                img = down2(img)

        # 创建元数据
        meta = {}
        meta['file_path'] = fnames
        meta['cam2world'] = cam2worlds
        meta['width'] = widths
        meta['height'] = heights
        meta['focal'] = focals
        meta['label'] = labels
        meta['near'] = nears
        meta['far'] = fars
        meta['lossmult'] = lossmults

        fx = np.array(focals)
        fy = np.array(focals)
        cx = np.array(meta['width']) * .5
        cy = np.array(meta['height']) * .5
        arr0 = np.zeros_like(cx)
        arr1 = np.ones_like(cx)
        k_inv = np.array([
            [arr1 / fx, arr0, -cx / fx],
            [arr0, -arr1 / fy, cy / fy],
            [arr0, arr0, -arr1],
        ])
        k_inv = np.moveaxis(k_inv, -1, 0)
        meta['pix2cam'] = k_inv.tolist()

        bigmeta[split] = meta

    for k in bigmeta:
        for j in bigmeta[k]:
            print(k, j, type(bigmeta[k][j]), np.array(bigmeta[k][j]).shape)

    jsonfile = os.path.join(newdir, 'metadata.json')
    with open(jsonfile, 'w') as f:
        json.dump(bigmeta, f, ensure_ascii=False, indent=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blender_dir", help="数据根路径", type=str,)
    parser.add_argument("--object_name", help="要制作多尺度的对象", type=str,
                        default=None)
    parser.add_argument("--out_dir", help="输出目录。", type=str)
    parser.add_argument("--n_down", help="要缩放的尺度数。", type=int, default=4)
    args = parser.parse_args()
    blenderdir = args.blender_dir
    outdir = args.out_dir
    n_down = args.n_down
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    scenes = os.listdir(blenderdir)
    if args.object_name is not None:
        scenes = [args.object_name]
    dirs = [os.path.join(blenderdir, f) for f in scenes]
    dirs = [d for d in dirs if os.path.isdir(d)]
    print(dirs)
    for basedir in dirs:
        print()
        newdir = os.path.join(outdir, os.path.basename(basedir))
        print('正在从', basedir, '转换到', newdir)
        convert_to_nerfdata(basedir, newdir, n_down)


if __name__ == '__main__':
    main()
