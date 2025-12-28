<p align="center">

  <h1 align="center">Mip-Splatting: 无锯齿的3D高斯溅射</h1>
  <p align="center">
    <a href="https://niujinshuchong.github.io/">虞泽浩</a>
    ·
    <a href="https://apchenstu.github.io/">陈安培</a>
    ·
    <a href="https://github.com/hbb1">黄彬彬</a>
    ·
    <a href="https://tsattler.github.io/">Torsten Sattler</a>
    ·
    <a href="http://www.cvlibs.net/">Andreas Geiger</a>

  </p>
  <h2 align="center">CVPR 2024 最佳学生论文</h2>

  <h3 align="center"><a href="https://drive.google.com/file/d/1Q7KgGbynzcIEyFJV1I17HgrYz6xrOwRJ/view?usp=sharing">论文</a> | <a href="https://arxiv.org/pdf/2311.16493.pdf">arXiv</a> | <a href="https://niujinshuchong.github.io/mip-splatting/">项目页面</a>  | <a href="https://niujinshuchong.github.io/mip-splatting-demo/">在线查看器</a> </h3>
  <div align="center"></div>
</p>


<p align="center">
  <a href="">
    <img src="./media/bicycle_3dgs_vs_ours.gif" alt="Logo" width="95%">
  </a>
</p>

<p align="center">
我们为3D高斯溅射（3DGS）引入了一个3D平滑滤波器和一个2D Mip滤波器，消除了多种伪影并实现了无锯齿的渲染。
</p>
<br>

# 更新
我们集成了一个在[高斯不透明度场](https://niujinshuchong.github.io/gaussian-opacity-fields/)中提出的改进的致密化度量，它显著改善了新视角合成的结果，详情请参阅[论文](https://arxiv.org/pdf/2404.10772.pdf)。请下载最新的代码并重新安装`diff-gaussian-rasterization`来尝试。

# 安装
克隆仓库并使用以下命令创建一个anaconda环境
```
git clone git@github.com:autonomousvision/mip-splatting.git
cd mip-splatting

conda create -y -n mip-splatting python=3.8
conda activate mip-splatting

pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
conda install cudatoolkit-dev=11.3 -c conda-forge

pip install -r requirements.txt

pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn/
```

# 数据集
## Blender 数据集
请从[NeRF的官方Google Drive](https.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1)下载并解压nerf_synthetic.zip。然后用以下命令生成多尺度blender数据集
```
python convert_blender_data.py --blender_dir nerf_synthetic/ --out_dir multi-scale
```

## Mip-NeRF 360 数据集
请从[Mip-NeRF 360](https.jonbarron.info/mipnerf360/)下载数据，并向作者索取treehill和flowers场景。

# 训练与评估
```
# 在NeRF-synthetic数据集上进行单尺度训练和多尺度测试
python scripts/run_nerf_synthetic_stmt.py 

# 在NeRF-synthetic数据集上进行多尺度训练和多尺度测试
python scripts/run_nerf_synthetic_mtmt.py 

# 在mip-nerf 360数据集上进行单尺度训练和单尺度测试
python scripts/run_mipnerf360.py 

# 在mip-nerf 360数据集上进行单尺度训练和多尺度测试
python scripts/run_mipnerf360_stmt.py 
```

# 在线查看器
训练后，您可以使用以下命令将3D平滑滤波器融合到高斯参数中
```
python create_fused_ply.py -m {model_dir}/{scene} --output_ply fused/{scene}_fused.ply"
```
然后使用我们的[在线查看器](https://niujinshuchong.github.io/mip-splatting-demo)来可视化训练好的模型。

# 致谢
该项目建立在[3DGS](https://github.com/graphdeco-inria/gaussian-splatting)之上。请遵守3DGS的许可证。我们感谢所有作者的出色工作和代码库。

# 引用
如果您发现我们的代码或论文有用，请引用
```bibtex
@InProceedings{Yu2024MipSplatting,
    author    = {Yu, Zehao and Chen, Anpei and Huang, Binbin and Sattler, Torsten and Geiger, Andreas},
    title     = {Mip-Splatting: Alias-free 3D Gaussian Splatting},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {19447-19456}
}
```
如果您发现我们改进的致密化度量有用，请一并引用
```
@article{Yu2024GOF,
  author    = {Yu, Zehao and Sattler, Torsten and Geiger, Andreas},
  title     = {Gaussian Opacity Fields: Efficient High-quality Compact Surface Reconstruction in Unbounded Scenes},
  journal   = {arXiv:2404.10772},
  year      = {2024},
}
```
