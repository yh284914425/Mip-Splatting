import torch
import os
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

if __name__ == "__main__":
    # 设置命令行参数解析器
    parser = ArgumentParser(description="测试脚本参数")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--output_ply", type=str, default="./output.ply")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("为 " + args.model_path + " 创建融合的ply")

    # 初始化系统状态 (RNG)
    safe_state(args.quiet)
    dataset = model.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
        
    gaussians.load_ply(os.path.join(dataset.model_path, "point_cloud", "iteration_30000", "point_cloud.ply"))
    gaussians.save_fused_ply(args.output_ply)
    
