import os
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
import argparse
import skimage.transform

class ConvBlock3D(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ConvBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class VolumeNet(nn.Module):
    def __init__(self, block, layers, in_channels=2, num_features=128, zero_init_residual=False):
        super(VolumeNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv3d(in_channels, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_features)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, ConvBlock3D):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def build_volume_net(in_channels=2, num_features=128):
    return VolumeNet(ConvBlock3D, [2, 2, 2, 2], in_channels=in_channels, num_features=num_features)

def normalize_intensity(ct_volume, clip_min=-1000, clip_max=400):
    ct_volume = np.clip(ct_volume, clip_min, clip_max)
    ct_volume = (ct_volume - clip_min) / (clip_max - clip_min)
    return ct_volume

def resize_3d(volume, new_shape=(64,64,64), is_seg=False):
    if is_seg:
        resized = skimage.transform.resize(volume, new_shape, order=0, preserve_range=True, mode='constant')
        resized = np.round(resized)
    else:
        resized = skimage.transform.resize(volume, new_shape, preserve_range=True, mode='constant')
    return resized

def load_nifti_as_array(nifti_path):
    nii = nib.load(nifti_path)
    volume = nii.get_fdata(dtype=np.float32)
    return volume

def load_ct_seg_as_multichannel(ct_path, seg_path, new_shape=(64,64,64)):
    ct_vol = load_nifti_as_array(ct_path)
    ct_vol = normalize_intensity(ct_vol)
    ct_vol = resize_3d(ct_vol, new_shape=new_shape, is_seg=False)

    seg_vol = load_nifti_as_array(seg_path)
    seg_vol = np.clip(seg_vol, 0, 1)
    seg_vol = resize_3d(seg_vol, new_shape=new_shape, is_seg=True)

    multi_channel = np.stack([ct_vol, seg_vol], axis=0)
    return multi_channel

def extract_features(images_dir, genotype_csv, output_csv, model_ckpt=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_volume_net(in_channels=2, num_features=128).to(device)

    if model_ckpt and os.path.exists(model_ckpt):
        print(f"加载模型权重: {model_ckpt}")
        model.load_state_dict(torch.load(model_ckpt, map_location=device))
    model.eval()

    df_geno = pd.read_csv(genotype_csv)
    df_geno.set_index("Case ID", inplace=True)

    results = []

    patient_folders = [d for d in os.listdir(images_dir) 
                       if os.path.isdir(os.path.join(images_dir, d))]

    for pid in tqdm(patient_folders, desc="处理特征"):
        patient_path = os.path.join(images_dir, pid)
        ct_path = os.path.join(patient_path, f"{pid}_CT.nii.gz")
        seg_path = os.path.join(patient_path, f"{pid}_SEG.nii.gz")

        if (not os.path.exists(ct_path)) or (not os.path.exists(seg_path)):
            print(f"警告: {pid} 缺少CT或SEG文件，跳过")
            continue

        if pid not in df_geno.index:
            print(f"警告: {pid} 不在基因型CSV文件中，跳过")
            continue

        multi_vol = load_ct_seg_as_multichannel(ct_path, seg_path)
        vol_tensor = torch.from_numpy(multi_vol).unsqueeze(0).float().to(device)

        with torch.no_grad():
            feat = model(vol_tensor)
        feat = feat.cpu().numpy().flatten().tolist()

        egfr_status = df_geno.loc[pid, "EGFR mutation status"]

        result_dict = {
            "Case ID": pid,
            "EGFR mutation status": egfr_status
        }
        for i, val in enumerate(feat):
            result_dict[f"Feature_{i}"] = val

        results.append(result_dict)

    df_result = pd.DataFrame(results)
    df_result.to_csv(output_csv, index=False)
    print(f"特征已保存到: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="提取CT和分割图像的3D卷积特征")

    parser.add_argument("--images_dir", type=str,
                        default=r"/root/autodl-tmp/MedicalImagingGenomicsProject/data/images",
                        help="图像数据文件夹路径")
    parser.add_argument("--genotype_csv", type=str,
                        default=r"/root/autodl-tmp/MedicalImagingGenomicsProject/data/metadata/genotype_processed.csv",
                        help="基因型数据CSV文件路径")
    parser.add_argument("--output_csv", type=str,
                        default=r"/root/autodl-tmp/MedicalImagingGenomicsProject/data/metadata/features.csv",
                        help="特征输出CSV文件路径")
    parser.add_argument("--model_ckpt", type=str, default=None,
                        help="预训练模型权重路径（.pth文件，可选）")

    args = parser.parse_args()

    extract_features(
        images_dir=args.images_dir,
        genotype_csv=args.genotype_csv,
        output_csv=args.output_csv,
        model_ckpt=args.model_ckpt
    )