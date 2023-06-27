import sys
from pathlib import Path

import torch

sys.path.append('.')  # nopep8

import model.modules.feat_extractors.visual.vision_transformer as vits
from model.modules.feat_extractors.visual.s3d import S3D
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet
from utils.utils import check_if_file_exists_else_download


def get_resnet_layers(arch_name):
    if arch_name == 'resnet18':
        block = BasicBlock
        layers = [2, 2, 2, 2]
    elif arch_name == 'resnet34':
        block = BasicBlock
        layers = [3, 4, 6, 3]
    elif arch_name == 'resnet50':
        block = Bottleneck
        layers = [3, 4, 6, 3]
    elif arch_name == 'resnet101':
        block = Bottleneck
        layers = [3, 4, 23, 3]
    elif arch_name == 'resnet152':
        block = Bottleneck
        layers = [3, 8, 36, 3]
    else:
        raise NotImplementedError
    return block, layers

def load_state_dict_resnet(model, ckpt_path, prefix):
    if ckpt_path is not None:
        check_if_file_exists_else_download(ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
        model_type = ckpt.get('model_type', False)
        ckpt = ckpt.get('model', ckpt)
        # if we are using pretraining for correspondence, the weights are stored with different keys
        if model_type == 'AVCorrModel':
            # it is bit complicated but we need to rm the fc layer because
            # for the corr model we have not trained it. For other model_types, we do have fc
            model.fc = torch.nn.Identity()
            ckpt = {k.replace(prefix, ''): v for k, v in ckpt.items() if prefix in k}
        model.load_state_dict(ckpt)

def load_state_dict_dino(model, ckpt_path):
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
        model.load_state_dict(ckpt)



class ChannelLastLayerNorm(torch.nn.LayerNorm):
    '''Pytorch LayerNorm digests (N, *, C) while BatchNorm2d digests (N, C, H, W).
    This module applies the torch.nn.LayerNorm on the permuted tensor (channel last).'''

    def __init__(self, normalized_shape, eps: float = 0.00001, elementwise_affine: bool = True,
                 device=None, dtype=None) -> None:
        super().__init__(normalized_shape, eps, elementwise_affine, device, dtype)

    def forward(self, x):
        # (N, C, H, W) <-- permute <-- LayerNorm (N, H, W, C) <-- permute <--  (N, C, H, W)
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

class ResNetVisual(ResNet):

    def __init__(self, arch_name, num_classes, extract_features, ckpt_path=None, **kwargs):
        block, layers = get_resnet_layers(arch_name)
        super().__init__(block, layers, num_classes, **kwargs)
        self.extract_features = extract_features

        # load the ckpt
        load_state_dict_resnet(self, ckpt_path, prefix='vfeat_extractor.')

        # do not keep fc as they hold ~300k params
        if extract_features:
            self.avgpool = torch.nn.Identity()
            self.fc = torch.nn.Identity()

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.extract_features:
            return x

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return super().forward(x)

class ResNetAudio(ResNet):

    def __init__(self, arch_name, num_classes, extract_features, ckpt_path=None, **kwargs):
        block, layers = get_resnet_layers(arch_name)
        super().__init__(block, layers, num_classes, **kwargs)
        # replacing the old conv1 to the new one (RGB - 3; spectrogram - 1)
        conv1 = self.conv1
        self.conv1 = torch.nn.Conv2d(1, conv1.out_channels, conv1.kernel_size,
                                     conv1.stride, conv1.padding, bias=conv1.bias)
        self.extract_features = extract_features

        # load the ckpt
        load_state_dict_resnet(self, ckpt_path, prefix='afeat_extractor.')

        # do not keep fc as they hold ~300k params
        if extract_features:
            self.avgpool = torch.nn.Identity()
            self.fc = torch.nn.Identity()

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.extract_features:
            return x

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return super().forward(x)


class ResNet18AudioFeatures(ResNetAudio):

    # ckpt_path should default to None, otherwise when no pre-training is desired it will throw an error
    def __init__(self, ckpt_path=None):
        super().__init__(arch_name='resnet18', num_classes=308, extract_features=True, ckpt_path=ckpt_path)

    def forward(self, x):
        return super().forward(x)

class ResNet18VisualFeatures(ResNetVisual):

    # ckpt_path should default to None, otherwise when no pre-training is desired it will throw an error
    def __init__(self, ckpt_path=None):
        super().__init__(arch_name='resnet18', num_classes=1000, extract_features=True, ckpt_path=ckpt_path)

    def forward(self, x):
        B, Tv, C, H, W = x.shape
        # flattening the batch dim and temporal dim
        x = x.view(B*Tv, C, H, W)
        x = super().forward(x)
        _, c, h, w = x.shape
        x = x.view(B, Tv, c, h, w)
        return x

class DinoVisualFeatures(torch.nn.Module):

    def __init__(self, arch, patch_size, ckpt_path=None) -> None:
        super().__init__()
        if ckpt_path is not None:
            assert f'{patch_size}' in Path(ckpt_path).stem
        self.model = vits.__dict__[arch](patch_size=patch_size, num_classes=0)
        load_state_dict_dino(self.model, ckpt_path)
        self.arch = arch
        self.patch_size = patch_size
        self.ckpt_path = ckpt_path

    def forward(self, x):
        B, Tv, C, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, 'fixable by visualize_attention L174'
        # form output dimension
        h, w, c = (H // self.patch_size, W // self.patch_size, self.model.embed_dim)
        # flattening the batch dim and temporal dim
        x = x.view(B*Tv, C, H, W)
        # getting features from from the last layer (n=1), it is a list so: [-1]
        x = self.model.get_intermediate_layers(x, n=1)[-1]
        # getting rid of CLS token -> unflattening the map -> channel first
        x = x[:, 1:, :].reshape(B, Tv, h, w, c).permute(0, 1, 4, 2, 3)
        return x


class S3DVisualFeatures(S3D):

    # ckpt_path should default to None, otherwise when no pre-training is desired it will throw an error
    def __init__(self, ckpt_path=None):
        super().__init__(num_class=400, extract_features=True, ckpt_path=ckpt_path)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        x = self.base(x)
        x = x.permute(0, 2, 1, 3, 4)
        return x


if __name__ == '__main__':
    B = 2
    # vfeat_extractor = ResNet18VisualFeatures(model='torchvision.models.resnet18', num_classes=1000, pretrained=True)
    vfeat_extractor = ResNet18VisualFeatures(
        ckpt_path='./logs/feature_extractors/ImageNet1k/ResNetVisual-ImageNet1k.pt')
    # x = torch.rand(B, 100, 3, 224, 224)
    x = torch.rand(B, 80, 3, 224, 224)
    x = vfeat_extractor(x)
    print(x.shape)

    img_size = 224
    vfeat_extractor = DinoVisualFeatures(
        'vit_small', patch_size=16,
        ckpt_path='./model/modules/feat_extractors/visual/dino_deitsmall16_pretrain.pth')
    x = torch.rand(1, 80, 3, img_size, img_size)
    vfeat_extractor = vfeat_extractor
    x = vfeat_extractor(x)
    print(x.shape)

    # afeat_extractor = ResNet18AudioFeatures(
    #     ckpt_path='./logs/feature_extractors/22-02-15T14-20-02/ResNetAudio-22-02-15T14-20-02.pt')
    afeat_extractor = ResNet18AudioFeatures(
        ckpt_path='./logs/feature_extractors/22-04-04T21-00-19/ResNetAudio-22-04-04T21-00-19.pt')
    # x = torch.rand(B, 1, 257, 1551)
    x = torch.rand(B, 1, 257, 1379)
    x = afeat_extractor(x)
    print(x.shape)

    vfeat_extractor = S3DVisualFeatures(
        ckpt_path='./model/modules/feat_extractors/visual/S3D_kinetics400_torchified.pt')
    # x = torch.rand(B, 200, 3, 224, 224)
    x = torch.rand(B, 125, 3, 224, 224)
    x = vfeat_extractor(x)
    print(x.shape)
