import copy
import timm
import torch
import torchvision
import torch.nn as nn

from torchvision import transforms, models


def get_transforms(img_size):
    train_tfms = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor(),
                     # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                     transforms.RandomResizedCrop(size=img_size, scale=(0.7, 1.0), ratio=(0.7, 1.0)),
                     transforms.RandomHorizontalFlip(p=0.5),
                     transforms.RandomVerticalFlip(p=0.5),
                     transforms.RandomAffine(degrees = 0, translate = (0.3, 0.3)),
                     transforms.RandomRotation(degrees=(0, 180), fill=0),
                 ])
    test_tfms  = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor(),
                     # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                 ])

    return train_tfms, test_tfms


class CreateModel(nn.Module):
    def __init__(self, model_name, arch, is_cls):
        super(CreateModel, self).__init__()

        self.arch = arch
        self.is_cls = is_cls

        if (is_cls == True): 

            if (arch == 'arch_1'):
                model = timm.create_model(model_name, pretrained=True, num_classes=2)
                self.model = nn.Sequential(model, nn.Softmax(dim=-1))

            elif (arch == 'arch_2'):
                model = timm.create_model(model_name, pretrained=True, num_classes=2)
                self.base = list(model.children)[:2]
                self.head_1 = copy.deepcopy(list(model.children)[2])
                self.head_2 = copy.deepcopy(list(model.children)[2])
                self.head_3 = copy.deepcopy(list(model.children)[2])

        else:
            if (arch == 'arch_1'):
                self.model = timm.create_model(model_name, pretrained=True, num_classes=1)

            elif (arch == 'arch_2'):
                init_model = timm.create_model(model_name, pretrained=True, num_classes=1)
                init_model = nn.Sequential(init_model)
                init_model = list(init_model[0].children())[0]
                self.model = nn.Sequential(
                                *init_model,
                                nn.Conv2d(512, 256, (3, 3), stride=1),
                                nn.ReLU(),
                                nn.Conv2d(256, 128, (3, 3), stride=1),
                                nn.ReLU(),
                                nn.Conv2d(128, 64, (3, 3), stride=1),
                                nn.Flatten(),
                                nn.Linear(256, 1)
                                )

            elif (arch == 'arch_3'):
                weight_name = 'VGG16_BN_Weights.IMAGENET1K_V1'
                model = models.vgg16_bn(weights=weight_name)
                self.model = nn.Sequential(model, nn.Linear(1000, 1))

            elif (arch == 'arch_4'):
                weight_name = 'VGG16_BN_Weights.IMAGENET1K_V1'
                model = models.vgg16_bn(weights=weight_name)
                self.model = nn.Sequential(model, nn.Linear(1000, 1), nn.Sigmoid())
            
            elif (arch == 'arch_5'):
                pass


    # print(list(model[0].children())[:1])
    
    def forward(self, x):

        if (self.is_cls == True and self.arch == 'arch_2'):
            x = self.base(x)
            out_1 = self.head(x)
            out_2 = self.head(x)
            out_3 = self.head(x)
            return (out_1, out_2, out_3)

        else:
            return self.model(x)

    def forward_1(self, x):
        x        = self.feature_extractor(x)
        output_1 = self.cls_head(x)
        output_2 = self.reg_head(x)
        return (output_1, output_2)


def get_model(model_name, is_cls=True, model_params=[]):
    if (model_name.find('efficientnet-b') != -1):
        idx = model_name[-1]
        assert idx in ['0', '1', '2', '3', '4', '5', '6', '7'], print(f'[error] no such model {model_name}.')
        model = timm.create_model(f'tf_efficientnet_b{idx}', pretrained=True, num_classes=2)

        idx = int(idx)
        size_map   = [256, 240, 260, 300, 380, 456, 528, 600]
        batch_map  = [16, 16, 16, 16, 8, 4, 4, 4]
        img_size   = size_map[idx]
        batch_size = batch_map[idx]

    elif (model_name.find('efficientnetv2_') != -1):
        try:
            idx = model_name.split('_')[1]
        except:
            assert False, print(f'[error] no such model {model_name}')
        assert idx in ['s', 'm', 'l', 'xl'], print(f'[error] no such model {model_name}')

        if (is_cls):
            model = timm.create_model(f'tf_efficientnetv2_{idx}_in21ft1k', pretrained=True, num_classes=2)
        else:
            model = timm.create_model(f'tf_efficientnetv2_{idx}_in21ft1k', pretrained=True, num_classes=1, )

        size_map   = {'s': 300, 'm': 256, 'l': 256, 'xl': 256}
        batch_map  = {'s': 32,  'm': 16,  'l': 8,   'xl': 8}
        img_size   = size_map[idx]
        batch_size = batch_map[idx]

    elif (model_name.find('vgg16') != -1):
        assert model_name in ['vgg16', 'vgg16_bn'], print(f'[error] no such model {model_name}')

        if (is_cls):
            model = timm.create_model(model_name, pretrained=True, num_classes=2)
        else:
            model = timm.create_model(model_name, pretrained=True, num_classes=1)

        img_size = 256
        batch_size = 16

    elif (model_name.find('resnet') != -1):
        assert model_name in ['resnet50', 'resnet101'], print(f'[error] no such model {model_name}')
        model = timm.create_model(model_name, pretrained=True, num_classes=2)

        size_map  = {'resnet50': 224, 'resnet101': 244}
        batch_map = {'resnet50': 32, 'resnet101': 32}
        img_size = size_map[model_name]
        batch_size = batch_map[model_name]

    elif (model_name.find('vit_') != -1):
        try:
            idx = model_name.split('_')[1]
        except:
            assert False, print(f'[error] no such model {model_name}')
        assert idx in ['base', 'small', 'large'], print(f'[error] no such model {model_name}')

        model = timm.create_model(f'vit_{idx}_patch16_224_in21k', pretrained=True, num_classes=2)

        size_map= {'base': 224, 'small': 224, 'large': 224}
        batch_map = {'base': 16, 'small': 16, 'large': 8}
        img_size = size_map[idx]
        batch_size = batch_map[idx]

    elif (model_name.find('convnext_') != -1):
        try:
            idx = model_name.split('_')[1]
        except:
            assert False, print(f'[error] no such model {model_name}')
        assert idx in ['base', 'small', 'large', 'xlarge'], print(f'[error] no such model {model_name}')

        size_map = {'base': 384, 'small': 384, 'large': 384, 'xlarge':384}
        batch_map = {'base': 8, 'small': 16, 'large': 4, 'xlarge': 8}
        img_size = size_map[idx]
        batch_size = batch_map[idx]

        model = timm.create_model(f'convnext_{idx}_384_in22ft1k', pretrained=True, num_classes=2)

    elif (model_name.find('coatnet_') != -1):
        try:
            idx = model_name.split('_')[1]
        except:
            assert False, print(f'[error] no such model {model_name}')
        assert idx in ['0', '1'], print(f'[error] no such model {model_name}') 

        model = timm.create_model(f'coatnet_{idx}_rw_224', pretrained=True, num_classes=2)

        size_map = {'0': 224, '1': 224}
        batch_map = {'0': 32, '1': 16}
        img_size = size_map[idx]
        batch_size = batch_map[idx]

    else:
        assert False, print(f'[error] no such model {model_name}.')


    if (is_cls):
        model = nn.Sequential(
                    model,
                    nn.Softmax(dim=-1)
                )
    else: 
       model = nn.Sequential(
                    model,
                    # nn.ReLU
                    # nn.Sigmoid()
                )

    for i in range(7):
        print(i)
        print(list(model[0].children())[i])
        print('---\n')
    # print(list(model[0].children())[0])
    # print(len(list(model[0].children())))
    assert False

    return model, img_size, batch_size


def get_blablabla(model_name):
    if (model_name.find('efficientnet-b') != -1):
        assert model_name[-1] in ['0', '1', '2', '3', '4', '5', '6', '7']
        size_map  = [224, 240, 260, 300, 380, 456, 528, 600]
        batch_map = [16, 16, 16, 16, 8, 4, 4, 4]
        # v1_map    = [320, 320, 352, 384, 448, 512, 576, 2560]
        # v2_map    = [160, 160, 176, 192, 224, 256, 288, 1280]
        
        model_idx      = int(model_name[-1])
        img_size       = size_map[model_idx]
        batch_size     = batch_map[model_idx]
        # v1_feature_num = v1_map[model_idx]
        # v2_feature_num = v2_map[model_idx]


    elif (model_name.find('efficientnetv2') != -1):
        size_map = {'s': 300, 'm': 256, 'l': 256, 'xl': 256}
        batch_map = {'s': 32, 'm': 32, 'l': 8, 'xl': 8}
        img_size = size_map[model_name]
        batch_size = batch_map[model_name]

        model_params = []

    elif (model_name == 'vgg_16'):
        img_size = 224
        batch_size = 32

        model_params = []

    elif (model_name.find('resnet') != -1):
        size_map = {'resnet_50': 224, 'resnet_101': 244}
        batch_map = {'resnet_50': 32, 'resnet_101': 32}
        img_size = size_map[model_name]
        batch_size = batch_map[model_name]

        model_params = []

    elif (model_name.find('vit_') != -1):
        # vit_large_patch16_224, vit_large_patch16_224_in21k
        size_map= {'vit_base_patch16_224': 224, 'vit_base_patch16_224_in21k': 224, 'vit_large_patch16_224': 224,
                   'vit_large_patch16_224_in21k': 224}
        batch_map = {'vit_base_patch16_224': 16, 'vit_base_patch16_224_in21k': 16, 'vit_large_patch16_224': 16,
                     'vit_large_patch16_224_in21k': 16}
        img_size = size_map[model_name]
        batch_size = batch_map[model_name]

        model_params = []

    elif (model_name.find('convnext_') != -1):
        size_map = {'convnext_tiny': 224, 'convnext_small': 224, 'convnext_base': 384, 'convnext_large': 384}
        batch_map = {'convnext_tiny': 32, 'convnext_small': 32, 'convnext_base': 32, 'convnext_large': 32}
        img_size = size_map[model_name]
        batch_size = batch_map[model_name]

        model_params = []

    else:
        assert False, print(f'[error] no model name {model_name}.')
