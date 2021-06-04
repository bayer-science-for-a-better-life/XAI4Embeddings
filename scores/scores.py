import argparse
import time
from typing import List
from scipy.special import softmax
import torch
import torchvision.models as models
import torch.nn as nn
import numpy as np
from scores.attributions import compute_importance_features
from pytorch_grad_cam import GradCAM


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def randomize_layer(net: nn.Module, layer_to_rand: int):
    layers = [module for module in net.modules() if type(module) in [torch.nn.modules.conv.Conv2d]]
    layer_to_randomize = layers[layer_to_rand]
    print('Re-initializaing layer', layer_to_randomize)

    torch.nn.init.xavier_uniform_(layer_to_randomize.weight)
    net.eval()
    return net


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def normalize(image):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if image.max() > 1:
        image /= 255
    image = (image - mean) / std
    return torch.tensor(image.swapaxes(-1, 1).swapaxes(2, 3)).float()


def aggregation_rule_grad(conc_import: List,
                          transform: str = 'abs',
                          summing: str = 'sum',
                          normalize_dims: bool = False,
                          threshold: bool = False):
    if transform == 'sqrt':
        new_SV_pret = [torch.sqrt(torch.abs(importance)).squeeze().sum(dim=0) for importance in
                       conc_import]
    elif transform == 'var':
        new_SV_pret = [importance.squeeze().var(axis=0) for importance in conc_import]
    elif transform == 'lin':
        new_SV_pret = [importance.squeeze().sum(dim=0) for importance in conc_import]
    elif transform == 'abs':
        new_SV_pret = [torch.abs(importance).squeeze().sum(dim=0) for importance in conc_import]
    else:
        raise NotImplementedError("Selected transform methods not implemented!")

    if normalize_dims:
        normalized = []
        for emb_dim in new_SV_pret:
            if emb_dim.max() > 0:
                emb_dim_arr = emb_dim
                normalized.append((emb_dim_arr - emb_dim_arr.min()) / (emb_dim_arr.max() - emb_dim_arr.min()))
            else:
                normalized.append(emb_dim)
        new_SV_pret = normalized
    if threshold:
        super_threshold_indxs = new_SV_pret >= 0.5
        under_threshold_indxs = new_SV_pret < 0.5
        new_SV_pret[super_threshold_indxs] = 1.0
        new_SV_pret[under_threshold_indxs] = 0.0

    if summing == 'sum':
        res = torch.stack(new_SV_pret, dim=0)
        res = res.sum(dim=0)
    elif summing == 'var':
        res = torch.stack(new_SV_pret, dim=0)
        res = res.var(dim=0)
    elif summing == 'abs':
        res = torch.stack(new_SV_pret, dim=0)
        res = torch.abs(res).sum(dim=0)
    else:
        raise NotImplementedError("Selected aggregation methods not implemented!")

    return res


def aggregate_channels(conc_import: List, transform: str = 'abs'):
    if transform == 'sqrt':
        new_SV_pret = [np.sqrt(np.abs(importance)).squeeze().sum(axis=0) for importance in conc_import]
    elif transform == 'lin':
        new_SV_pret = [importance.squeeze().sum(axis=0) for importance in conc_import]
    elif transform == 'abs':
        new_SV_pret = [np.abs(importance).squeeze().sum(axis=0) for importance in conc_import]
    elif transform == 'sqrt_max':
        new_SV_pret = [np.sqrt(np.abs(importance) * np.max(importance)).squeeze().sum(axis=0) for importance in
                       conc_import]
    elif transform == 'softmax':
        new_SV_pret = [np.abs(importance) * np.exp(np.max(np.abs(importance))).squeeze().sum(axis=0) for importance in
                       conc_import]
    elif transform == 'var':
        new_SV_pret = [importance.squeeze().var(axis=0) for importance in conc_import]
    else:
        raise NotImplementedError("Selected transform methods not implemented!")

    return new_SV_pret


def aggregation_rule(conc_import: List,
                     transform: str = 'abs',
                     summing: str = 'sum',
                     normalize_dims: bool = False,
                     threshold: bool = False):
    if len(conc_import[0].shape) > 2:
        new_SV_pret = aggregate_channels(conc_import=conc_import, transform=transform)
    else:
        new_SV_pret = conc_import

    if normalize_dims:
        normalized = []
        for emb_dim in new_SV_pret:
            if np.array(emb_dim).max() > 0:
                emb_dim_arr = np.array(emb_dim)
                normalized.append((emb_dim_arr - emb_dim_arr.min()) / (emb_dim_arr.max() - emb_dim_arr.min()))
            else:
                normalized.append(np.array(emb_dim))
        new_SV_pret = normalized
    if threshold:
        new_SV_pret = np.array(new_SV_pret)
        super_threshold_indxs = new_SV_pret >= 0.5
        under_threshold_indxs = new_SV_pret < 0.5
        new_SV_pret[super_threshold_indxs] = 1.0
        new_SV_pret[under_threshold_indxs] = 0.0

    if summing == 'sum':
        res = np.array(new_SV_pret).sum(axis=0)
    elif summing == 'abs':
        res = np.abs(np.array(new_SV_pret)).sum(axis=0)
    elif summing == 'var':
        res = np.array(new_SV_pret).var(axis=0)
    elif summing == 'var_over_mean':
        res = np.array(new_SV_pret).var(axis=0) / np.abs(np.array(new_SV_pret).mean(axis=0) + 0.00001)
    elif summing == 'exp':
        res = np.array([np.exp(nsp) for nsp in new_SV_pret]).sum(axis=0)
    elif summing == 'softmax':
        res = np.array([softmax(nsp) for nsp in new_SV_pret]).sum(axis=0)
    else:
        raise NotImplementedError("Selected aggregation methods not implemented!")

    return res


def noise_score(importance,
                importance_noise,
                transform: str = 'abs',
                summing: str = 'sum',
                grad: bool = False,
                normalize_dim: bool = False,
                threshold: bool = False):
    if not grad:
        aggr = aggregation_rule(importance, transform=transform, summing=summing, normalize_dims=normalize_dim,
                                threshold=threshold)
        aggr_noise = aggregation_rule(importance_noise, transform=transform, summing=summing,
                                      normalize_dims=normalize_dim, threshold=threshold)
    else:
        aggr = aggregation_rule_grad(importance, transform=transform, summing=summing)
        aggr_noise = aggregation_rule_grad(importance_noise, transform=transform, summing=summing)

    if not grad:
        if len(aggr.shape) > 1:
            return np.mean(np.abs((aggr - aggr.min()) / (aggr.max() - aggr.min())
                                  - (aggr_noise - aggr_noise.min()) / (aggr_noise.max() - aggr_noise.min())))
        else:
            return np.abs(aggr - aggr_noise)
    else:
        if len(aggr.shape) > 1:
            return torch.mean(torch.abs(aggr / aggr.max() - aggr_noise / aggr_noise.max()))
        else:
            return torch.abs(aggr - aggr_noise)


def noise_score_dims(importance, importance_noise, transform: str = 'abs'):
    aggr = aggregate_channels(importance, transform=transform)
    aggr_noise = aggregate_channels(importance_noise, transform=transform)
    noise_scores = []
    for channel_aggr, noise_aggr in zip(aggr, aggr_noise):
        noise_scores.append(np.mean(np.abs(channel_aggr / channel_aggr.max() - noise_aggr / noise_aggr.max())))

    return noise_scores


def var_score(importance,
              transform: str = 'abs',
              grad: bool = False,
              over_mean: bool = False,
              normalize_dim: bool = False,
              threshold: bool = False,
              return_mean: bool = True):
    if over_mean:
        summing = 'var_over_mean'
    else:
        summing = 'var'
    if not grad:
        aggr = aggregation_rule(importance, transform=transform, summing=summing, normalize_dims=normalize_dim,
                                threshold=threshold)
    else:
        aggr = aggregation_rule_grad(importance, transform=transform, summing=summing, normalize_dims=normalize_dim,
                                     threshold=threshold)
    if return_mean:
        return aggr.mean()
    else:
        return aggr.var()


def score_curve(
        model_type,
        dataset,
        rand_layer,
        pretrain,
        device: torch.device,
        sample_idx: int = 0,
        n_samples: int = 1,
        if_noise: bool = False,
        with_activations: bool = True,
        noise_noise: bool = False,
):
    net_1 = initialize_model(model_type=model_type, pretrained=pretrain, rand_layer=rand_layer)

    l1 = [module for module in net_1.modules() if
          type(module) in [torch.nn.modules.conv.Conv2d, torch.nn.modules.conv.Conv1d]]
    skip_if_too_many = 1
    if len(l1) > 55:
        skip_if_too_many = 5

    start = time.time()
    results = []
    for layer_idx in range(len(l1)):
        if layer_idx % skip_if_too_many != 0:
            continue

        net_1 = initialize_model(model_type=model_type, pretrained=pretrain, rand_layer=rand_layer)
        list_layer = [module for module in net_1.modules() if
                      type(module) in [torch.nn.modules.conv.Conv2d, torch.nn.modules.conv.Conv1d]]
        layer_to_compute = list_layer[layer_idx]
        size = dataset[0].shape[2]
        RANDOM = np.random.uniform(size=(1, size, size, 3))
        if not noise_noise:
            try:
                importance_layer = compute_importance_features(pre_model=net_1,
                                                               layer=layer_to_compute,
                                                               data=dataset,
                                                               samples_idx=[sample_idx],
                                                               n_samples=n_samples,
                                                               device=device,
                                                               with_activations=with_activations,
                                                               size=size)
            except torch.nn.modules.module.ModuleAttributeError:
                continue
        else:
            RANDOM2 = np.random.uniform(size=(1, size, size, 3))
            try:
                importance_layer = compute_importance_features(pre_model=net_1,
                                                               layer=layer_to_compute,
                                                               data=normalize(RANDOM2),
                                                               samples_idx=[0],
                                                               n_samples=n_samples,
                                                               device=device,
                                                               with_activations=with_activations,
                                                               size=size)
            except torch.nn.modules.module.ModuleAttributeError:
                continue

        if if_noise:
            importance_noise = compute_importance_features(pre_model=net_1,
                                                           layer=layer_to_compute,
                                                           data=normalize(RANDOM),
                                                           samples_idx=[0],
                                                           n_samples=n_samples,
                                                           device=device,
                                                           with_activations=with_activations,
                                                           size=size)
        else:
            importance_noise = None
        end = time.time()
        print(f'Finished layer {layer_idx}/{len(l1)}: {round((end - start) / 60, 2)} minutes')

        results.append({
            'importance_layer': importance_layer,
            'importance_noise': importance_noise,
            'layer': str(l1),
            'network': str(net_1)
        })

    end = time.time()

    print('Finished in', round((end - start) / 60, 2), 'minutes')

    return results


def score_curve_CAM(net_1,
                    dataset,
                    layer_idx: int,
                    sample_idx=0,
                    if_noise: bool = False,
                    noise_noise: bool = True):
    l1 = [module for module in net_1.modules() if
          type(module) in [torch.nn.modules.conv.Conv2d, torch.nn.modules.conv.Conv1d]]

    layer_to_compute = l1[layer_idx]
    start = time.time()

    cam = GradCAM(model=net_1, target_layer=layer_to_compute, use_cuda=True)
    size = dataset[0].shape[2]

    RANDOM = np.random.uniform(size=(1, size, size, 3))
    cam_res = []
    cam_res_noise = []

    if not noise_noise:
        for dim in range(512):
            cam_res.append(np.nan_to_num(cam(input_tensor=dataset[[sample_idx]], target_category=dim)))
    else:
        RANDOM2 = np.random.uniform(size=(1, size, size, 3))
        for dim in range(512):
            cam_res.append(np.nan_to_num(cam(input_tensor=normalize(RANDOM2), target_category=dim)))

    if if_noise:
        for dim in range(512):
            cam_res_noise.append(np.nan_to_num(cam(input_tensor=normalize(RANDOM), target_category=dim)))
    else:
        cam_res_noise = None
    end = time.time()
    print(f'Finished layer {layer_idx}/{len(l1)}: {round((end - start) / 60, 2)} minutes')

    return {
        'importance_layer': cam_res,
        'importance_noise': cam_res_noise,
        'layer': str(layer_to_compute),
        'network': str(net_1)
    }


def initialize_model(model_type: str, pretrained: bool, rand_layer: int):
    if model_type == 'res18':
        nets = models.resnet18(pretrained=pretrained)
        nets = nets.eval()
        nets.fc = Identity()
    elif model_type == 'inception':
        nets = models.inception_v3(pretrained=pretrained)
        nets.eval()
        nets.fc = Identity()
    elif model_type == 'alexnet':
        nets = models.alexnet(pretrained=pretrained)
        nets.eval()
        nets.classifier = Identity()
    else:
        raise NotImplementedError

    if rand_layer != -1:
        nets = randomize_layer(nets, layer_to_rand=rand_layer)

    return nets
