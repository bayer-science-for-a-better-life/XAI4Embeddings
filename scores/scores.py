
import argparse
import os
import pickle
import time
from typing import List
from scipy.special import softmax
import shap
import torch
import torchvision.models as models
import torch.nn as nn
import numpy as np
from scores.attributions import compute_importance_features
from pytorch_grad_cam import GradCAM


def get_parser():
    parser = argparse.ArgumentParser(description='Script for XAI in PathDrive')

    parser.add_argument('--device', action='store', dest='device', default=6, type=int,
                        help="Which gpu device to use. Defaults to 6")

    parser.add_argument('--sample', action='store', dest='sample_idx', default=0, type=int,
                        help='Sample to compute. Defaults to 0.')

    parser.add_argument('--n_samples', action='store', dest='n_samples', default=1, type=int,
                        help='Sample to compute. Defaults to 1.')

    parser.add_argument('--save_dir', action='store', dest='save_dir', type=str,
                        help='Name of directory to save the results.')

    parser.add_argument('--model', action='store', dest='model', default='res18', type=str,
                        help='Which model to examine. Defaults to res18')

    parser.add_argument('--pretrain', action='store', dest='pretrain', default='True', type=str,
                        help='Whether pretrained weights. Defaults to True.')

    parser.add_argument('--activations', action='store', dest='activations', default='True', type=str,
                        help='Whether to multiply by activations. Defaults to True.')

    parser.add_argument('--rand_layer', action='store', dest='rand_layer', default=-1, type=int,
                        help='Which layer to randomize. Defaults to -1 (no randomization).')

    parser.add_argument('--redund', action='store', dest='redunt', default='False', type=str,
                        help='Whether to make the embedding redundant.')

    parser.add_argument('--imagenet', action='store', dest='imagenet', default='True', type=str,
                        help='Whether to use imagenet samples.')

    parser.add_argument('--use_cam', action='store', dest='use_cam', default='False', type=str,
                        help='Whether to compute GradCAM.')

    parser.add_argument('--noisenoise', action='store', dest='noisenoise', default='False', type=str,
                        help='Whether to compute Noise-noise score.')

    return parser.parse_args()


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


# TODO: remobe this not used
# class Resnet18_redunt(torch.nn.Module):
#     def __init__(self, device, pretrained=True):
#         super(Resnet18_redunt, self).__init__()
#
#         self.temp = models.resnet18(pretrained=pretrained)
#         self.temp.fc = Identity()
#         self.device = device
#         # self.mask = torch.rand(512) < 0.5
#         # self.mask = torch.ones(512) * self.mask
#
#     def forward(self, x):
#         x = self.temp(x)
#         x = x.view(x.size(0), -1)
#
#         mask = torch.rand(512) < 0.5
#         mask = torch.ones(512) * mask
#         mask = mask.to(self.device)
#         x = x * mask + torch.mean(x) * (1 - mask)
#
#         return x


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
                          transform: str = 'sqrt',
                          summing: str = 'sum',
                          normalize_dims: bool = False,
                          threshold: bool = False):
    if transform == 'sqrt':
        new_SV_pret = [torch.sqrt(torch.abs(importance)).squeeze().sum(dim=0) for importance in
                       conc_import]  # summing over channels
    elif transform == 'lin':
        new_SV_pret = [importance.squeeze().sum(dim=0) for importance in conc_import]
    elif transform == 'abs':
        new_SV_pret = [torch.abs(importance).squeeze().sum(dim=0) for importance in conc_import]
    # elif transform == 'sqrt_max':
    #     new_SV_pret = [np.sqrt(np.abs(importance) * np.max(importance)).squeeze().sum(axis=0) for importance in conc_import]
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
    else:
        raise NotImplementedError("Selected aggregation methods not implemented!")

    return res


def aggregate_channels(conc_import: List, transform: str = 'sqrt'):
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
                     transform: str = 'sqrt',
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


# TODO: include this?
def chain_rule(conc_import: List,
               dn_import: List,
               class_idx: int,
               transform: str = 'sqrt',
               summing: str = 'sum',
               abs_value: bool = False):
    new_SV_pret = aggregate_channels(conc_import=conc_import, transform=transform)
    SV_class = dn_import[class_idx].squeeze()

    if summing == 'sum':
        if abs_value:
            res = sum(abs(SVc) * nsp for SVc, nsp in zip(SV_class, new_SV_pret))
        else:
            res = sum(SVc * nsp for SVc, nsp in zip(SV_class, new_SV_pret))
    elif summing == 'var':
        if abs_value:
            res = np.array([abs(SVc) * nsp for SVc, nsp in zip(SV_class, new_SV_pret)]).var(axis=0)
        else:
            res = np.array([SVc * nsp for SVc, nsp in zip(SV_class, new_SV_pret)]).var(axis=0)
    elif summing == 'exp':
        if abs_value:
            res = sum(np.exp(abs(SVc)) * nsp for SVc, nsp in zip(SV_class, new_SV_pret))
        else:
            res = sum(np.exp(SVc) * nsp for SVc, nsp in zip(SV_class, new_SV_pret))
    elif summing == 'softmax':
        if abs_value:
            res = sum(softmax(abs(SVc)) * nsp for SVc, nsp in zip(SV_class, new_SV_pret))
        else:
            res = sum(softmax(SVc) * nsp for SVc, nsp in zip(SV_class, new_SV_pret))
    elif summing == 'relu':
        assert abs_value is False
        res = sum(SVc * nsp for SVc, nsp in zip(SV_class, new_SV_pret) if SVc > 0)
    else:
        raise NotImplementedError("Selected aggregation methods not implemented!")

    return res


def noise_score(importance,
                importance_noise,
                transform: str = 'sqrt',
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


def noise_score_dims(importance, importance_noise, transform: str = 'sqrt'):
    aggr = aggregate_channels(importance, transform=transform)
    aggr_noise = aggregate_channels(importance_noise, transform=transform)
    noise_scores = []
    for channel_aggr, noise_aggr in zip(aggr, aggr_noise):
        noise_scores.append(np.mean(np.abs(channel_aggr / channel_aggr.max() - noise_aggr / noise_aggr.max())))

    return noise_scores


def var_score(importance,
              transform: str = 'sqrt',
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
        redunt,
        sample_idx=0,
        n_samples: int = 1,
        if_noise: bool = False,
        with_activations: bool = True,
        noise_noise: bool = False
):
    net_1 = initialize_model(model_type=model_type, pretrained=pretrain, rand_layer=rand_layer, redund=redunt)

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

        net_1 = initialize_model(model_type=model_type, pretrained=pretrain, rand_layer=rand_layer, redund=redunt)
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
        nets = randomize_layer(net, layer_to_rand=args.rand_layer)

    return nets


if __name__ == '__main__':

    args = get_parser()
    redunt = str2bool(args.redunt)
    imagenet = str2bool(args.imagenet)
    use_cam = str2bool(args.use_cam)
    activations = str2bool(args.activations)
    noisenoise = str2bool(args.noisenoise)

    os.makedirs(f'/gpfs01/home/glsvu/pathological-suite/paxo/XAI_valid/results/{args.save_dir}', exist_ok=True)

    pretrain = str2bool(args.pretrain)

    print('Loading the dataset...')
    X, y = shap.datasets.imagenet50()
    print('Done loading the dataset!')
    X /= 255
    X = normalize(X)

    device = torch.device(f'cuda:{args.device}') if torch.cuda.is_available() else torch.device('cpu')

    if args.model == 'res18':
        net = models.resnet18(pretrained=pretrain)
        net = net.eval()
        net.fc = Identity()
    elif args.model == 'inception':
        net = models.inception_v3(pretrained=pretrain)
        net.eval()
        net.fc = Identity()
    elif args.model == 'alexnet':
        net = models.alexnet(pretrained=pretrain)
        net.eval()
        net.classifier = Identity()

    # FIXME: reorganize the randomization of the layer
    add_to_name = ''
    if args.rand_layer != -1:
        net = randomize_layer(net, layer_to_rand=args.rand_layer)
        add_to_name = f'rand{args.rand_layer}_'

    if not use_cam:
        # net = initialize_model(args.model, pretrained=pretrain, redund=redunt)
        res_tr = score_curve(args.model,
                             pretrain=pretrain,
                             rand_layer=args.rand_layer,
                             redunt=redunt,
                             dataset=X,
                             sample_idx=args.sample_idx,
                             if_noise=True,
                             with_activations=activations,
                             noise_noise=noisenoise)

        file_to_save = f'{args.save_dir}_{add_to_name}{args.sample_idx}.pckl'
        with open(f'/gpfs01/home/glsvu/pathological-suite/paxo/XAI_valid/results/{args.save_dir}/{file_to_save}',
                  'wb') as f:
            pickle.dump(res_tr, f)

        print('Saved to file!')
    else:
        res_tr = []
        for layer_idx in range(20):
            # We re-inizialize the model otherwise the memory explodes.
            net = initialize_model(args.model, pretrained=pretrain, rand_layer=args.rand_layer, redund=redunt)
            res_tr.append(score_curve_CAM(net,
                                          dataset=X,
                                          layer_idx=layer_idx,
                                          sample_idx=args.sample_idx,
                                          if_noise=True,
                                          noise_noise=noisenoise))

        file_to_save = f'{args.save_dir}_{add_to_name}{args.sample_idx}_CAM.pckl'
        with open(f'/gpfs01/home/glsvu/pathological-suite/paxo/XAI_valid/results/{args.save_dir}/{file_to_save}',
                  'wb') as f:
            pickle.dump(res_tr, f)

        print('Saved to file!')
