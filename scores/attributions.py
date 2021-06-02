import numpy as np
import torch
import time
from gradients import concept_importance
from typing import List, Optional


def compute_importance_class(model: torch.nn.Module,
                             layer: torch.nn.Module,
                             data: torch.Tensor,
                             samples_idx: Optional[List[int]],
                             device: torch.device,
                             output: int = 0,
                             n_samples: int = 50,
                             smoothing: float = 0.0,
                             black_ref: bool = True,
                             with_grads: bool = False):
    if black_ref:
        ref_image = torch.zeros(1, data.shape[1])
    else:
        ref_image = torch.rand(n_samples, data.shape[1])

    if samples_idx is not None:
        tiles = data[samples_idx]
    else:
        tiles = data

    model = model.to(device)
    importance_features_class = concept_importance(model=model,
                                                   layer=layer,
                                                   model_input=tiles.to(device),
                                                   background=ref_image.to(device),
                                                   nsamples=n_samples,
                                                   output_idx=output,
                                                   local_smoothing=smoothing,
                                                   with_grads=with_grads)
    return importance_features_class


def compute_importance_features(pre_model: torch.nn.Module,
                                layer: torch.nn.Module,
                                data: torch.Tensor,
                                samples_idx: Optional[List[int]],
                                device: torch.device,
                                size: int = 224,
                                n_samples: int = 10,
                                smoothing: float = 0.5,
                                black_ref: bool = True,
                                verbose: bool = False,
                                with_activations: bool = True,
                                with_grads: bool = False) -> List[np.array]:
    pre_model = pre_model.to(device)

    if black_ref:
        if len(data.shape) != 2:
            if data.shape[1] != 1:
                ref_image = torch.zeros(1, 3, size, size)
            else:
                ref_image = torch.zeros(1, 1, size, size)
        else:
            ref_image = torch.zeros(1, data.shape[1])
    else:
        ref_image = torch.rand(10, 3, size, size)

    start = time.time()

    if samples_idx is not None:
        tiles = data[samples_idx]
    else:
        tiles = data

    importance_features = concept_importance(model=pre_model,
                                             layer=layer,
                                             model_input=tiles.to(device),
                                             background=ref_image.to(device),
                                             nsamples=n_samples,
                                             local_smoothing=smoothing,
                                             multiply_grad_by_activation=with_activations,
                                             with_grads=with_grads,
                                             device=device,
                                             verbose=verbose)

    end = time.time()
    if verbose:
        print('Computed importance features for the pretrained model with shape', len(importance_features),
              importance_features[0].shape, 'in time = ', round((end - start) / 60, 4))

    return importance_features


def chain_rule_downstream(shap_values, shap_values_pret, abs_value=True, summing='sum'):
    result = []
    for output in range(len(shap_values)):
        result_output = []
        for tile in range(shap_values[output].shape[0]):
            SV_class = shap_values[output][tile, :]

            SV_pret = [s[tile] for s in shap_values_pret]  # fixing tile
            new_SV_pret = [np.sqrt(np.abs(shap)).sum(axis=0) for shap in SV_pret]

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
            else:
                raise NotImplementedError("Selected aggregation methods not implemented!")

            result_output.append(res)
        result.append(np.array(result_output))

    return result
