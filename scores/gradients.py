import torch
import numpy as np
import time
from typing import List


def get_interim_input(layer, layer_input, output):
    try:
        del layer.target_input
    except AttributeError:
        pass
    setattr(layer, 'target_input', layer_input)


def add_handles(layer):
    input_handle = layer.register_forward_hook(get_interim_input)
    return input_handle


def extract_input_layer(model: torch.nn.Module, layer: torch.nn.Module, data: torch.Tensor) -> List[torch.Tensor]:

    model = model.eval()
    add_handles(layer)

    with torch.no_grad():
        _ = model(data)
        interim_inputs = layer.target_input
        if type(interim_inputs) is tuple:
            inter_inp = [i.clone().detach() for i in interim_inputs]
        else:
            inter_inp = [interim_inputs.clone().detach()]

    return inter_inp


def get_gradient(model: torch.nn.Module,
                 layer: torch.nn.Module,
                 data: torch.Tensor,
                 idx: int = None,
                 output_idx: int = 0,
                 with_grads: bool = True) -> List[torch.Tensor]:

    model_inputs = data.requires_grad_()
    model = model.eval()

    add_handles(layer)
    model.zero_grad()

    outputss = model(model_inputs)
    if isinstance(outputss, tuple):
        outputs = outputss[output_idx]
    else:
        outputs = outputss

    interim_inputs = layer.target_input

    if idx is not None:
        if not with_grads:
            selected = [val for val in outputs[:, idx]]
            grads = [torch.autograd.grad(selected, input_,
                                         retain_graph=True if idx + 1 < len(interim_inputs) else None)[0].cpu().numpy()
                     for idx, input_ in enumerate(interim_inputs)]
        else:
            selected = [val for val in outputs[:, idx]]
            grads = [torch.autograd.grad(selected, input_, create_graph=True,
                                         retain_graph=True if idx + 1 < len(interim_inputs) else None)[0]
                     for idx, input_ in enumerate(interim_inputs)]
        del layer.target_input
    else:
        grads = []
        n_output = outputs.shape[-1]
        for i in range(n_output):
            selected = [val for val in outputs[:, i]]
            if not with_grads:
                grads_i = [torch.autograd.grad(selected, input_, retain_graph=True)[0].cpu().numpy()
                           for idx, input_ in enumerate(interim_inputs)]
            else:
                grads_i = [torch.autograd.grad(selected, input_, retain_graph=True)[0]
                           for idx, input_ in enumerate(interim_inputs)]
            grads.append(grads_i)
        grads = [torch.Tensor(gr) for gr in grads]
        del layer.target_input

    return grads


def concept_importance(model: torch.nn.Module,
                       layer: torch.nn.Module,
                       model_input: torch.Tensor,
                       background: torch.Tensor,
                       multiply_grad_by_activation: bool = True,
                       nsamples: int = 10,
                       output_idx: int = 0,
                       rseed: bool = None,
                       local_smoothing: float = 0.2,
                       verbose: bool = False,
                       with_grads: bool = False,
                       device: torch.device = None):

    outputss = model(model_input)
    if isinstance(outputss, tuple):
        if verbose:
            print('The model has several outputs, computing wrt to output #', output_idx)
        outputs = outputss[output_idx]
    else:
        outputs = outputss

    add_handles(layer)

    data = extract_input_layer(model=model, layer=layer, data=model_input)
    data_background = extract_input_layer(model=model, layer=layer, data=background)
    input_handle = False
    interim = True

    X_batches = model_input.shape[0]
    output_phis = []

    samples_input = [torch.zeros((nsamples,) + model_input.shape[1:], device=model_input.device)]
    if verbose:
        print('shape of samples_input', samples_input[0].shape)  # torch.Size([200, 3, 224, 224])
    if not with_grads:
        samples_delta = [np.zeros((nsamples,) + data[l].shape[1:]).astype('float32') for l in range(len(data))]
    else:
        samples_delta = [torch.zeros((nsamples,) + data[l].shape[1:]) for l in range(len(data))]

    if rseed is None:
        rseed = np.random.randint(0, 1e6)

    start_compute = time.time()


    for i in range(outputs.shape[1]):
        np.random.seed(rseed)
        phis = []
        if not with_grads:
            phis.append(np.zeros((X_batches,) + data[0].shape[1:]).astype('float32'))
        else:
            phis.append(torch.zeros((X_batches,) + data[0].shape[1:]))

        for j in range(model_input.shape[0]):
            for k in range(nsamples):
                rind = np.random.choice(data_background[0].shape[0])
                t = np.random.uniform()

                if local_smoothing > 0:
                    if not with_grads:
                        x = model_input[j].clone().detach() + torch.empty(model_input[j].shape,
                                                                          device=model_input.device).normal_() \
                            * local_smoothing
                    else:
                        x = model_input[j].clone() + torch.empty(model_input[j].shape,
                                                                 device=model_input.device).normal_() \
                            * local_smoothing
                else:
                    if not with_grads:
                        x = model_input[j].clone().detach()
                    else:
                        x = model_input[j].clone()
                if not with_grads:
                    samples_input[0][k] = (t * x + (1 - t) * (background[rind]).clone().detach()). \
                        clone().detach()
                else:
                    samples_input[0][k] = (t * x + (1 - t) * (background[rind]).clone()).clone()
                if input_handle is None:
                    if not with_grads:
                        samples_delta[0][k] = (x - (data_background[0][rind]).clone().detach()).cpu().numpy()
                    else:
                        samples_delta[0][k] = (x - (data_background[0][rind]).clone())

                if interim is True:
                    with torch.no_grad():
                        _ = model(*[samples_input[0][k].unsqueeze(0)])
                        interim_inputs = layer.target_input
                        del layer.target_input
                        if type(interim_inputs) is tuple:
                            if not with_grads:
                                for l in range(len(interim_inputs)):
                                    samples_delta[l][k] = interim_inputs[l].cpu().numpy()
                            else:
                                for l in range(len(interim_inputs)):
                                    samples_delta[l][k] = interim_inputs[l]
                        else:
                            if not with_grads:
                                samples_delta[0][k] = interim_inputs.cpu().numpy()
                            else:
                                samples_delta[0][k] = interim_inputs.cpu()

            batch_size = 50
            grads = []
            for b in range(0, nsamples, batch_size):
                if not with_grads:
                    batch = [samples_input[l][b:min(b + batch_size, nsamples)].clone().detach() for l in range(len(data))]
                    grads.append(get_gradient(
                        model=model,
                        layer=layer,
                        data=batch[0],
                        idx=i,
                        output_idx=output_idx,
                        with_grads=False))
                else:
                    batch = [samples_input[l][b:min(b + batch_size, nsamples)].clone() for l in
                             range(len(data))]
                    grads.append(get_gradient(
                        model=model,
                        layer=layer,
                        data=batch[0],
                        idx=i,
                        output_idx=output_idx,
                        with_grads=True))
            if not with_grads:
                grad = [np.concatenate([g[l] for g in grads], 0) for l in range(len(data))]
            else:
                grad = [torch.stack([g[l] for g in grads], 0) for l in range(len(data))]
            del grads

            for l in range(len(data)):
                if multiply_grad_by_activation:
                    try:
                        samples = grad[l] * samples_delta[l]
                    except RuntimeError:
                        samples = grad[l] * samples_delta[l].to(device)
                else:
                    samples = grad[l]
                if not with_grads:
                    phis[l][j] = samples.mean(0)
                else:
                    phis[l][j] = torch.mean(samples, dim=0)
        output_phis.append(phis[0] if len(data) == 1 else phis)

        if verbose:
            if i % 100 == 0:
                end = time.time()
                print(f'Done with # {i} outputs in {round((end - start_compute) / 60, 3)}. Sample = {nsamples}.')

        return output_phis
