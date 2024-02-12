from collections import defaultdict

import torch


def multi_crop_forward(model, x, is_batch_size_dependent=None, *args, **kwargs):
    input_was_list = True
    if not isinstance(x, list):
        x = [x]
        input_was_list = False

    if is_batch_size_dependent is None:
        is_batch_size_dependent = model.is_batch_size_dependent
    if is_batch_size_dependent:
        # seperate forward pass
        results = [model(data, *args, **kwargs) for data in x]
    else:
        # group by (channels, height, width) or whatever x.shape[1:] is
        grouped_x = defaultdict(list)
        for global_idx, view in enumerate(x):
            if isinstance(view, dict):
                batch_size = view["x"].shape[0]
                key = view["x"].shape[1:]
            else:
                batch_size = view.shape[0]
                key = view.shape[1:]
            local_idx = len(grouped_x[key])
            grouped_x[key].append((global_idx, local_idx, view, batch_size))
        # concat
        for key, value in grouped_x.items():
            global_idxs, local_idxs, inputs, batch_sizes = zip(*value)
            if isinstance(inputs[0], dict):
                to_concat = defaultdict(list)
                for input_dict in inputs:
                    for k, v in input_dict.items():
                        to_concat[k].append(v)
                inputs = _concat_tensors(to_concat)
            else:
                inputs = torch.concat(inputs)
            grouped_x[key] = (global_idxs, local_idxs, inputs, batch_sizes)
        # forward pass per resolution
        results = [None for _ in range(len(x))]
        for global_idxs, local_idxs, data, batch_sizes in grouped_x.values():
            output = model(data, *args, **kwargs)
            batch_idx = 0
            for global_idx, local_idx, batch_size in zip(global_idxs, local_idxs, batch_sizes):
                if isinstance(output, dict):
                    result = {k: v[batch_idx:batch_idx + batch_size] for k, v in output.items()}
                else:
                    assert torch.is_tensor(output)
                    result = output[batch_idx:batch_idx + batch_size]
                results[global_idx] = result
                batch_idx += batch_size

    # if input was a tensor -> return result as tensor
    if not input_was_list:
        assert len(results) == 1
        results = results[0]
    return results


def _concat_tensors(to_concat):
    # TODO momentum stuff is not concated (but currently not needed as all contrastive heads have batchnorm
    if isinstance(to_concat, list):
        if isinstance(to_concat[0], dict):
            return {
                key: torch.concat([to_concat[i][key] for i in range(len(to_concat))])
                for key in to_concat[0].keys()
            }
        return torch.concat(to_concat)
    if isinstance(to_concat, dict):
        for key in to_concat.keys():
            to_concat[key] = _concat_tensors(to_concat[key])
    return to_concat


# adapted from https://github.com/facebookresearch/dino/blob/main/main_dino.py#L394
def multi_crop_loss(input0, input1, loss_fn):
    assert len(input0) > 1 and len(input1) > 1

    losses = {}
    for i, in0 in enumerate(input0):
        for j, in1 in enumerate(input1):
            if i == j:
                # skip if same view
                continue
            losses[(i, j)] = loss_fn(in0, in1)
    return losses
