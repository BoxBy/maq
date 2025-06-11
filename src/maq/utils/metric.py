import torch

# z-score and LIM score metrics based on https://arxiv.org/abs/2406.17415v2

def calc_z_score(weight: torch.Tensor) -> float:
    """
    Calculates the ratio of values in the weight tensor with an absolute z-score greater than 1.

    This function normalizes the input weight tensor using its mean and standard deviation, then 
    computes the z-score for each element. It returns the fraction of elements that are more than 
    one standard deviation away from the mean. If the standard deviation is zero, it returns 0.0.

    Parameters:
        weight (torch.Tensor): The weight tensor to compute z-scores for.

    Returns:
        float: The fraction of tensor elements with |z-score| > 1.
    """
    weight = weight.float()  # Convert to float if the tensor is of integer type
    mu = torch.mean(weight)
    sigma = torch.std(weight)
    if sigma == 0:
        return 0.0
    z = (weight - mu) / sigma
    count_exceed = (z.abs() > 1).float().sum().item()
    total = weight.numel()
    return count_exceed / total

def cal_z_score_all(modules, inps, outs) -> dict:
    """
    Calculates the average z-score ratio for each provided module.

    For each module in the list, this function locates its sub-modules that have either a 'weight' or  
    'qweight' attribute. For each of these sub-modules, it computes the z-score ratio using calc_z_score.
    If a module does not contain any linear sub-modules (with applicable weight attributes), 
    it assigns infinity (float('inf')) as the score.

    Parameters:
        modules (iterable): A collection of module objects to evaluate.
        inps: Placeholder for inputs (unused in this function).
        outs: Placeholder for outputs (unused in this function).

    Returns:
        dict: A dictionary mapping each module to its average z-score ratio.
    """
    scores = {}
    for module in modules:
        linears = [m for m in module.modules() if hasattr(m, "qweight") or hasattr(m, "weight")]
        if len(linears) == 0:
            scores[module] = float('inf')
        else:
            score_list = [
                calc_z_score(linear.weight.data) if hasattr(linear, "weight")
                else calc_z_score(linear.qweight.data)
                for linear in linears
            ]
            scores[module] = sum(score_list) / len(score_list)
    return scores

def cal_lim_score(layer_input: torch.Tensor, layer_output: torch.Tensor) -> float:
    """
    Computes the LIM Score between an input tensor and an output tensor using cosine similarity.

    This function first flattens the input and output tensors into one-dimensional vectors. 
    It then computes their 2-norms and the dot product. The LIM score is defined as the negative 
    cosine similarity of the two vectors. If either vector has a norm of zero, the function returns 0.0.

    Parameters:
        layer_input (torch.Tensor): The input tensor to the layer.
        layer_output (torch.Tensor): The output tensor from the layer.

    Returns:
        float: The calculated LIM score (negative cosine similarity).
    """
    inp = layer_input.flatten()
    out = layer_output.flatten()
    norm_inp = torch.norm(inp, p=2)
    norm_out = torch.norm(out, p=2)
    if norm_inp.item() == 0 or norm_out.item() == 0:
        return 0.0
    dot = torch.dot(inp, out)
    return (-dot / (norm_inp * norm_out)).item()

def cal_lim_score_all(modules, inps, outs) -> dict:
    """
    Computes the LIM Score for each module based on its captured layer inputs and outputs.

    For each module, it retrieves the corresponding input and output vectors from the provided 
    inps and outs tensors using an index. Then, it calculates the LIM score for these vectors. 
    The resulting dictionary maps each module to its corresponding LIM score.

    Parameters:
        modules (iterable): A collection of module objects.
        inps (torch.Tensor): A tensor capturing the input vectors for each module (assumed multi-dimensional).
        outs (torch.Tensor): A tensor capturing the output vectors for each module (assumed multi-dimensional).

    Returns:
        dict: A dictionary mapping each module to its LIM score.
    """
    scores = {}
    for idx, module in enumerate(modules):
        layer_inputs = inps[:, idx]
        layer_outputs = outs[:, idx]
        scores[module] = cal_lim_score(layer_inputs, layer_outputs)
    return scores

# Mapping of metric names to their respective calculation functions.
METRIC_MAPPING = {
    "z-score": cal_z_score_all,
    "lim": cal_lim_score_all
}