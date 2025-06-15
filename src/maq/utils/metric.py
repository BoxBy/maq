import torch

# z-score and LIM score metrics based on https://arxiv.org/abs/2406.17415v2

from transformers.utils import logging
logger = logging.get_logger(__name__)

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
    # Ensure calculations are done in float32 for precision
    inp_flat = layer_input.flatten().to(torch.float32)
    out_flat = layer_output.flatten().to(torch.float32)

    norm_inp = torch.norm(inp_flat, p=2)
    norm_out = torch.norm(out_flat, p=2)

    # Add epsilon for numerical stability, especially if norms are very small (but not zero)
    epsilon = 1e-8
    if norm_inp < epsilon or norm_out < epsilon:
        # If either norm is effectively zero, cosine similarity is undefined or unstable.
        # Depending on the desired behavior, could return 0.0 or handle as an error.
        # Returning 0.0 is a common practice.
        return 0.0

    dot_product = torch.dot(inp_flat, out_flat)
    cosine_similarity = dot_product / (norm_inp * norm_out)
    
    # Clamp the cosine similarity to the valid range [-1, 1] to handle potential floating point inaccuracies
    clamped_cosine_similarity = torch.clamp(cosine_similarity, -1.0, 1.0)
    
    return (-clamped_cosine_similarity).item()

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
    # Expected inps/outs shape: (num_samples, num_layers, seq_len, hidden_dim)
    scores = {}
    if inps.nelement() == 0 or outs.nelement() == 0:
        logger.warning("Input or output tensors for LIM score calculation are empty.")
        for i in range(len(modules)): # Use index if module objects are not directly usable as keys or if modules is just a count
            scores[modules[i] if isinstance(modules, list) else i] = 0.0 # Assign a default score
        return scores

    num_samples, num_layers_from_tensor, _, _ = inps.shape

    if num_layers_from_tensor != len(modules):
        logger.error(f"Mismatch in number of layers. Tensor has {num_layers_from_tensor} layers, but {len(modules)} modules were provided.")
        # Fallback: assign default score or raise error
        for i in range(len(modules)):
            scores[modules[i] if isinstance(modules, list) else i] = 0.0
        return scores

    for i in range(num_layers_from_tensor):
        module_key = modules[i] # Assumes modules is a list of module objects or suitable keys
        layer_i_inputs = inps[:, i, :, :]  # Shape: (num_samples, seq_len, hidden_dim)
        layer_i_outputs = outs[:, i, :, :] # Shape: (num_samples, seq_len, hidden_dim)
        scores[module_key] = cal_lim_score(layer_i_inputs, layer_i_outputs)
    return scores

# Mapping of metric names to their respective calculation functions.
METRIC_MAPPING = {
    "z-score": cal_z_score_all,
    "lim": cal_lim_score_all
}