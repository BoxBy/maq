import torch
import shap
import numpy as np
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
    Computes the average LIM Score between input and output tensors across a batch of samples.

    This function assumes the first dimension of the input tensors is the batch dimension.
    It calculates the cosine similarity for each sample in the batch and returns the 
    average of these scores. The LIM score is defined as the negative cosine similarity.

    Parameters:
        layer_input (torch.Tensor): The input tensor to the layer, with shape 
                                    (num_samples, ...).
        layer_output (torch.Tensor): The output tensor from the layer, with shape 
                                     (num_samples, ...).

    Returns:
        float: The calculated average LIM score (negative cosine similarity).
    """
    # If input is a single un-batched sample, add a batch dimension for consistent processing.
    if layer_input.dim() == 1:
        layer_input = layer_input.unsqueeze(0)
        layer_output = layer_output.unsqueeze(0)

    num_samples = layer_input.shape[0]
    if num_samples == 0:
        return 0.0

    # Ensure calculations are done in float32 for precision
    inp_flat = layer_input.view(num_samples, -1).to(torch.float32)
    out_flat = layer_output.view(num_samples, -1).to(torch.float32)

    # Calculate norms for each sample in the batch
    norm_inp = torch.norm(inp_flat, p=2, dim=1)
    norm_out = torch.norm(out_flat, p=2, dim=1)

    # Calculate dot product for each sample
    dot_product = torch.sum(inp_flat * out_flat, dim=1)

    # Add epsilon for numerical stability
    epsilon = 1e-8
    denominator = norm_inp * norm_out
    valid_mask = denominator > epsilon
    
    if not torch.any(valid_mask):
        return 0.0

    # Initialize cosine similarity tensor. Invalid samples will have a score of 0.
    cosine_similarity = torch.zeros_like(dot_product)
    cosine_similarity[valid_mask] = dot_product[valid_mask] / denominator[valid_mask]
    
    # Clamp the cosine similarity to the valid range [-1, 1] to handle potential floating point inaccuracies
    clamped_cosine_similarity = torch.clamp(cosine_similarity, -1.0, 1.0)
    
    # The LIM score is the negative cosine similarity. We average over all samples.
    # Samples with zero-norm vectors contribute 0 to the sum, which is a reasonable default.
    avg_lim_score = -torch.mean(clamped_cosine_similarity)
    
    return avg_lim_score.item()

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

def cal_activation_norm_all(modules, inps, outs) -> dict:
    """
    Computes the difference in average L2 norm between output and input activations for each module.

    This metric addresses the issue of activation norms naturally increasing in deeper layers.
    It calculates the "delta" or change in norm caused by the layer (`norm(output) - norm(input)`).
    This provides a better measure
    of a layer's individual contribution to signal magnitude. A higher score indicates a more
    important layer. With default sorting, layers with lower scores (less important) will be
    quantized first.

    Parameters:
        modules (iterable): A collection of module objects to evaluate.
        inps (torch.Tensor): A tensor of captured inputs.
                             Shape: (num_samples, num_layers, seq_len, hidden_dim).
        outs (torch.Tensor): A tensor capturing the output vectors for each module.
                             Shape: (num_samples, num_layers, seq_len, hidden_dim).

    Returns:
        dict: A dictionary mapping each module to its average activation norm score.
    """
    scores = {}
    if outs.nelement() == 0 or inps.nelement() == 0:
        logger.warning("Input or output tensor for activation norm calculation is empty.")
        for i, module in enumerate(modules):
            scores[module] = 0.0
        return scores

    num_samples, num_layers, _, _ = outs.shape

    if num_layers != len(modules) or inps.shape[1] != len(modules):
        logger.error(f"Mismatch in number of layers. Tensors have {num_layers} layers, but {len(modules)} modules were provided.")
        for module in modules:
            scores[module] = 0.0
        return scores

    for i, module in enumerate(modules):
        # Calculate norm for output
        layer_outputs = outs[:, i, :, :].to(torch.float32)  # Shape: (num_samples, seq_len, hidden_dim)
        layer_outputs_flat = layer_outputs.view(num_samples, -1)
        output_norms = torch.norm(layer_outputs_flat, p=2, dim=1)
        avg_output_norm = torch.mean(output_norms)

        # Calculate norm for input
        layer_inputs = inps[:, i, :, :].to(torch.float32)
        layer_inputs_flat = layer_inputs.view(num_samples, -1)
        input_norms = torch.norm(layer_inputs_flat, p=2, dim=1)
        avg_input_norm = torch.mean(input_norms)
        
        # Score is the difference
        scores[module] = (avg_output_norm - avg_input_norm).item()
        
    return scores

def cal_activation_variance_all(modules, inps, outs) -> dict:
    """
    Computes the difference in mean variance between output and input activations for each module.

    This metric assumes that layers that significantly increase the variance of activations
    are performing more complex transformations and are thus more important. It calculates
    the "delta" or change in mean variance caused by the layer (`mean_var(output) - mean_var(input)`).
    A higher score indicates a more important layer. With default sorting, layers with lower
    scores (less sensitive) will be quantized first.

    Note: Variance calculation requires at least two samples. If `num_samples < 2`, the score
    for that layer will be 0.0.
    Parameters:
        modules (iterable): A collection of module objects to evaluate.
        inps (torch.Tensor): A tensor of captured inputs (unused).
        outs (torch.Tensor): A tensor capturing the output vectors for each module.
                             Shape: (num_samples, num_layers, seq_len, hidden_dim).

    Returns:
        dict: A dictionary mapping each module to its mean activation variance score.
    """
    scores = {}
    if outs.nelement() == 0 or inps.nelement() == 0:
        logger.warning("Input or output tensor for activation variance calculation is empty.")
        for i, module in enumerate(modules):
            scores[module] = 0.0
        return scores

    num_samples, num_layers, _, _ = outs.shape

    if num_layers != len(modules):
        logger.error(f"Mismatch in number of layers. Tensor has {num_layers} layers, but {len(modules)} modules were provided.")
        for module in modules:
            scores[module] = 0.0
        return scores

    for i, module in enumerate(modules):
        layer_inputs = inps[:, i, :, :].to(torch.float32)
        layer_outputs = outs[:, i, :, :].to(torch.float32)  # Shape: (num_samples, seq_len, hidden_dim)
        
        if num_samples < 2:
            logger.warning(f"Cannot calculate variance for layer {i} with less than 2 samples. Returning 0.")
            scores[module] = 0.0
            continue
            
        # Calculate variance for output
        output_variance_tensor = torch.var(layer_outputs, dim=0, unbiased=True) # Shape: (seq_len, hidden_dim)
        mean_output_variance = torch.mean(output_variance_tensor)

        # Calculate variance for input
        input_variance_tensor = torch.var(layer_inputs, dim=0, unbiased=True)
        mean_input_variance = torch.mean(input_variance_tensor)
        
        scores[module] = (mean_output_variance - mean_input_variance).item()
        
    return scores

def cal_awq_proxy_score_all(modules, inps, outs) -> dict:
    """
    Computes a sensitivity score based on the change in average magnitude of activations (AWQ proxy).

    This metric, inspired by AWQ, assumes that layers that significantly change the magnitude
    of activations are more sensitive to quantization and thus more important. It calculates
    the "delta" or change in mean absolute activation (`mean_abs(output) - mean_abs(input)`).
    A higher score indicates a more important layer. With default sorting, layers with lower
    scores (less sensitive) will be quantized first.

    Parameters:
        modules (iterable): A collection of module objects to evaluate.
        inps (torch.Tensor): A tensor of captured inputs. Shape: (num_samples, num_layers, seq_len, hidden_dim).
        outs (torch.Tensor): A tensor of captured outputs (unused).

    Returns:
        dict: A dictionary mapping each module to its AWQ proxy score.
    """
    scores = {}
    if inps.nelement() == 0 or outs.nelement() == 0:
        logger.warning("Input or output tensor for AWQ proxy score calculation is empty.")
        for i, module in enumerate(modules):
            scores[module] = 0.0
        return scores

    num_samples, num_layers, _, _ = inps.shape

    if num_layers != len(modules):
        logger.error(f"Mismatch in number of layers. Tensor has {num_layers} layers, but {len(modules)} modules were provided.")
        for module in modules:
            scores[module] = 0.0
        return scores

    for i, module in enumerate(modules):
        layer_inputs = inps[:, i, :, :].to(torch.float32)
        layer_outputs = outs[:, i, :, :].to(torch.float32)

        mean_abs_input = torch.mean(torch.abs(layer_inputs))
        mean_abs_output = torch.mean(torch.abs(layer_outputs))
        scores[module] = (mean_abs_output - mean_abs_input).item()
        
    return scores

def cal_perturb_dist_all(modules, inps, outs) -> dict:
    """
    Computes the average L2 distance between a layer's input and output.

    This metric assumes that layers that significantly transform their input vectors
    (i.e., the L2 distance between input and output is large) are more important.
    This captures the functional transformation magnitude of each layer. A higher
    score indicates a more important layer.

    Parameters:
        modules (iterable): A collection of module objects to evaluate.
        inps (torch.Tensor): A tensor of captured inputs.
                             Shape: (num_samples, num_layers, seq_len, hidden_dim).
        outs (torch.Tensor): A tensor capturing the output vectors for each module.
                             Shape: (num_samples, num_layers, seq_len, hidden_dim).

    Returns:
        dict: A dictionary mapping each module to its perturbation distance score.
    """
    scores = {}
    if outs.nelement() == 0 or inps.nelement() == 0:
        logger.warning("Input or output tensor for perturbation distance calculation is empty.")
        for i, module in enumerate(modules):
            scores[module] = 0.0
        return scores

    num_samples, num_layers, _, _ = outs.shape

    if num_layers != len(modules) or inps.shape[1] != len(modules):
        logger.error(f"Mismatch in number of layers. Tensors have {num_layers} layers, but {len(modules)} modules were provided.")
        for module in modules:
            scores[module] = 0.0
        return scores

    for i, module in enumerate(modules):
        layer_inputs = inps[:, i, :, :].to(torch.float32)
        layer_outputs = outs[:, i, :, :].to(torch.float32)
        diff = (layer_outputs - layer_inputs).view(num_samples, -1)
        distances = torch.norm(diff, p=2, dim=1)
        scores[module] = torch.mean(distances).item()
    return scores

def cal_activation_aware_sensitivity_all(modules, inps, outs) -> dict:
    """
    Computes a sensitivity score based on perturbation distance and input activation magnitude.
    A layer is considered sensitive if it both transforms data significantly (high perturb_dist)
    and processes high-magnitude activations.
    """
    if outs.nelement() == 0 or inps.nelement() == 0:
        logger.warning("Input or output tensor for sensitivity calculation is empty.")
        return {module: 0.0 for module in modules}

    num_samples, num_layers, _, _ = outs.shape

    if num_layers != len(modules) or inps.shape[1] != len(modules):
        logger.error(f"Mismatch in number of layers. Tensors have {num_layers} layers, but {len(modules)} modules were provided.")
        return {module: 0.0 for module in modules}

    perturb_scores_list = []
    input_mag_list = []

    for i in range(num_layers):
        layer_inputs = inps[:, i, :, :].to(torch.float32)
        layer_outputs = outs[:, i, :, :].to(torch.float32)

        # 1. Perturbation Distance
        diff = (layer_outputs - layer_inputs).view(num_samples, -1)
        distances = torch.norm(diff, p=2, dim=1)
        perturb_scores_list.append(torch.mean(distances).item())

        # 2. Input Activation Magnitude
        avg_mag = torch.mean(torch.abs(layer_inputs)).item()
        input_mag_list.append(avg_mag)

    # Normalize both scores to [0, 1]
    perturb_scores_t = torch.tensor(perturb_scores_list, dtype=torch.float32)
    input_mags_t = torch.tensor(input_mag_list, dtype=torch.float32)

    norm_perturb = (perturb_scores_t - torch.min(perturb_scores_t)) / (torch.max(perturb_scores_t) - torch.min(perturb_scores_t) + 1e-8)
    norm_mags = (input_mags_t - torch.min(input_mags_t)) / (torch.max(input_mags_t) - torch.min(input_mags_t) + 1e-8)

    # Combine by multiplication
    combined_scores = norm_perturb * norm_mags

    return {module: combined_scores[i].item() for i, module in enumerate(modules)}

def cal_dynamic_combined_score_all(modules, inps, outs) -> dict:
    """
    Computes a dynamic combined score based on perturbation distance and LIM score,
    weighted by the average bit-width of the model.
    """
    if outs.nelement() == 0 or inps.nelement() == 0:
        logger.warning("Input or output tensor for dynamic combined score calculation is empty.")
        return {module: 0.0 for module in modules}

    num_samples, num_layers, _, _ = outs.shape

    if num_layers != len(modules) or inps.shape[1] != len(modules):
        logger.error(f"Mismatch in number of layers. Tensors have {num_layers} layers, but {len(modules)} modules were provided.")
        return {module: 0.0 for module in modules}

    # Calculate average bit-width
    total_bits = sum(getattr(m, 'current_bit', 16) for m in modules)
    avg_bits = total_bits / len(modules) if modules else 16

    # Calculate dynamic weight alpha
    alpha = 0.5 + 0.3 * ((avg_bits - 2) / 14)  # Scales from 0.5 (at 2 bits) to 0.8 (at 16 bits)
    logger.info(f"Dynamic metric: avg_bits={avg_bits:.2f}, alpha={alpha:.4f}")

    perturb_scores = cal_perturb_dist_all(modules, inps, outs)
    lim_scores = cal_lim_score_all(modules, inps, outs)
    
    

    # Normalize scores
    p_scores = torch.tensor(list(perturb_scores.values()), dtype=torch.float32)
    l_scores = torch.tensor(list(lim_scores.values()), dtype=torch.float32)
    norm_p = (p_scores - torch.min(p_scores)) / (torch.max(p_scores) - torch.min(p_scores) + 1e-8)
    norm_l = (l_scores - torch.min(l_scores)) / (torch.max(l_scores) - torch.min(l_scores) + 1e-8)

    dynamic_scores = alpha * norm_p + (1 - alpha) * norm_l
    
    metrics = {}
    for key, value in METRIC_MAPPING.items():
        if "_combined" in key:
            continue
        try:
            metrics[key] = list(value(modules, inps, outs).values())
        except :
            pass
        
    metrics["dynamic_combined"] = list({module: dynamic_scores[i].item() for i, module in enumerate(modules)}.values())
    
    scores = {}
    
    for i, module in enumerate(modules):
        scores[module] = ""
        for key in metrics.keys():
            scores[module] += f" {key} : {metrics[key][i]:.4f}"
            
    for module, score in scores.items():
        logger.info(f"Layer: {module.name}, {score}")
    
    
    return {module: dynamic_scores[i].item() for i, module in enumerate(modules)}

def cal_enhanced_combined_score_all(modules, inps, outs) -> dict:
    """
    Computes an enhanced combined score based on perturbation distance, LIM score, and z-score.
    """
    if outs.nelement() == 0 or inps.nelement() == 0:
        logger.warning("Input or output tensor for enhanced combined score calculation is empty.")
        return {module: 0.0 for module in modules}

    # 1. Individual metric calculation
    perturb_scores = cal_perturb_dist_all(modules, inps, outs)
    lim_scores = cal_lim_score_all(modules, inps, outs)
    z_scores = cal_z_score_all(modules, inps, outs)

    # 2. Convert to tensors for normalization
    p_scores = torch.tensor(list(perturb_scores.values()), dtype=torch.float32)
    l_scores = torch.tensor(list(lim_scores.values()), dtype=torch.float32)

    # 3. Normalization (handle cases with single or no valid scores)
    if (p_scores.max() - p_scores.min()) > 1e-8:
        norm_p = (p_scores - p_scores.min()) / (p_scores.max() - p_scores.min())
    else:
        norm_p = torch.zeros_like(p_scores)

    if (l_scores.max() - l_scores.min()) > 1e-8:
        norm_l = (l_scores - l_scores.min()) / (l_scores.max() - l_scores.min())
    else:
        norm_l = torch.zeros_like(l_scores)

    z_scores_norm_map = {}
    valid_z_scores = [s for s in z_scores.values() if s != float('inf')]
    if valid_z_scores:
        z_min, z_max = min(valid_z_scores), max(valid_z_scores)
        if (z_max - z_min) > 1e-8:
            for i, module in enumerate(modules):
                if z_scores[module] != float('inf'):
                    z_scores_norm_map[module] = (z_scores[module] - z_min) / (z_max - z_min)
                else:
                    z_scores_norm_map[module] = 1.0 # Assign max importance if no linear layers
        else:
            for module in modules:
                z_scores_norm_map[module] = 0.0
    else:
        for module in modules:
            z_scores_norm_map[module] = 0.0

    norm_z_list = [z_scores_norm_map[m] for m in modules]
    norm_z = torch.tensor(norm_z_list, dtype=torch.float32)

    # 4. Weighted combination (weights can be tuned)
    # Example weights: perturb_dist: 0.5, lim_score: 0.3, z-score: 0.2
    combined_scores = 0.5 * norm_p + 0.3 * norm_l + 0.2 * norm_z

    return {module: combined_scores[i].item() for i, module in enumerate(modules)}

def cal_activation_aware_sensitivity_all(modules, inps, outs) -> dict:
    """
    Computes a sensitivity score based on perturbation distance and input activation magnitude.
    A layer is considered sensitive if it both transforms data significantly (high perturb_dist)
    and processes high-magnitude activations.
    """
    if outs.nelement() == 0 or inps.nelement() == 0:
        logger.warning("Input or output tensor for sensitivity calculation is empty.")
        return {module: 0.0 for module in modules}

    num_samples, num_layers, _, _ = outs.shape

    if num_layers != len(modules) or inps.shape[1] != len(modules):
        logger.error(f"Mismatch in number of layers. Tensors have {num_layers} layers, but {len(modules)} modules were provided.")
        return {module: 0.0 for module in modules}

    perturb_scores_list = []
    input_mag_list = []

    for i in range(num_layers):
        layer_inputs = inps[:, i, :, :].to(torch.float32)
        layer_outputs = outs[:, i, :, :].to(torch.float32)

        # 1. Perturbation Distance
        diff = (layer_outputs - layer_inputs).view(num_samples, -1)
        distances = torch.norm(diff, p=2, dim=1)
        perturb_scores_list.append(torch.mean(distances).item())

        # 2. Input Activation Magnitude
        avg_mag = torch.mean(torch.abs(layer_inputs)).item()
        input_mag_list.append(avg_mag)

    # Normalize both scores to [0, 1]
    perturb_scores_t = torch.tensor(perturb_scores_list, dtype=torch.float32)
    input_mags_t = torch.tensor(input_mag_list, dtype=torch.float32)

    norm_perturb = (perturb_scores_t - torch.min(perturb_scores_t)) / (torch.max(perturb_scores_t) - torch.min(perturb_scores_t) + 1e-8)
    norm_mags = (input_mags_t - torch.min(input_mags_t)) / (torch.max(input_mags_t) - torch.min(input_mags_t) + 1e-8)

    # Combine by multiplication
    combined_scores = norm_perturb * norm_mags

    return {module: combined_scores[i].item() for i, module in enumerate(modules)}

def cal_combined_score_all(modules, inps, outs) -> dict:
    """
    Computes a combined score based on perturbation distance and LIM score.

    This metric combines the normalized `perturb_dist` and `lim_score` to provide a
    more robust measure of layer importance. It gives higher weight to `perturb_dist`
    (0.7) as it captures the magnitude of transformation, while `lim_score` (0.3)
    captures the change in direction. A higher score indicates a more important layer.

    Parameters:
        modules (iterable): A collection of module objects to evaluate.
        inps (torch.Tensor): A tensor of captured inputs.
        outs (torch.Tensor): A tensor capturing the output vectors.

    Returns:
        dict: A dictionary mapping each module to its combined importance score.
    """
    if outs.nelement() == 0 or inps.nelement() == 0:
        logger.warning("Input or output tensor for combined score calculation is empty.")
        return {module: 0.0 for module in modules}

    num_samples, num_layers, _, _ = outs.shape

    if num_layers != len(modules) or inps.shape[1] != len(modules):
        logger.error(f"Mismatch in number of layers. Tensors have {num_layers} layers, but {len(modules)} modules were provided.")
        return {module: 0.0 for module in modules}

    perturb_scores_list = []
    lim_scores_list = []

    for i in range(num_layers):
        layer_inputs = inps[:, i, :, :].to(torch.float32)
        layer_outputs = outs[:, i, :, :].to(torch.float32)

        # Calculate perturb_dist for the layer
        diff = (layer_outputs - layer_inputs).view(num_samples, -1)
        distances = torch.norm(diff, p=2, dim=1)
        perturb_scores_list.append(torch.mean(distances).item())

        # Calculate lim_score for the layer
        lim_scores_list.append(cal_lim_score(layer_inputs, layer_outputs))

    # Convert lists to tensors for normalization
    perturb_scores = torch.tensor(perturb_scores_list, dtype=torch.float32)
    lim_scores = torch.tensor(lim_scores_list, dtype=torch.float32)

    # Normalize scores to [0, 1] range where higher is more important
    norm_perturb = (perturb_scores - torch.min(perturb_scores)) / (torch.max(perturb_scores) - torch.min(perturb_scores) + 1e-8)
    norm_lim = (lim_scores - torch.min(lim_scores)) / (torch.max(lim_scores) - torch.min(lim_scores) + 1e-8)

    # Combine normalized scores with weights
    combined_scores = 0.7 * norm_perturb + 0.3 * norm_lim

    # Create the final dictionary
    return {module: combined_scores[i].item() for i, module in enumerate(modules)}

def cal_shapley_score_all(modules, model, tokenizer, dataset, n_samples=4, shap_samples='auto') -> dict:
    """
    Computes layer importance using Shapley values from the 'shap' library.

    This metric offers a more theoretically grounded approach to measuring layer importance by
    quantifying each layer's marginal contribution to the model's performance on sample data.
    A higher Shapley value indicates a more important layer.

    NOTE: This method is computationally intensive compared to other proxy metrics.
    The cost scales with the number of layers, `n_samples` (prompts from the dataset),
    and `shap_samples` (coalitions sampled by SHAP).

    Parameters:
        modules (list): A list of module objects (layers) to be scored.
        model: The full transformer model.
        tokenizer: The model's tokenizer.
        dataset: A dataset of calibration data.
        n_samples (int): The number of prompts to use from the dataset for calculation.
        shap_samples (int or 'auto'): The number of coalitions for SHAP's KernelExplainer to sample.
                                      'auto' uses a default from the shap library.

    Returns:
        dict: A dictionary mapping each module to its Shapley value score.
    """
    logger.info("Calculating Shapley values for layer importance. This may take a very long time.")

    # 1. Prepare a batch of sample data
    prompts = [dataset[i]['text'] for i in range(n_samples)]
    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True, max_length=256
    ).to(model.device)

    # 2. Define the prediction function for SHAP
    n_layers = len(modules)

    def predict(coalition_masks):
        # coalition_masks is a numpy array of shape (num_shap_samples, n_layers)
        num_masks = coalition_masks.shape[0]
        total_loss = np.zeros(num_masks)

        for i in range(num_masks):
            mask = coalition_masks[i]
            hooks = []

            for layer_idx, layer in enumerate(modules):
                if mask[layer_idx] == 0:  # If layer is "off" in the coalition
                    # Bypass the layer by making it an identity function
                    hook = layer.register_forward_hook(lambda module, input, output: input)
                    hooks.append(hook)

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = inputs.input_ids[..., 1:].contiguous()
                loss_fct = torch.nn.CrossEntropyLoss()
                # Use negative loss as SHAP explains the function's output. Higher is better.
                loss = -loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                total_loss[i] = loss.item()

            for hook in hooks:
                hook.remove()

        return total_loss

    # 3. Run SHAP explainer
    explainer = shap.KernelExplainer(predict, np.zeros((1, n_layers)))
    shap_values = explainer.shap_values(np.ones((1, n_layers)), nsamples=shap_samples)

    return {module: shap_values[0, i] for i, module in enumerate(modules)}

def cal_hybrid_shap_score_all(modules, model, tokenizer, dataset, inps, outs, n_samples=4, shap_samples=32) -> dict:
    """
    Computes a hybrid importance score by combining Shapley values with other fast proxy metrics.

    This approach leverages the theoretical soundness of Shapley values (measuring performance
    contribution) with the speed and diverse insights of other metrics like perturbation
    distance and LIM score. It requires both captured tensors (inps, outs) for fast metrics
    and direct model access for Shapley values.

    Parameters:
        modules (list): A list of module objects (layers) to be scored.
        model: The full transformer model.
        tokenizer: The model's tokenizer.
        dataset: A dataset of raw calibration data (for Shapley).
        inps (torch.Tensor): A tensor of captured inputs (for fast metrics).
        outs (torch.Tensor): A tensor of captured outputs (for fast metrics).
        n_samples (int): The number of prompts for Shapley calculation.
        shap_samples (int or 'auto'): The number of coalitions for SHAP's KernelExplainer.

    Returns:
        dict: A dictionary mapping each module to its final hybrid score.
    """
    logger.info("Calculating hybrid score with Shapley values. This will be time-consuming.")

    # 1. Calculate fast proxy metrics first using captured tensors
    perturb_scores = cal_perturb_dist_all(modules, inps, outs)
    lim_scores = cal_lim_score_all(modules, inps, outs)

    # 2. Calculate the computationally expensive Shapley values
    shap_scores = cal_shapley_score_all(
        modules, model, tokenizer, dataset, n_samples=n_samples, shap_samples=shap_samples
    )

    # 3. Convert all scores to tensors for normalization
    p_scores = torch.tensor(list(perturb_scores.values()), dtype=torch.float32)
    l_scores = torch.tensor(list(lim_scores.values()), dtype=torch.float32)
    s_scores = torch.tensor(list(shap_scores.values()), dtype=torch.float32)

    # 4. Normalize each metric's scores to a [0, 1] range
    def normalize(scores_tensor):
        if (scores_tensor.max() - scores_tensor.min()) > 1e-8:
            return (scores_tensor - scores_tensor.min()) / (scores_tensor.max() - scores_tensor.min())
        return torch.zeros_like(scores_tensor)

    norm_p, norm_l, norm_s = normalize(p_scores), normalize(l_scores), normalize(s_scores)

    # 5. Combine with weights (tunable). Give higher weight to Shapley.
    hybrid_scores = 0.5 * norm_s + 0.3 * norm_p + 0.2 * norm_l

    return {module: hybrid_scores[i].item() for i, module in enumerate(modules)}

def cal_stabilized_combined_score_all(modules, inps, outs, **kwargs) -> dict:
    """
    가장 안정적인 `dynamic_combined`를 기반으로 AWQ 프록시 점수를 결합하여
    안정성과 정확성을 모두 향상시킨 메트릭입니다.
    """
    logger.info("Calculating StabilizedCombinedMetric scores.")

    # 1. 안정적인 dynamic_combined 점수를 계산합니다.
    dynamic_scores = cal_dynamic_combined_score_all(modules, inps, outs, **kwargs)

    # 2. AWQ 프록시 점수를 계산하여 보조 지표로 활용합니다.
    awq_scores = cal_awq_proxy_score_all(modules, inps, outs, **kwargs)
    
    def normalize_scores(scores: dict) -> dict:
        """
        모듈 객체를 키로 갖는 점수 딕셔너리를 받아 [0, 1] 범위로 정규화합니다.
        점수가 없는 경우 빈 딕셔너리를 반환합니다.
        """
        if not scores:
            return {}

        values = torch.tensor(list(scores.values()), dtype=torch.float32)
        min_val, max_val = torch.min(values), torch.max(values)

        if (max_val - min_val) > 1e-8:
            normalized_values = (values - min_val) / (max_val - min_val)
        else:
            normalized_values = torch.zeros_like(values)

        return {key: normalized_values[i].item() for i, key in enumerate(scores.keys())}

    # 3. 두 점수를 정규화합니다.
    norm_dynamic = normalize_scores(dynamic_scores)
    norm_awq = normalize_scores(awq_scores)

    combined_scores = {}
    # 4. 두 점수를 가중 평균하여 최종 점수를 계산합니다. (안정성을 위해 dynamic_combined에 더 높은 가중치 부여)
    weight = 0.7 
    for module in modules:
        score = (weight * norm_dynamic.get(module, 0)) + ((1 - weight) * norm_awq.get(module, 0))
        combined_scores[module] = score
        
    logger.info(f"Finished calculating StabilizedCombinedMetric scores (dynamic * {weight} + awq * {1-weight}).")
    
    return combined_scores

# Mapping of metric names to their respective calculation functions.
METRIC_MAPPING = {
    "z-score": cal_z_score_all,
    "lim": cal_lim_score_all,
    "act_norm": cal_activation_norm_all,
    "act_var": cal_activation_variance_all,
    "awq_proxy": cal_awq_proxy_score_all,
    "perturb_dist": cal_perturb_dist_all,
    "sensitivity": cal_activation_aware_sensitivity_all,
    "dynamic_combined": cal_dynamic_combined_score_all,
    "combined": cal_combined_score_all,
    "enhanced_combined": cal_enhanced_combined_score_all,
    "shapely" : cal_shapley_score_all,
    "hybrid_shap": cal_hybrid_shap_score_all,
    "stabilized_combined": cal_stabilized_combined_score_all,
}