import torch
import numpy as np

def _get_padded_features(features, pad_value=np.nan):
    """
    Pads variable-length sequences for RNN inputs to a consistent length with a specified padding value.

    This function was inspired by similar utility functions found in the DeepSurvivalMachines repository.
    For more details, see: https://github.com/autonlab/DeepSurvivalMachines/blob/c454774199c389e7bb9fa3077f153cdf4f1e7696/dsm/utilities.py

    Parameters
    ----------
    features : list of numpy.ndarray
        A list of arrays where each array is a sequence of feature vectors for a sample. 
        Each sequence may have a different length.
    pad_value : float, optional
        The value used to pad shorter sequences to match the length of the longest sequence. 
        The default is numpy.nan.

    Returns
    -------
    numpy.ndarray
        A numpy array containing the padded feature sequences. All sequences will have the same
        length equal to the length of the longest sequence in the original list.

    Examples
    --------
    >>> features = [np.array([[1, 2], [3, 4]]), np.array([[5, 6]])]
    >>> pad_rnn_inputs(features)
    array([[[ 1.,  2.],
            [ 3.,  4.]],

           [[ 5.,  6.],
            [nan, nan]]])
    """
    max_length = max(len(sample) for sample in features)  # Find the maximum sequence length
    padded_features = []

    for sample in features:
        padding_length = max_length - len(sample)
        padding_shape = (padding_length,) + sample.shape[1:]  # Shape of the padding needed
        padding = np.full(padding_shape, pad_value)  # Create padding with the pad_value
        padded_sample = np.concatenate([sample, padding])  # Concatenate the original sample with the padding
        padded_features.append(padded_sample)

    return np.array(padded_features)

def _prepare_rt_tensors(x, t, e, d, device='cpu', pad_value=100, train_ratio=0.70, val_ratio=0.10, test_ratio=0.20):
    """
    Prepares recurrent event data with terminal events for training in PyTorch.

    Parameters
    ----------
    x : list of numpy.ndarray
        List of feature arrays for each sample.
    t : list or array
        Times at which events or censoring are observed.
    e : list of numpy.ndarray
        Times of recurrent events for each sample.
    d : list or array
        Indicator for censoring in survival data (1 if the event is observed, 0 if censored).
    device : str, optional
        The device to which the tensors will be moved ('cpu' or 'cuda').
    pad_value : int, optional
        The value used for padding variable length sequences of recurrent events.
    train_ratio : float, optional
        Fraction of the data to be used for training (default is 0.70).
    val_ratio : float, optional
        Fraction of the data to be used for validation (default is 0.10).
    test_ratio : float, optional
        Fraction of the data to be used for testing (default is 0.20).

    Returns
    -------
    dict
        A dictionary containing tensors for features, times, events, deaths and their respective splits: train, test, and validation.
    """
    # Pad the recurrent event sequences to make PyTorch tensors
    e = _get_padded_features(e, pad_value=pad_value)

    # Convert all data to PyTorch tensors
    t = torch.tensor(t, dtype=torch.int64)
    x = torch.tensor(x, dtype=torch.float32)
    e = torch.tensor(e, dtype=torch.int64)
    d = torch.tensor(d, dtype=torch.int64)

    # Move all tensors to the specified device
    x = x.to(device)
    t = t.to(device)
    e = e.to(device)
    d = d.to(device)

    # Calculate dataset sizes for splits
    n = len(x)
    tr_size = int(n * train_ratio)
    vl_size = int(n * val_ratio)
    te_size = n - tr_size - vl_size  # Adjust test size to ensure all data is used

    # Split the data into training, validation, and test sets
    data = {
        'x_train': x[:tr_size], 'x_val': x[tr_size:tr_size + vl_size], 'x_test': x[-te_size:],
        't_train': t[:tr_size], 't_val': t[tr_size:tr_size + vl_size], 't_test': t[-te_size:],
        'e_train': e[:tr_size], 'e_val': e[tr_size:tr_size + vl_size], 'e_test': e[-te_size:],
        'd_train': d[:tr_size], 'd_val': d[tr_size:tr_size + vl_size], 'd_test': d[-te_size:]
    }

    return data
