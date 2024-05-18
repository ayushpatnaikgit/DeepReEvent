import numpy as np
from sksurv.metrics import concordance_index_ipcw, brier_score, cumulative_dynamic_auc

def calculate_survival_metrics(predictions, d_train, t_train, d_test, t_test, d_val, t_val):
    """
    Calculate survival analysis metrics including concordance index, Brier score, and ROC AUC.

    Parameters:
    predictions (torch.Tensor): The survival predictions from a model.
    d_train, t_train (np.array): Censoring indicators and times for training data.
    d_test, t_test (np.array): Censoring indicators and times for test data.
    d_val, t_val (np.array): Censoring indicators and times for validation data.

    Returns:
    dict: A dictionary containing computed metrics for each time horizon.
    """
    out_risk = predictions.cpu().detach().numpy()
    cumulative_hazard = np.cumsum(out_risk, axis=1)
    survival_probabilities = np.exp(-cumulative_hazard)

    times = np.arange(1, max(t_test).long() + 1, 1)

    et_train = np.array([(d_train[i], t_train[i]) for i in range(len(d_train))],
                        dtype=[('e', 'bool'), ('t', 'float64')])
    et_test = np.array([(d_test[i], t_test[i]) for i in range(len(d_test))],
                        dtype=[('e', 'bool'), ('t', 'float64')])
    et_val = np.array([(d_val[i], t_val[i]) for i in range(len(d_val))],
                        dtype=[('e', 'bool'), ('t', 'float64')])

    # Calculate metrics
    cis = [concordance_index_ipcw(et_train, et_test, out_risk[:, i], times[i])[0] for i in range(len(times))]
    ### CHECK THIS:  out_risk[:, i]

    brs = brier_score(et_train, et_test, survival_probabilities[:, 1:int(max(t_test))], np.arange(1, int(max(t_test)), 1))[1]

    # roc_auc = [cumulative_dynamic_auc(et_train, et_test, out_risk[:, 1:int(max(t_test))], np.arange(1, int(max(t_test)), 1)[i])[0] for i in range(len(times))]

    metrics = {
        'Brier Score': brs,
        # 'ROC AUC': roc_auc,
        '25th Quantile CI': cis[int(len(cis)*0.25)],
        '50th Quantile CI': cis[int(len(cis)*0.5)],
        '75th Quantile CI': cis[int(len(cis)*0.75)]
    }

    return metrics

def recurrent_cindex(out_risk, event_times, max_time):
    current_time = 20
    # out_risk = out_risk.cpu().detach().numpy()
    current_time = 20

    expected_number_of_events = torch.zeros(out_risk.size(0))

    for i in range(out_risk.size(0)):
        clamped_time = min(current_time, t_test[i].item())
        expected_number_of_events[i] = out_risk[i, 0:clamped_time].sum()
    mask = event_times < current_time
    observed_number_of_events = mask.sum(dim=1)
    concordant_pairs = 0
    total_pairs = 0
    for i in range(0, expected_number_of_events.shape[1]):
        for j in range(0, expected_number_of_events.shape[1]):
            if expected_number_of_events[i] > expected_number_of_events[j] and observed_number_of_events[i] > observed_number_of_events[j]:
                concordant_pairs += 1
            total_pairs += 1  
    return concordant_pairs / total_pairs
