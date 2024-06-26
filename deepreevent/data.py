import pandas as pd
import numpy as np
import math
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def _load_readmission_dataset(sequential):
    """
    Helper function to load and preprocess the readmission dataset.

    Parameters
    ----------
    sequential : bool
        If True, returns a list of numpy arrays for each individual, suitable
        for training recurrent neural models. If False, returns collapsed results
        for each time step.

    Returns
    -------
    Depending on the `sequential` parameter:
    - If sequential is False, returns (x_, time, event)
    - If sequential is True, returns (x, t, e, d)

    Example
    -------
    >>> x, t, e, d = _load_readmission_dataset(sequential=True)
    
    References
    ----------
    [1] Gonzalez, JR., Fernandez, E., Moreno, V., Ribes, J., Peris, M., Navarro, M.,
        Cambray, M., and Borras, JM (2005). "Sex differences in hospital readmission
        among colorectal cancer patients." Journal of Epidemiology and Community Health,
        59(6), 506-511.
    [2] Rondeau, Virginie, Yassin Marzroui, and Juan R. Gonzalez. "frailtypack: an R package
        for the analysis of correlated survival data with frailty models using penalized
        likelihood estimation or parametrical estimation." Journal of Statistical Software,
        47 (2012): 1-28.

    Acknowledgments
    ---------------
    The structure and approach of this function were inspired by the source code found at:
    https://github.com/autonlab/DeepSurvivalMachines/blob/c454774199c389e7bb9fa3077f153cdf4f1e7696/dsm/datasets.py
    The dataset used was downloaded from the 'frailtypack' package in R.
    """
    # Load dataset
    data = pd.read_csv('data/readmission.csv', delimiter=';')
    data['max_time'] = data.groupby('id')['t.stop'].transform(max)

    # Process categorical variables
    dat_cat = data[['chemo', 'sex', 'dukes', 'charlson']]
    x_ = pd.get_dummies(dat_cat).values
    
    # Calculate times and events
    event = data['t.stop'].values.reshape(-1, 1) / 365.25
    event_round = (data['t.stop'] / 365.25).apply(math.ceil).values.reshape(-1, 1)
    time = (data['max_time'] / 365.25).apply(math.ceil).values
    death = data['death'].values

    if not sequential:
        return x_, time, event
    else:
        x, t, d, e = [], [], [], []
        for id_ in sorted(list(set(data['id']))):
            group_indices = data['id'] == id_
            x_group = x_[group_indices]
            mode_x_group = pd.DataFrame(x_group).mode().values[0]
            x.append(mode_x_group)
            t.append(time[group_indices][0])
            e_tmp = event_round[group_indices]
            e.append(e_tmp.reshape(e_tmp.shape[0]))
            d.append(death[group_indices][0])
        return x, t, e, d

def _load_simData(sequential):
    """
    Helper function to load and preprocess the readmission dataset.

    Parameters
    ----------
    sequential : bool
        If True, returns a list of numpy arrays for each individual, suitable
        for training recurrent neural models. If False, returns collapsed results
        for each time step.

    Returns
    -------
    Depending on the `sequential` parameter:
    - If sequential is False, returns (x_, time, event)
    - If sequential is True, returns (x, t, e, d)

    Example
    -------
    >>> x, t, e, d = _load_readmission_dataset(sequential=True)
    
    References
    ----------
    [1] Gonzalez, JR., Fernandez, E., Moreno, V., Ribes, J., Peris, M., Navarro, M.,
        Cambray, M., and Borras, JM (2005). "Sex differences in hospital readmission
        among colorectal cancer patients." Journal of Epidemiology and Community Health,
        59(6), 506-511.
    [2] Rondeau, Virginie, Yassin Marzroui, and Juan R. Gonzalez. "frailtypack: an R package
        for the analysis of correlated survival data with frailty models using penalized
        likelihood estimation or parametrical estimation." Journal of Statistical Software,
        47 (2012): 1-28.

    Acknowledgments
    ---------------
    The structure and approach of this function were inspired by the source code found at:
    https://github.com/autonlab/DeepSurvivalMachines/blob/c454774199c389e7bb9fa3077f153cdf4f1e7696/dsm/datasets.py
    The dataset used was downloaded from the 'frailtypack' package in R.
    """
    # Load dataset
    data = pd.read_csv('data/simData.csv', delimiter=';')
    data['max_time'] = data.groupby('id')['t.stop'].transform(max)

    # Process categorical variables
    dat_cat = data[['x1']]
    x_ = pd.get_dummies(dat_cat).values
    
    dat_cat = data[['x1']]
    dat_num = data[['x2']]

    x1 = pd.get_dummies(dat_cat).values
    x2 = dat_num.values
    x2 = StandardScaler().fit_transform(x2)
    x = np.hstack([x1, x2])

    # x = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(x)
    x_ = x  #StandardScaler().fit_transform(x)

    # Calculate times and events
    event = data['t.stop'].values.reshape(-1, 1)
    event_round = data['t.stop'].apply(math.ceil).values.reshape(-1, 1)
    time = data['max_time'].apply(math.ceil).values
    status = data['status'].values

    if not sequential:
        return x_, time, event
    else:
        x, t, d, e = [], [], [], []
        for id_ in sorted(list(set(data['id']))):
            x.append(x_[data['id'] == id_][0])
            t.append(time[data['id'] == id_][0])
            e_tmp = event_round[data['id'] == id_]
            e.append(e_tmp.reshape(e_tmp.shape[0]))
            d.append(status[data['id'] == id_][0])
        return x, t, e, d



