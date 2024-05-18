import torch

def recurrent_loss(h, t, e):
    """
    Computes the loss for recurrent events using a log-likelihood function.

    Parameters
    ----------
    h : torch.Tensor
        Predicted hazard probabilities for each time step for each sample.
    t : torch.Tensor
        Times at which events are observed for each sample.
    e : torch.Tensor
        Event indicator array where e[i] < t[i] indicates the time of the event.

    Returns
    -------
    torch.Tensor
        Mean loss computed over all samples in the batch.
    """
    def L(h, t, e):
        # Create a mask for all time points up to time t, initially setting all to True
        mask = torch.ones_like(h[0:t], dtype=torch.bool)
        # Set mask to False for time points where events occurred
        mask[e[e < t]] = False

        events = (e[e <= t] - 1)
        indices_to_exclude = events.tolist()

        # Create a set of all indices
        all_indices = set(range(h[0:t].size(0)))

        # Remove the indices to exclude
        non_events = list(all_indices - set(indices_to_exclude))
        # if t == 0: 
        #     # skip if censoring or death at 0. This is an issue due to discretisation. 
        #     return torch.tensor(0.0)

        # Calculate the negative log-likelihood for both the event occurrences and non-occurrences
        return -1 * (torch.sum(torch.log(h[0:t][events])) + torch.sum(torch.log(1 - h[0:t][non_events])))


    # Calculate loss for each sample in the batch and store in a list
    losses = [L(h[i], t[i], e[i]) for i in range(h.shape[0])]
    # Return the average loss across all samples
    return torch.sum(torch.stack(losses)) / len(h)

def survival_loss(h, t, d):
    """
    Computes the survival analysis loss using log-likelihood for censored and uncensored data.

    Parameters
    ----------
    h : torch.Tensor
        Predicted survival probabilities at each time step for each sample.
    t : torch.Tensor
        Observed time (either time of event or censoring).
    d : torch.Tensor
        Indicator for censoring where d[i] = 1 if the event is observed (uncensored), and 0 if the data is censored.

    Returns
    -------
    torch.Tensor
        Mean loss computed over all samples in the batch.
    """
    def L_uncensored(h, t):
        # Log-likelihood contribution from observed event
        # if t == 0: 
        #     # skip if death at 0. This is an issue due to discretisation. 
        #     return torch.tensor(0.0)

        L1 = torch.log(h[t-1])
        # Log-likelihood contribution from survival until time t
        L2 = torch.sum(torch.log(1 - h[0:(t-1)]))

        # L3 = torch.sum(torch.log(1 - torch.prod(1 - h[0:t]))) 
        # L2 = torch.sum(torch.log(1 - torch.cat((h[:t-1], h[t:]), dim=0)))
        # Combine and return the negative log-likelihood for uncensored data
        return -1 * torch.sum(L1 + L2)
    
    def L_censored(h, t):
        # Return the negative log-likelihood for censored data, only considering survival
        return -1 * torch.sum(torch.log(1 - h[0:t]))

    def L(h, t, d):
        # Choose loss calculation based on whether data is censored or not
        if d == 1:
            return L_uncensored(h, t)
        else:
            return L_censored(h, t)

    # Calculate loss for each sample in the batch and store in a list
    losses = [L(h[i], t[i], d[i]) for i in range(h.shape[0])]
    # Return the average loss across all samples
    return torch.sum(torch.stack(losses)) / len(h)

def recurrent_terminal_loss(h, t, e, d):
    """
    Computes the combined loss for models that involve both survival and recurrent event data.

    Parameters
    ----------
    h : torch.Tensor
        Hazard function estimates for each sample, split into hazards for survival and intensities for recurrent events.
    t : torch.Tensor
        Times at which events or censoring are observed.
    e : torch.Tensor
        Times of recurrent events.
    d : torch.Tensor
        Indicator for censoring in survival data.

    Returns
    -------
    torch.Tensor
        The weighted sum of survival and recurrent event losses.
    """
    # Split the hazard predictions into separate tensors for survival and recurrent event analyses
    hazards = h[:, :, 0:1]  # For survival loss
    intensities = h[:, :, 1:2]  # For recurrent event loss
  
    # Calculate and return the sum of survival and recurrent event losses
    return survival_loss(hazards, t, d) + recurrent_loss(intensities, t, e)
