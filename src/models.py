import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    """
    A simple implementation of a recurrent neural network (RNN) using PyTorch's built-in RNN layer.
    
    Attributes
    ----------
    rnn : torch.nn.RNN
        The RNN layer that processes the input sequence.

    Parameters
    ----------
    input_size : int
        The number of expected features in the input `x`.
    hidden_size : int
        The number of features in the hidden state `h`.
    num_events : int
        The number of event outcomes to predict per sample.

    Methods
    -------
    forward(x):
        Defines the forward pass of the model.

    Example
    -------
    >>> torch.manual_seed(0)
    >>> input_size = 10
    >>> hidden_size = 20
    >>> num_events = 2
    >>> seq_length = 15
    >>> batch_size = 3
    >>> x = torch.randn(batch_size, seq_length, input_size)
    >>> model = SimpleRNN(input_size, hidden_size, num_events)
    >>> output = model(x)
    >>> print("Output Shape:", output.shape)
    >>> print("Output Tensor:", output)
    """
    def __init__(self, input_size, hidden_size, output_size, num_events):
        super(SimpleRNN, self).__init__()
        self.num_events = num_events
        self.output_size = output_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_events) 
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        """
        Perform a forward pass of the RNN model using input tensor `x`.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor containing the features of the sequence.

        Returns
        -------
        torch.Tensor
            The output tensor after applying the RNN, reshaped to have specified number of outputs per sample
            and passed through a sigmoid activation function.
        """
        x = x.unsqueeze(1).repeat(1, self.output_size, 1)  # Replace this with time-varying covariates if needed
        output, hidden = self.rnn(x)  # Process input through RNN
        output = self.dropout(output)
        output = self.fc(output)
        output = self.dropout(output)
        return torch.sigmoid(output)  # Apply sigmoid activation function to output

class SimpleGRU(nn.Module):
    """
    A simple implementation of a recurrent neural network (GRU) using PyTorch's built-in GRU layer.
    
    Attributes
    ----------
    gru : torch.nn.GRU
        The GRU layer that processes the input sequence.

    Parameters
    ----------
    input_size : int
        The number of expected features in the input `x`.
    hidden_size : int
        The number of features in the hidden state `h`.
    output_size : int
        The size of the output layer.
    num_events : int
        The number of event outcomes to predict per sample.
    dropout_prob : float
        The probability of an element to be zeroed in the dropout layer.

    Methods
    -------
    forward(x):
        Defines the forward pass of the model.

    Example
    -------
    >>> torch.manual_seed(0)
    >>> input_size = 10
    >>> hidden_size = 20
    >>> output_size = 15
    >>> num_events = 2
    >>> batch_size = 3
    >>> x = torch.randn(batch_size, output_size, input_size)
    >>> model = SimpleGRU(input_size, hidden_size, output_size, num_events, dropout_prob=0.5)
    >>> output = model(x)
    >>> print("Output Shape:", output.shape)
    >>> print("Output Tensor:", output)
    """
    def __init__(self, input_size, hidden_size, output_size, num_events, dropout_prob=0.1):
        super(SimpleGRU, self).__init__()
        self.num_events = num_events
        self.output_size = output_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_events) 
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        """
        Perform a forward pass of the GRU model using input tensor `x`.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor containing the features of the sequence.

        Returns
        -------
        torch.Tensor
            The output tensor after applying the GRU, reshaped to have specified number of outputs per sample
            and passed through a sigmoid activation function.
        """
        x = x.unsqueeze(1).repeat(1, self.output_size, 1)  # Replace this with time-varying covariates if needed
        output, hidden = self.gru(x)  # Process input through GRU
        output = self.dropout(output)
        output = self.fc(output)
        output = self.dropout(output)
        return torch.sigmoid(output)  # Apply sigmoid activation function to output
