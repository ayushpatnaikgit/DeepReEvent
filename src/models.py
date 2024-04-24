import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    """
    A simple implementation of a recurrent neural network (RNN) using PyTorch's built-in RNN layer.
    
    Attributes
    ----------
    rnn : torch.nn.RNN
        The RNN layer that processes the input sequence.
    fc : torch.nn.Linear
        A fully connected layer that maps the hidden state to the output space corresponding to the specified number of events.

    Parameters
    ----------
    input_size : int
        The number of expected features in the input `x`.
    hidden_size : int
        The number of features in the hidden state `h`.
    output_size : int
        The number of features in the output for each event.
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
    >>> output_size = 5
    >>> num_events = 2
    >>> seq_length = 15
    >>> batch_size = 3
    >>> x = torch.randn(batch_size, seq_length, input_size)
    >>> model = SimpleRNN(input_size, hidden_size, output_size, num_events)
    >>> output = model(x)
    >>> print("Output Shape:", output.shape)
    >>> print("Output Tensor:", output)
    """
    def __init__(self, input_size, hidden_size, output_size, num_events):
        super(SimpleRNN, self).__init__()
        self.output_size = output_size
        self.num_events = num_events
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size * num_events)  # Adjust the output size based on the number of events

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
            The output tensor after applying the RNN and fully connected layer,
            reshaped to have specified number of outputs per sample and passed through a sigmoid activation function.
        """
        x = x.unsqueeze(1).repeat(1, self.output_size, 1)
        output, hidden = self.rnn(x)  # Process input through RNN
        output = self.fc(output[:, -1, :])  # Apply the fully connected layer to the last hidden state
        output = output.view(output.size(0), -1, self.num_events)  # Reshape output to have specified number of events per sample
        return torch.sigmoid(output)  # Apply sigmoid activation function to output
