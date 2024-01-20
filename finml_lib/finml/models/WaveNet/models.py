import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.functional.regression import concordance_corrcoef

import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    """
    Focal Loss is designed to address class imbalance in classification tasks.

    Args:
        weight (Tensor, optional): A manual rescaling weight given to each class. Default is None.
        reduction (str, optional): Specifies the reduction to be applied to the output. 
                                  Should be one of: 'none', 'mean', 'sum'. Default is 'mean'.
        gamma (float, optional): Focusing parameter for adjusting the loss. Default is 0.
        eps (float, optional): Small value to prevent division by zero. Default is 1e-7.

    Attributes:
        gamma (float): Focusing parameter.
        eps (float): Small value to prevent division by zero.
        ce (CrossEntropyLoss): Cross-entropy loss function.

    Note:
        Focal Loss combines the standard cross-entropy loss with a focusing parameter (gamma) 
        to give more importance to hard-to-classify samples, addressing class imbalance.

    """
    def __init__(self, weight=None, reduction='mean', gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss(weight=weight, reduction=reduction)

    def forward(self, input, target):
        """
        Calculate the Focal Loss.

        Args:
            input (Tensor): Predicted logits from the model.
            target (Tensor): Ground truth labels.

        Returns:
            Tensor: Focal Loss value.

        """
        # Calculate the standard cross-entropy loss
        logp = self.ce(input, target)
        
        # Calculate the probability and focal loss
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp

        # Return the mean loss value
        return loss.mean()



class ConcordanceLoss(nn.Module):
    """
    Loss based on the concordance correlation coefficient.
    """
    def __init__(self, scale=100):
        super(ConcordanceLoss, self).__init__()
        self.scale = scale

    def forward(self, y_pred, y_true):
        """
        Calculate the concordance correlation coefficient loss.

        Args:
            y_pred (torch.Tensor): Tensor of predicted values.
            y_true (torch.Tensor): Tensor of true values.

        Returns:
            torch.Tensor: Concordance correlation coefficient loss.
        """
        concordance_loss = - concordance_corrcoef(self.scale * y_pred, self.scale * y_true)
        return concordance_loss




class CausalConv1d(nn.Module):
    """
    1D Causal Convolution Layer

    Args:
        in_size (int): Number of input channels.
        out_size (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel.
        dilation (int): Dilation factor for the convolution.
        stride (int): Stride of the convolution.
    """
    def __init__(self, in_size, out_size, kernel_size, dilation=1, stride=1):
        super(CausalConv1d, self).__init__()

        # Calculate padding to ensure causality
        self.pad = (kernel_size - 1) // 2 * dilation

        # Define the 1D causal convolutional layer
        self.conv1 = nn.Conv1d(in_size, out_size, kernel_size, padding=self.pad, stride=stride, dilation=dilation)

    def forward(self, x):
        """
        Forward pass of the CausalConv1d layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the causal convolution.
        """
        x = self.conv1(x)
        return x


class ResidualLayer(nn.Module):
    def __init__(self, residual_size, skip_size, dilation):
        """
        Residual Layer with Dilated Causal Convolutions

        Args:
            residual_size (int): Number of channels for the residual output.
                This parameter defines the size of the hidden representations within the block.
            skip_size (int): Number of channels for the skip connection output.
                Determines the size of the features preserved in the skip connection.
            dilation (int): Dilation factor for the causal convolutions.
                Controls the spacing between kernel elements, capturing long-range dependencies.
        """
        super(ResidualLayer, self).__init__()
        
        # Causal convolution filters (assuming CausalConv1d is defined elsewhere)
        self.conv_filter = CausalConv1d(residual_size, residual_size,
                                        kernel_size=3, dilation=dilation)
        self.conv_gate = CausalConv1d(residual_size, residual_size,
                                      kernel_size=3, dilation=dilation)   
        
        # 1x1 convolution layers for processing the output of the dilated convolution
        self.resconv1_1 = nn.Conv1d(residual_size, residual_size, kernel_size=1)
        self.skipconv1_1 = nn.Conv1d(residual_size, skip_size, kernel_size=1)

    def forward(self, x):
        """
        Forward pass of the Residual Layer

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, residual_size, seq_len).
        
        Returns:
            skip (torch.Tensor): Output tensor for the skip connection of shape (batch_size, skip_size, seq_len).
                Represents high-level features captured by the residual block.
            residual (torch.Tensor): Output tensor for the residual connection of shape (batch_size, residual_size, seq_len).
                Represents the sum of the input and the processed output.
        """
        # Causal convolution operations
        conv_filter = self.conv_filter(x)
        conv_gate = self.conv_gate(x)  
        
        # Gating mechanism using tanh and sigmoid
        fx = torch.tanh(conv_filter) * torch.sigmoid(conv_gate)
        
        # 1x1 convolutions to process the gated output
        fx = self.resconv1_1(fx) 
        
        # Output for the skip connection
        skip = self.skipconv1_1(fx) 
        
        # Residual connection: add the gated output to the input
        residual = fx + x  
        
        # Return the skip connection and the residual
        # skip=[batch, skip_size, seq_len], residual=[batch, residual_size, seq_len]
        return skip, residual



class DilatedStack(nn.Module):
    def __init__(self, residual_size, skip_size, dilation_depth):
        """
        Dilated Stack of Residual Layers

        Args:
            residual_size (int): Number of channels for the residual output.
            skip_size (int): Number of channels for the skip connection output.
            dilation_depth (int): Depth of the dilation in the stack.
        """
        super(DilatedStack, self).__init__()

        # Create a stack of dilated residual layers
        self.residual_stack = nn.ModuleList([
            ResidualLayer(residual_size, skip_size, 2**layer)
            for layer in range(dilation_depth)
        ])
        
    def forward(self, x):
        """
        Forward pass of the Dilated Stack.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Skip connection outputs (stacked over layers).
            torch.Tensor: Final residual output.
        """
        skips = []

        # Apply each residual layer in the stack
        for layer in self.residual_stack:
            skip, x = layer(x)
            skips.append(skip.unsqueeze(0))

        # Stack skip connection outputs along the layers dimension
        skips = torch.cat(skips, dim=0)

        return skips, x  # [layers,batch,skip_size,seq_len]


class WaveNet(pl.LightningModule):
    def __init__(self, args):
        """
        WaveNet model.

        Args:
            args (Namespace): Command-line arguments.

            Detailed Explanation of Arguments:
            - input_dim (int): The dimensionality of the input features.
            - output_dim (int): The desired output dimensionality of the model.
            - residual_dim (int): The dimensionality of the residual blocks in the model.
            - skip_dim (int): The dimensionality of the skip connections.
            - dilation_cycles (int): The number of dilation cycles in the model.
            - dilation_depth (int): The depth of dilation in each residual block.
            - is_classification (bool): Is the task is classification.
            - learning_rate (float): The learning rate used by the optimizer during training.
        """
        super(WaveNet, self).__init__()
        self.args = args
        self.save_hyperparameters()
        self.input_dim = args.input_dim
        self.output_dim = args.output_dim
        self.residual_dim = args.residual_dim
        self.skip_dim = args.skip_dim
        self.dilation_cycles = args.dilation_cycles
        self.dilation_depth = args.dilation_depth
        self.is_classification = args.is_classification

        self.input_conv = CausalConv1d(self.input_dim, self.residual_dim, kernel_size=3)

        self.dilated_stacks = nn.ModuleList(
            [DilatedStack(self.residual_dim, self.skip_dim, self.dilation_depth)
             for cycle in range(self.dilation_cycles)]
        )
        self.attention_weights = nn.Parameter(torch.ones(self.dilation_depth))
        
       
        self.linear = nn.Linear(self.skip_dim, self.output_dim)
        self.tanh = nn.Tanh()

        # Criterion 1 represents almost the same
        if self.is_classification:
            self.loss_func = FocalLoss(gamma=1)
            self.criterion = FocalLoss(gamma=1) 
        else:
            self.loss_func = ConcordanceLoss(args.scale)
            self.criterion = concordance_corrcoef

    def forward(self, x):
        """
        Forward pass of the WaveNet model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Model output.
        """
        # [batch_size, seq_len, feature_dim] => [batch_size, feature_dim, seq_len]
        x = x.permute(0, 2, 1)

        # [batch_size, residual_dim, seq_len]
        x = self.input_conv(x)  

        skip_connections = []

        for cycle in self.dilated_stacks:
            skips, x = cycle(x)
            skip_connections.append(skips)

        # Compute mean value of skip connections
        #[dilation_cycles, dilation_depth, batch_size, skip_size, seq_len]=>
        #[dilation_depth, batch_size, skip_size, seq_len]
        mean_skip_connections = torch.mean(torch.stack(skip_connections), dim=0)

        # Weighted sum of mean_skip_connections
        attention_weights = torch.softmax(self.attention_weights, dim=0)
        attention_weights = attention_weights.view((self.dilation_depth, 1, 1, 1))
        #=>[batch_size, skip_size, seq_len]
        attended_skips = torch.sum(attention_weights * mean_skip_connections, dim=0)

        # [batch_size, skip_dim, seq_len] => [batch_size, seq_len, skip_dim]
        attended_skips = attended_skips.permute(0, 2, 1)
        #[batch_size, skip_dim] => [batch_size, output_dim] 
        signal = self.linear(attended_skips[:, -1, :])
        if self.is_classification:
            return signal
        else:
            return self.tanh(signal)

    def training_step(self, batch, batch_idx):
        """
        Training step for the WaveNet model.

        Args:
            batch (tuple): Tuple containing input data and return data.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Loss tensor.
        """
        input_data, label = batch
        signal = self(input_data)
        loss = self.loss_func(signal, label)

        # Log training loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the WaveNet model.

        Args:
            batch (tuple): Tuple containing input data and return data.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Loss tensor.
        """
        input_data, label = batch
        signal = self(input_data)
        loss = self.criterion(signal, label)

        # Log validation loss
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        """
        Configure the optimizer.

        Returns:
            torch.optim.Optimizer: Optimizer.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.learning_rate)

        #configure a learning rate scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        return [optimizer], [scheduler]


