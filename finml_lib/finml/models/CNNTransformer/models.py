import torch
import torch.nn as nn
import pytorch_lightning as pl


class SharpeLoss(object):
    """
    Loss of Sharpe ratio with penalties for transaction cost and holding cost.
    Aims for fewer open trades and a faster earning and exit strategy.
    """
    def __init__(
        self,
        trans_cost_ratio,
        hold_cost_ratio
    ):
        """
        Initialize the SharpeLoss object.

        Args:
            trans_cost_ratio (float): Ratio for transaction cost penalty.
            hold_cost_ratio (float): Ratio for holding cost penalty.
        """
        super().__init__()
        self.trans_cost_ratio = trans_cost_ratio
        self.hold_cost_ratio = hold_cost_ratio

    def __call__(self, return_data, weight_pred):
        """
        Calculate the Sharpe ratio loss with penalties for transaction cost and holding cost.

        Args:
            return_data (torch.Tensor): Tensor of shape [batch_size, asset_nums] representing returns.
            weight_pred (torch.Tensor): Tensor of shape [batch_size, asset_nums] representing predicted weights.

        Returns:
            torch.Tensor: Negative mean of final returns divided by standard deviation.
        """

        final_returns = torch.sum(return_data * weight_pred) - \
            self.trans_cost_ratio * torch.cat((
                torch.zeros(1, device=weight_pred.device),
                torch.sum(torch.abs(weight_pred[1:] - weight_pred[:-1]), axis=1)
            )) - self.hold_cost_ratio * torch.sum(torch.abs(weight_pred), axis=1)

        # Minimize the negative mean of final returns divided by standard deviation
        return -torch.mean(final_returns) / (torch.std(final_returns)+1e-8)


class CNN_Block(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        use_normalization=True,
        filter_size=2
    ):
        """
        Convolutional Neural Network (CNN) Block.

        Args:
            in_dim (int): Number of input channels (feature dimensions).
            out_dim (int): Number of output channels.
            use_normalization (bool, optional): Whether to use instance normalization. Default is True.
            filter_size (int, optional): Size of the convolutional filters. Default is 2.

        Note:
            The output dimension (out_dim) must be divisible by the input dimension (in_dim).

        This block consists of two convolutional layers with optional instance normalization and
        a residual connection.

        Shape of input and output tensors both looks like: [batch_size, feature_dim, sequence_length]
        """
        # Ensure that out_dim is a multiple of in_dim
        super().__init__()
        assert out_dim % in_dim == 0
        
        # Store the parameters as attributes
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_normalization = use_normalization
        self.filter_size = filter_size

        # Define the convolutional layers
        self.conv1 = nn.Conv1d(
            in_channels=in_dim,
            out_channels=out_dim,
            kernel_size=filter_size,
            padding="same"
        )
        self.conv2 = nn.Conv1d(
            in_channels=out_dim,
            out_channels=out_dim,
            kernel_size=filter_size,
            padding="same"
        )
        
        # Rectified Linear Unit (ReLU) activation function
        self.relu = nn.ReLU(inplace=True)
        
        # Instance normalization layers
        self.normalization1 = nn.InstanceNorm1d(in_dim)
        self.normalization2 = nn.InstanceNorm1d(out_dim)

    def forward(self, x):
        '''
        The input data  shape looks like [batch_size, feature_dim, sequence_length]
        The output data shape looks like [batch_size, feature_dim, sequence_length]
        '''
        # Apply instance normalization if specified

        if self.use_normalization:
            x = self.normalization1(x)
        
        # First convolutional layer
        out = self.conv1(x)
        out = self.relu(out)
        
        # Apply instance normalization if specified
        if self.use_normalization:
            out = self.normalization2(out)
        
        # Second convolutional layer
        out = self.conv2(out)
        out = self.relu(out)
        
        # Residual connection: add the input to the output
        out = out + x.repeat(1, int(self.out_dim / self.in_dim), 1)
        
        return out



class CNNTransformer(pl.LightningModule):
    def __init__(self, args):
        """
        CNNTransformer model combining CNN blocks and Transformer Encoder.

        Args:
            args (Namespace): A namespace containing model configuration parameters.

        Model Configuration Parameters:
            - filter_numbers (list): List of integers, specifying the number of filters in each CNN block.
            - hidden_units_factor (int): Factor to determine the number of hidden units in the Transformer Encoder.
            - use_normalization (bool): Whether to use normalization in the CNN blocks.
            - filter_size (int): Size of the filters in the CNN blocks.
            - attention_heads (int): Number of attention heads in the Transformer Encoder.
            - dropout (float): Dropout rate in the Transformer Encoder.
            - output_dim (int): Dimensionality of the final output.
            - trans_cost_ratio (float): Ratio for transaction cost in the SharpeLoss.
            - hold_cost_ratio (float): Ratio for holding cost in the SharpeLoss.
            - learning_rate (float): Learning rate for the optimizer.
            - limit_for_pair_trading (bool): Whether to use a limit for pair trading.

        Note:
            The input data shape should be [batch_size, feature_dim, sequence_length].
        """
        super(CNNTransformer, self).__init__()
        self.args = args
        self.save_hyperparameters()

        # Limit for pair trading
        self.limit_for_pair_trading = args.limit_for_pair_trading

        # Extract configuration parameters
        self.filter_numbers = args.filter_numbers
        self.hidden_units = args.hidden_units_factor * args.filter_numbers[-1]

        # List to store CNN blocks
        self.convBlocks = nn.ModuleList()

        # Create CNN blocks
        for i in range(len(args.filter_numbers) - 1):
            self.convBlocks.append(
                CNN_Block(
                    in_dim=args.filter_numbers[i],
                    out_dim=args.filter_numbers[i + 1],
                    use_normalization=args.use_normalization,
                    filter_size=args.filter_size
                )
            )

        assert args.filter_numbers[-1] % args.attention_heads == 0

        # Create Transformer Encoder layer
        self.encoder = nn.TransformerEncoderLayer(
            d_model=args.filter_numbers[-1],
            nhead=args.attention_heads,
            dim_feedforward=self.hidden_units,
            dropout=args.dropout,
            batch_first=True
        )

        # Linear layer for final output
        if self.limit_for_pair_trading:
            self.linear = nn.Linear(args.filter_numbers[-1], 1)
        else:
            self.linear = nn.Linear(args.filter_numbers[-1], args.output_dim)
        self.tanh = nn.Tanh()

        # SharpeLoss for training
        self.loss = SharpeLoss(args.trans_cost_ratio, args.hold_cost_ratio)


    def forward(self, x):
        """
        Forward pass of the CNNTransformer model.

        Args:
            x (torch.Tensor): Input tensor with shape [batch_size, sequence_length, feature_dim].

        Returns:
            torch.Tensor: Output tensor after passing through the model.
        """
        # [batch_size, sequence_length, feature_dim] => [batch_size, feature_dim, sequence_length]
        x = x.permute(0,2,1)
        # Apply CNN blocks
        for block in self.convBlocks:
            x = block(x)

        # [batch_size, feature_dim, sequence_length] => [batch_size, sequence_length, feature_dim]
        x = x.permute(0, 2, 1)

        # Apply Transformer Encoder
        x = self.encoder(x)

        weight = self.tanh(self.linear(x[:, -1, :]))
    
        if self.limit_for_pair_trading:
            #limit weight similar to [w,-w]
            return torch.cat([weight,-weight],dim=1)
        else:
            return weight 

    def training_step(self, batch, batch_idx):
        """
        Training step for the CNNTransformer model.

        Args:
            batch (tuple): Tuple containing input data and return data.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Loss tensor.
        """
        input_data, return_data = batch
        weight_pred = self(input_data)
        loss = self.loss(return_data, weight_pred)

        # Log training loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the CNNTransformer model.

        Args:
            batch (tuple): Tuple containing input data and return data.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Loss tensor.
        """
        input_data, return_data = batch
        weight_pred = self(input_data)
        loss = self.loss(return_data, weight_pred)

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

        # Uncomment and configure a learning rate scheduler if needed
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
        # return [optimizer], [scheduler] if using a scheduler

        return optimizer
