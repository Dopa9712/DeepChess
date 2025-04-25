import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any


class EnhancedResidualBlock(nn.Module):
    """
    Enhanced residual block with squeeze-excitation and additional regularization.
    """

    def __init__(self, channels: int, se_ratio: int = 16):
        super(EnhancedResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

        # Squeeze-excitation block
        self.se = SqueezeExcitation(channels, reduction=se_ratio)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Apply squeeze-excitation
        out = self.se(out)

        out += residual
        out = F.relu(out)

        # Apply dropout
        out = self.dropout(out)

        return out


class SqueezeExcitation(nn.Module):
    """
    Squeeze-Excitation block for channel attention.
    """

    def __init__(self, channels, reduction=16):
        super(SqueezeExcitation, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y


class EnhancedChessNetwork(nn.Module):
    """
    Enhanced neural network for chess AI with reinforcement learning.
    Includes improvements like attention mechanisms, deeper architecture, and more.
    """

    def __init__(
            self,
            input_channels: int = 14,
            num_res_blocks: int = 20,
            num_filters: int = 256,
            policy_output_size: int = 64 * 64 + 64 * 64 * 4,
            fc_size: int = 512
    ):
        super(EnhancedChessNetwork, self).__init__()

        self.input_channels = input_channels
        self.num_res_blocks = num_res_blocks
        self.num_filters = num_filters
        self.policy_output_size = policy_output_size

        # Initial convolution layers with larger filters
        self.conv_input = nn.Conv2d(input_channels, num_filters, kernel_size=5, padding=2)
        self.bn_input = nn.BatchNorm2d(num_filters)

        # Residual blocks with squeeze-excitation
        self.res_blocks = nn.ModuleList([
            EnhancedResidualBlock(num_filters) for _ in range(num_res_blocks)
        ])

        # Policy head (predicting optimal moves)
        self.policy_conv = nn.Conv2d(num_filters, 64, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(64)
        self.policy_fc1 = nn.Linear(64 * 8 * 8, fc_size)
        self.policy_fc2 = nn.Linear(fc_size, policy_output_size)

        # Value head (evaluating positions)
        self.value_conv = nn.Conv2d(num_filters, 64, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(64)
        self.value_fc1 = nn.Linear(64 * 8 * 8, fc_size)
        self.value_fc2 = nn.Linear(fc_size, fc_size // 2)
        self.value_fc3 = nn.Linear(fc_size // 2, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape [batch_size, input_channels, 8, 8]

        Returns:
            Tuple of:
            - policy: Probability distribution over possible moves
            - value: Evaluation of the current position (-1 to 1)
        """
        # Shared path
        x = F.relu(self.bn_input(self.conv_input(x)))

        # Apply residual blocks
        for res_block in self.res_blocks:
            x = res_block(x)

        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.reshape(-1, 64 * 8 * 8)
        policy = F.relu(self.policy_fc1(policy))
        policy = self.policy_fc2(policy)
        policy = F.log_softmax(policy, dim=1)

        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.reshape(-1, 64 * 8 * 8)
        value = F.relu(self.value_fc1(value))
        value = F.relu(self.value_fc2(value))
        value = torch.tanh(self.value_fc3(value))

        return policy, value

