import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any


class ResidualBlock(nn.Module):
    """
    Residual Block für das DeepChess Neural Network.
    """

    def __init__(self, channels: int):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class ChessNetwork(nn.Module):
    """
    Neuronales Netzwerk für die Schach-KI mit Reinforcement Learning.
    Architektur inspiriert von AlphaZero, aber vereinfacht.
    """

    def __init__(
            self,
            input_channels: int = 14,  # Standardmäßig 14 Eingangskanäle (12 für Figuren + 2 für Spielzustand)
            num_res_blocks: int = 10,
            num_filters: int = 128,
            policy_output_size: int = 4672,  # Maximale Anzahl möglicher Züge (vereinfacht)
            fc_size: int = 256
    ):
        super(ChessNetwork, self).__init__()

        self.input_channels = input_channels
        self.num_res_blocks = num_res_blocks
        self.num_filters = num_filters
        self.policy_output_size = policy_output_size

        # Eingangslayer
        self.conv_input = nn.Conv2d(input_channels, num_filters, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(num_filters)

        # Residual Blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_filters) for _ in range(num_res_blocks)
        ])

        # Policy Head (Vorhersage der optimalen Züge)
        self.policy_conv = nn.Conv2d(num_filters, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, policy_output_size)

        # Value Head (Bewertung der Stellungen)
        self.value_conv = nn.Conv2d(num_filters, 32, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 8 * 8, fc_size)
        self.value_fc2 = nn.Linear(fc_size, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward-Pass durch das Netzwerk.

        Args:
            x: Eingabetensor der Form [batch_size, input_channels, 8, 8]

        Returns:
            Tuple aus:
            - policy: Wahrscheinlichkeitsverteilung der möglichen Züge
            - value: Bewertung der aktuellen Position (-1 bis 1)
        """
        # Gemeinsame Pfad
        x = F.relu(self.bn_input(self.conv_input(x)))

        for res_block in self.res_blocks:
            x = res_block(x)

        # Policy Head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(-1, 32 * 8 * 8)
        policy = self.policy_fc(policy)
        policy = F.log_softmax(policy, dim=1)

        # Value Head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(-1, 32 * 8 * 8)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy, value


class ChessValueNetwork(nn.Module):
    """
    Ein vereinfachtes neuronales Netzwerk, das nur Stellungsbewertungen liefert.
    Kann als Baseline oder einfachere Alternative zum vollen ChessNetwork verwendet werden.
    """

    def __init__(
            self,
            input_channels: int = 14,
            num_filters: int = 64,
            fc_size: int = 128
    ):
        super(ChessValueNetwork, self).__init__()

        # Konvolutionelle Schichten
        self.conv1 = nn.Conv2d(input_channels, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)
        self.conv3 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(num_filters)

        # Fully connected Schichten
        self.fc1 = nn.Linear(num_filters * 8 * 8, fc_size)
        self.fc2 = nn.Linear(fc_size, fc_size // 2)
        self.fc3 = nn.Linear(fc_size // 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward-Pass durch das Netzwerk.

        Args:
            x: Eingabetensor der Form [batch_size, input_channels, 8, 8]

        Returns:
            torch.Tensor: Bewertung der Position zwischen -1 und 1
        """
        # Konvolutionelle Schichten
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Flatten und fully connected Schichten
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))

        return x