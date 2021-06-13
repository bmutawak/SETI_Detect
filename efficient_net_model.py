from efficientnet_pytorch import EfficientNet
import torch


class EfficientNetModel(torch.nn.Module):
    """
    EfficientNet model used for training/evaluation. Slightly modified
    to account for single class + 6 input channels
    
    """
    def __init__(self, in_channels, out_channels, out_dim, state_dict=None):
        super(EfficientNetModel, self).__init__()
        self._model = EfficientNet.from_pretrained('efficientnet-b0')
        
        if not state_dict is None:
            self._model.load_state_dict(state_dict)
        
        # Define custom layers
        self.out_layer = torch.nn.Linear(1000, out_dim)
        self.down_channel_layer = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.seq = torch.nn.Sequential(self.down_channel_layer,
                                       self._model,
                                       self.out_layer)
        return
    
    def forward(self, x):
        x = self.seq(x)
        return x