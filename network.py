import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    
    def __init__(self, num_classes = 10):
        
        super().__init__()
        
        self.num_classes = num_classes
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size = 5),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size = 2),
            nn.Conv2d(6, 16, kernel_size = 5),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size = 2)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, num_classes)  
        )
        
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        logit = self.classifier(x)
        return logit
    
    def save_model(self, filepath: str) -> None:
        """
        Save a PyTorch model's state_dict and config

        Args:
            model: the NeuralNet instance
            filepath: the file to save to
        """
        torch.save({
            'model_state_dict': self.state_dict(),
        }, filepath)
        print(f"Model saved to {filepath}")