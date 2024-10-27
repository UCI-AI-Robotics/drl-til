import torch
from torch import nn

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()

        # Quiz: Fill in the missing layers
        # Layer 1: Linear(28*28, 512)
        # Layer 2: ReLU
        # Layer 3: Linear(512, 512)
        # Layer 4: ReLU
        # Layer 5: Linear(512, 10)
        self.layer = nn.Sequential(
        )

    def forward(self, x):
        data = self.flatten(x)
        output = self.layer(data)
        return output

model = NeuralNetwork().to(device)

print(f"Model structure: {model}\n\n")
for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

X = torch.rand(1, 28, 28, device=device)
model_output = model(X)

pred_result = nn.Softmax(dim=1)(model_output)
print(f"pred_result : {pred_result.argmax(1)}")
