import torch
import torch.nn as nn
from torchvision import models


class EfficientNetCustom(nn.Module):
    def __init__(self, num_classes, size_inner=100, droprate=0.2):
        super(EfficientNetCustom, self).__init__()
        self.base_model = models.efficientnet_b0(weights='IMAGENET1K_V1')
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.base_model.classifier = nn.Identity()
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.inner = nn.Linear(1280, size_inner)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(droprate)
        self.output = nn.Linear(size_inner, num_classes)

    def forward(self, x):
        x = self.base_model.features(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.inner(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.output(x)
        return x


def export_to_onnx(model, img_size=224, onnx_path="smoke_fire_classifier.onnx"):
    model.eval()
    dummy_input = torch.randn(1, 3, img_size, img_size)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        }
    )
    print(f"ONNX model saved to {onnx_path}")


if __name__ == "__main__":
    num_classes = 3
    inner_size = 200
    droprate = 0.4

    model = EfficientNetCustom(num_classes=num_classes, size_inner=inner_size, droprate=droprate)

    # Load saved weights
    model.load_state_dict(torch.load("smoke_fire_classifier.pth", map_location="cpu"))

    # Export
    export_to_onnx(model)
