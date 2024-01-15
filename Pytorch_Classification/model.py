from torch import nn, Tensor
from torchvision import models

class ModelHandler():
    def __init__(self) -> None:
        self.models_dict = {"Transfer":TransferLearningModel,
                            "TinyVGG": TinyVGG,
                            "TensorFlowModel": TensorFlowModel}

    def get_model(self,
                  model_name: str = "TinyVGG",
                  **kwargs) -> nn.Module:
        return self.models_dict[model_name](**kwargs)


class TransferLearningModel(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        # Load a pre-trained ResNet model
        self.base_model = models.resnet50(pretrained=True)
        # Replace the last fully connected layer
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.base_model(x)

class TensorFlowModel(nn.Module):
    """
    def build_model(X_train):
    input_shape =  X_train[0].shape
    output_shape = 23

    model = keras.Sequential([
        layers.Conv2D(filters = 16, kernel_size = 3, input_shape = input_shape, activation= 'relu', padding='same'),
        layers.MaxPooling2D(pool_size=2),
        layers.Conv2D(filters = 32, kernel_size = 2, activation= 'relu', padding='same'),
        layers.MaxPooling2D(pool_size=2),
        layers.Conv2D(filters = 64, kernel_size = 2, activation= 'relu', padding='same'),
        layers.MaxPooling2D(pool_size=2),
        layers.Conv2D(filters = 128, kernel_size = 2, activation= 'relu', padding='same'),
        layers.MaxPooling2D(pool_size=2),
        layers.Flatten(),
        layers.Dropout(0.2),
        layers.Dense(500, activation = 'relu'),
        # layers.Dropout(0.2),
        layers.Dense(150, activation = 'relu'),
        # layers.Dropout(0.2),
        layers.Dense(output_shape, activation = 'softmax'),
    ])
    model.summary()
    return model
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=16,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2),
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=32,
                      kernel_size=2,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2),
        )
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=2,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2),
        )
        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=2,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(in_features=128*8*8,
                      out_features=500),
            nn.ReLU(),
            nn.Linear(in_features=500,
                      out_features=150),
            nn.ReLU(),
            nn.Linear(in_features=150,
                      out_features=output_shape),
        )

    def forward(self, x: Tensor):
        # x = self.conv_block_1(x)
        # x = self.conv_block_2(x)
        # x = self.conv_block_3(x)
        # x = self.conv_block_4(x)
        # x = self.classifier(x)
        return self.classifier(self.conv_block_4(self.conv_block_3(self.conv_block_2(self.conv_block_1(x))))) # <- leverage the benefits of operator fusion


class TinyVGG(nn.Module):
    """
    Model architecture copying TinyVGG from: 
    https://poloclub.github.io/cnn-explainer/
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2),
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*16*16*4,
                      out_features=output_shape)
        )

    def forward(self, x: Tensor):
        # x = self.conv_block_1(x)
        # x = self.conv_block_2(x)
        # x = self.classifier(x)
        # return x
        return self.classifier(self.conv_block_2(self.conv_block_1(x))) # <- leverage the benefits of operator fusion
