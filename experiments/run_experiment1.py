import mlflow
import torch.nn as nn
import torch.nn.functional as F
from trainer.trainer import PneumoniaTrainer

class PneumoniaCNN(nn.Module):
    def __init__(self):
        super(PneumoniaCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 37 * 37, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

if __name__ == "__main__":
    mlflow.set_experiment("Pneumonia Detection")

    with mlflow.start_run():
        mlflow.log_param("model_type", "SimpleCNN")
        mlflow.log_param("batch_size", 32)
        mlflow.log_param("lr", 0.0001)
        mlflow.log_param("image_size", (150, 150))
        mlflow.log_param("epochs", 5)

        model = PneumoniaCNN()
        trainer = PneumoniaTrainer(
            data_map_path="data/data_map.csv",
            model=model,
            image_size=(150, 150),
            batch_size=32
        )

        trainer.load_data()
        trainer.train(epochs=5, lr=0.0001, mlflow_logger=mlflow)
        mlflow.pytorch.log_model(model, "model")
        mlflow.log_artifact("cnn_model.pth")
        