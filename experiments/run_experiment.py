import mlflow
import mlflow.pytorch
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.resnet_model import ResNetModel
from trainer.trainer import PneumoniaTrainer

if __name__ == "__main__":
    mlflow.set_experiment("chest_xray_use_resnet18_experiment")

    with mlflow.start_run():

        trainer = PneumoniaTrainer(
            model=ResNetModel,
            data_map_path=r"data\data_map.csv",
            batch_size=32,
            image_size=(224, 224),
            experiment_name="resnet18"
        )

        trainer.load_data()
        trainer.train(epochs=2)

        mlflow.log_param("model", "ResNet18")
        mlflow.log_param("batch_size", trainer.batch_size)
        mlflow.log_param("image_size", trainer.image_size)
        mlflow.pytorch.log_model(trainer.best_model, "model")
        
        
        
        
