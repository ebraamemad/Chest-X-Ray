import mlflow
import mlflow.pytorch
import pandas as pd
from models.resnet_model import ResNetModel
from trainer.trainer import PneumoniaTrainer

if __name__ == "__main__":
    mlflow.set_experiment("chest_xray_resnet18")

    with mlflow.start_run():
        df = pd.read_csv("data\data_map.csv")

        trainer = PneumoniaTrainer(
            model_class=ResNetModel,
            data_map_df=df,
            batch_size=32,
            image_size=(224, 224),
            model_name="resnet18"
        )

        trainer.load_data()
        trainer.train(epochs=5)

        mlflow.log_param("model", "ResNet18")
        mlflow.log_param("batch_size", trainer.batch_size)
        mlflow.log_param("image_size", trainer.image_size)
        mlflow.pytorch.log_model(trainer.best_model, "model")
        
        
        
        
