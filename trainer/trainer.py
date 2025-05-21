# trainer/trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score
import mlflow
import mlflow.pytorch
from tqdm import tqdm

from trainer.dataset import ChestXrayDataset
from trainer.transforms import get_train_transforms, get_test_transforms
import pandas as pd
import numpy as np


class PneumoniaTrainer:
    def __init__(self, model, data_map_path, image_size=(224, 224), batch_size=32, lr=1e-4, experiment_name="Pneumonia Detection"):
        self.model = model
        self.data_map_path = data_map_path
        self.image_size = image_size
        self.batch_size = batch_size
        self.lr = lr
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.experiment_name = experiment_name

        self.train_transform = get_train_transforms(self.image_size)
        self.test_transform = get_test_transforms(self.image_size)

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.class_weights = None

    def load_data(self):
        df = pd.read_csv(self.data_map_path)
        train_df = df[df['split'] == 'train']
        val_df = df[df['split'] == 'val']
        test_df = df[df['split'] == 'test']

        train_dataset = ChestXrayDataset(train_df, transform=self.train_transform)
        val_dataset = ChestXrayDataset(val_df, transform=self.test_transform)
        test_dataset = ChestXrayDataset(test_df, transform=self.test_transform)

        labels = train_df['clas'].values
        class_counts = np.bincount([1 if label == 'PNEUMONIA' else 0 for label in labels])
        class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
        sample_weights = class_weights[[1 if label == 'PNEUMONIA' else 0 for label in labels]]

        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=sampler, num_workers=4)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

        self.class_weights = torch.tensor([class_counts[0]/class_counts.sum(), class_counts[1]/class_counts.sum()], device=self.device)

    def evaluate(self, model, dataloader, criterion):
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                labels = labels.float().unsqueeze(1).to(self.device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                preds = torch.round(outputs)
                running_corrects += torch.sum(preds == labels.data)

                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(outputs.cpu().numpy())

        loss = running_loss / len(dataloader.dataset)
        acc = running_corrects.double() / len(dataloader.dataset)
        auc = roc_auc_score(all_labels, all_probs)
        return loss, acc, auc

    def train(self, epochs=2):
        mlflow.set_experiment(self.experiment_name)
        if mlflow.active_run():
            mlflow.end_run()
        with mlflow.start_run():
            
            model = self.model.to(self.device)
            criterion = nn.BCELoss(weight=self.class_weights[1])
            optimizer = optim.Adam(model.parameters(), lr=self.lr)

            best_val_auc = 0
            best_model_path = "best_model.pth"

            for epoch in range(epochs):
                model.train()
                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                    inputs = inputs.to(self.device)
                    labels = labels.float().unsqueeze(1).to(self.device)

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    preds = torch.round(outputs)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(self.train_loader.dataset)
                epoch_acc = running_corrects.double() / len(self.train_loader.dataset)
                val_loss, val_acc, val_auc = self.evaluate(model, self.val_loader, criterion)

                print(f"Epoch {epoch+1}: Train Loss={epoch_loss:.4f}, Acc={epoch_acc:.4f} | Val Loss={val_loss:.4f}, Acc={val_acc:.4f}, AUC={val_auc:.4f}")

                mlflow.log_metrics({
                    "train_loss": epoch_loss,
                    "train_acc": epoch_acc.item(),
                    "val_loss": val_loss,
                    "val_acc": val_acc.item(),
                    "val_auc": val_auc
                }, step=epoch)

                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    torch.save(model.state_dict(), best_model_path)

            # Log best model
            mlflow.pytorch.log_model(model, "model")

            # Evaluate on test set
            model.load_state_dict(torch.load(best_model_path))
            test_loss, test_acc, test_auc = self.evaluate(model, self.test_loader, criterion)
            print(f"Test Results — Loss: {test_loss:.4f} | Acc: {test_acc:.4f} | AUC: {test_auc:.4f}")
            mlflow.log_metrics({
                "test_loss": test_loss,
                "test_acc": test_acc.item(),
                "test_auc": test_auc
            })
            mlflow.log_artifact(best_model_path)
            mlflow.log_artifact(self.data_map_path)
