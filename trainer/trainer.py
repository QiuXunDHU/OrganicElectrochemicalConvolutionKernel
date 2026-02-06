import torch
import torch.nn as nn
import torch.optim as optim
from config import DEVICE


class AdvancedTrainer:
    def __init__(self, model, device=DEVICE, logger=None):
        self.model = model.to(device)
        self.device = device
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'lr': []
        }
        self.logger = logger

    def train(self, train_loader, val_loader, backbone, kernel_name, epochs=100):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=2e-4, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5)

        best_acc = 0.0
        early_stop_counter = 0

        for epoch in range(epochs):
            self.model.train()
            train_loss, correct, total = 0.0, 0, 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            val_loss, val_acc = self.validate(val_loader, criterion)
            current_lr = optimizer.param_groups[0]['lr']

            self.history['train_loss'].append(train_loss / len(train_loader))
            self.history['train_acc'].append(100. * correct / total)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)

            scheduler.step(val_acc)

            if val_acc > best_acc:
                model_name = f"best_{backbone}_{kernel_name}.pth"  # 使用规范名称
                model_path = self.logger.base_dir / 'models' / model_name
                torch.save(self.model.state_dict(), model_path)
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            if early_stop_counter >= 10:
                print(f"Early stopping at epoch {epoch + 1}")
                break

            if self.logger:
                self.logger.log_metrics(
                    backbone=backbone,
                    kernel_name=kernel_name,  # 传递名称
                    epoch=epoch + 1,
                    train_loss=train_loss / len(train_loader),
                    val_loss=val_loss,
                    train_acc=100. * correct / total,
                    val_acc=val_acc
                )

            print(f"Epoch {epoch + 1}/{epochs}: "
                  f"Train Loss: {self.history['train_loss'][-1]:.4f} "
                  f"Train Acc: {self.history['train_acc'][-1]:.2f}% | "
                  f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.2f}% | "
                  f"LR: {current_lr:.2e}")

    def validate(self, loader, criterion):
        self.model.eval()
        loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss += criterion(outputs, labels).item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        return loss / len(loader), 100. * correct / total
