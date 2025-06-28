import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split, Subset
from PIL import Image
import numpy as np


# Aceste transformări pregătesc imaginile pentru intrarea în rețeaua neurală.
# Transformări pentru antrenament (cu augmentare)
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=(0, 90)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Transformări pentru validare/test (fără augmentare, doar redimensionare și normalizare)
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Wrapper pentru aplicarea transformărilor specifice la Subset-uri
class DatasetWithTransform(torch.utils.data.Dataset):
    def __init__(self, subset: Subset, transform=None):

        self.subset = subset
        self.transform = transform
    """
           Un wrapper care permite aplicarea unui transform specific unui Subset.

           Args:
               subset (torch.utils.data.Subset): Subset-ul de date.
               transform (torchvision.transforms.Compose, optional): Transformarea de aplicat.
           """

    def __getitem__(self, index):
        # Subset-ul conține deja indexul original și referința la dataset-ul părinte
        original_idx = self.subset.indices[index]
        x, y = self.subset.dataset[original_idx]  # Obține datele din dataset-ul original

        if self.transform:
            x = self.transform(x) # Aplică transformarea specifică
        return x, y

    def __len__(self):
        return len(self.subset)


dataset_path = "../EuroSAT_RGB"
# Încărcăm setul complet de antrenament fără transformări inițial
full_train_dataset_raw = datasets.ImageFolder(
    root=os.path.join(dataset_path, "train"),
)

# Splităm setul de date de antrenament în seturi de antrenament și validare
train_size = int(0.8 * len(full_train_dataset_raw)) # 80% pentru antrenament
val_size = len(full_train_dataset_raw) - train_size # Restul pentru validare
train_subset, val_subset = random_split(full_train_dataset_raw, [train_size, val_size])

# Aplicăm transformările specifice fiecărui subset folosind wrapper-ul
train_dataset = DatasetWithTransform(train_subset, transform=train_transform)
val_dataset = DatasetWithTransform(val_subset, transform=val_transform)


# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=os.cpu_count() // 2 or 1)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=os.cpu_count() // 2 or 1)


print(f"S-au încărcat {len(full_train_dataset_raw)} imagini în setul complet de antrenament/validare!")
print(f"S-au împărțit în {len(train_dataset)} imagini de antrenament și {len(val_dataset)} imagini de validare.")
print(f"Clase detectate: {full_train_dataset_raw.classes}")



class SatelliteCNN(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.5):
        """
        Inițializează modelul SatelliteCNN cu straturi convoluționale și complet conectate,
        plus straturi Batch Normalization și un strat Dropout pentru regularizare și stabilizare.

        Args:
            num_classes (int): Numărul de clase de ieșire pentru clasificare.
            dropout_rate (float): Rata de dropout pentru stratul Dropout (între 0 și 1).
        """
        super(SatelliteCNN, self).__init__()
        # Straturi convoluționale cu activare ReLU și MaxPooling pentru extragerea de trăsături
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32) # Batch Normalization după conv1
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64) # Batch Normalization după conv2
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128) # Batch Normalization după conv3
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256) # Batch Normalization după conv4

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

        # Calculează dimensiunea așezată dinamic pentru stratul complet conectat (Fully Connected)
        # Un input fals (dummy input) pentru a determina dimensiunea după operațiile convoluționale și de pooling.
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            dummy_output = self.pool(self.relu(self.bn1(self.conv1(dummy_input))))
            dummy_output = self.pool(self.relu(self.bn2(self.conv2(dummy_output))))
            dummy_output = self.pool(self.relu(self.bn3(self.conv3(dummy_output))))
            dummy_output = self.pool(self.relu(self.bn4(self.conv4(dummy_output))))
            self.flattened_size = dummy_output.view(1, -1).shape[1]

        # Straturi complet conectate (Fully Connected) pentru clasificare
        self.fc1 = nn.Linear(self.flattened_size, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

        # Adăugare strat Dropout pentru regularizare, pentru a preveni overfitting-ul
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        Definește propagarea înainte a datelor prin rețea.

        Args:
            x (torch.Tensor): Tensorul de intrare (imaginea).

        Returns:
            torch.Tensor: Scorurile logit finale pentru fiecare clasă.
        """
        # Propagare înainte prin straturile convoluționale cu Batch Normalization și ReLU, urmate de Pooling
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.pool(self.relu(self.bn4(self.conv4(x))))

        # Aplatizează output-ul pentru straturile complet conectate (Flatten)
        x = x.view(x.size(0), -1)

        # Aplică primul strat FC, activarea ReLU și apoi Dropout
        x = self.relu(self.fc1(x))
        x = self.dropout(x) # Aplicați dropout aici, după activarea ReLU

        # Aplică al doilea strat FC pentru a obține scorurile finale (logits)
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    # Setează dispozitivul de antrenament (GPU dacă este disponibil, altfel CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Folosind dispozitivul: {device}")

    # Ne asigurăm că num_classes este corect, obținându-l din dataset-ul complet
    model = SatelliteCNN(num_classes=len(full_train_dataset_raw.classes)).to(device)

    # Definește funcția de pierdere și optimizatorul
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    # Schedulerul va reduce rata de învățare atunci când acuratețea de validare nu se îmbunătățește
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)

    num_epochs = 50

    print(f"Începerea antrenamentului pentru {num_epochs} epoci...")
    best_val_accuracy = 0.0 # Pentru a salva cel mai bun model bazat pe acuratețea de validare
    best_epoch = 0

    for epoch in range(num_epochs):
        # --- Faza de Antrenament ---
        model.train() # Setează modelul în modul de antrenament
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_accuracy = 100 * correct_train / total_train
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss Antrenament: {running_loss / len(train_loader):.4f}, Acuratețe Antrenament: {train_accuracy:.2f}%")

        # --- Faza de Validare ---
        model.eval() # Setează modelul în modul de evaluare
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad(): # Nu calculăm gradienți în timpul validării
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_accuracy = 100 * correct_val / total_val
        print(f"Pierdere Validare: {val_loss / len(val_loader):.4f}, Acuratețe Validare: {val_accuracy:.2f}%")

        # Ajustează rata de învățare pe baza acurateții de validare
        scheduler.step(val_accuracy)

        print(f"Rata de învățare curentă: {optimizer.param_groups[0]['lr']:.6f}")

        # Salvează cel mai bun model bazat pe acuratețea de validare
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch + 1
            torch.save(model.state_dict(), "satellite_model_best_accuracy.pth")
            print(f"Model salvat! Acuratețea de validare s-a îmbunătățit la {best_val_accuracy:.2f}% (Epoca {best_epoch})")

    print(f" Antrenament finalizat. Cel mai bun model (cu acuratețea de validare: {best_val_accuracy:.2f}% la Epoca {best_epoch}) a fost salvat în 'satellite_model_best_accuracy.pth'!")
