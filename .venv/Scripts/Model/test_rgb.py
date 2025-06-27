import torch
import torchvision.transforms as transforms
from PIL import Image
import tifffile
import numpy as np
import os
import torch.nn as nn

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
        # NOU: Strat convoluțional suplimentar pentru mai multă capacitate de învățare
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
            # NOU: Includem conv4 și bn4 în calculul dimensiunii
            dummy_output = self.pool(self.relu(self.bn4(self.conv4(dummy_output))))
            self.flattened_size = dummy_output.view(1, -1).shape[1]

        # Straturi complet conectate (Fully Connected) pentru clasificare
        # NOU: Am mărit dimensiunea primului strat FC de la 512 la 1024
        self.fc1 = nn.Linear(self.flattened_size, 1024)
        self.fc2 = nn.Linear(1024, num_classes) # Trebuie să ajustăm input-ul lui fc2 dacă fc1 e modificat

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
        # NOU: Adăugăm propagarea prin conv4 și bn4
        x = self.pool(self.relu(self.bn4(self.conv4(x))))

        # Aplatizează output-ul pentru straturile complet conectate (Flatten)
        x = x.view(x.size(0), -1)

        # Aplică primul strat FC, activarea ReLU și apoi Dropout
        x = self.relu(self.fc1(x))
        x = self.dropout(x) # Aplicați dropout aici, după activarea ReLU

        # Aplică al doilea strat FC pentru a obține scorurile finale (logits)
        x = self.fc2(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SatelliteCNN(num_classes=10)
model.load_state_dict(torch.load("../Models/satellite_model_best_accuracy.pth", map_location=device))
model.to(device)
model.eval()
print("Model Loaded Successfully!")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_image(image_path, model, class_names):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        predicted_class = torch.argmax(output, dim=1).item()
    return class_names[predicted_class]

class_names = sorted(os.listdir("../EuroSAT_RGB/train"))
count_right=0
count_all=0
count_all2=[0,0,0,0,0,0,0,0,0,0]
count_right2=[0,0,0,0,0,0,0,0,0,0]
test_folder = "../EuroSAT_RGB/test"
for filename in os.listdir(test_folder):
    if filename.endswith(".jpg"):
        image_path = os.path.join(test_folder, filename)
        prediction = predict_image(image_path, model, class_names)
        text_name=filename.split("_")[0]
        count_all+=1
        if text_name =="AnnualCrop":
            count_all2[0]+=1
            if prediction == text_name:
                count_right2[0]+=1
        if text_name =="Forest":
            count_all2[1]+=1
            if prediction == text_name:
                count_right2[1]+=1
        if text_name =="HerbaceousVegetation":
            count_all2[2]+=1
            if prediction == text_name:
                count_right2[2]+=1
        if text_name =="Highway":
            count_all2[3]+=1
            if prediction == text_name:
                count_right2[3]+=1
        if text_name =="Industrial":
            count_all2[4]+=1
            if prediction == text_name:
                count_right2[4]+=1
        if text_name =="Pasture":
            count_all2[5]+=1
            if prediction == text_name:
                count_right2[5]+=1
        if text_name =="PermanentCrop":
            count_all2[6]+=1
            if prediction == text_name:
                count_right2[6]+=1
        if text_name =="Residential":
            count_all2[7]+=1
            if prediction == text_name:
                count_right2[7]+=1
        if text_name =="River":
            count_all2[8]+=1
            if prediction == text_name:
                count_right2[8]+=1
        if text_name == "SeaLake":
            count_all2[9] += 1
            if prediction == text_name:
                count_right2[9] += 1
        if prediction == text_name:
            count_right+=1
        print(f"{filename} → {prediction}")
print(count_right/count_all)
print(count_right2)
print(count_all2)
