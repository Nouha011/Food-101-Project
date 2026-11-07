import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0
from PIL import Image
import pandas as pd
from utils import get_calories

# 1. Rebuild the model structure for 20 classes
num_classes = 20
model = efficientnet_b0()
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

# 2. Load your trained weights (forcing CPU)
model.load_state_dict(torch.load('efficientnet_food101.pth', map_location=torch.device('cpu')))
model.eval()

# 3. Prepare the test image
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Updated to use your new file name (change to 'test_image_2.jpg' if that is what you settled on)
image_path = 'test_image_1.jpg' 

try:
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)
except FileNotFoundError:
    print(f"Error: Could not find {image_path}. Please check the exact file name and extension.")
    exit()

# 4. Make the prediction
with torch.no_grad():
    outputs = model(input_tensor)
    _, predicted_idx = torch.max(outputs, 1)

# 5. Map the prediction to the food name
df = pd.read_csv('calories.csv')
class_names = sorted(df['class'].str.lower().tolist())
predicted_food = class_names[predicted_idx.item()]

print("\n--- AI PREDICTION ---")
print(f"Detected Food: {predicted_food.capitalize()}")

# 6. Fetch the calories using your utils.py script
calories = get_calories(predicted_food, 'calories.csv')
print(f"Nutritional Info: {calories} calories")
print("---------------------\n")