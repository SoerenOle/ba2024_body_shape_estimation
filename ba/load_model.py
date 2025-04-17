import torch
from ba import FBNet
from torchvision.io import read_image
from torchvision import transforms

## 1: Load the Saved Model
# Define the device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Recreate the model architecture
model = FBNet(num_classes=6)
model.load_state_dict(torch.load("/home/g037552/Bachelorarbeit/saved_model/model_epoch_501.pth", map_location=device))  # Load weights
model.to(device)
model.eval() 

## 2: Preprocess New Images
image_path = "/home/g037552/Bachelorarbeit_Data/Blender Results/results_top-view_w-o_breasts-param/results_blender/frame_100_id_1049.png"
image = read_image(image_path).to(torch.get_default_dtype()) / 255.0

# Define and apply transformations
transform = transforms.Compose([
    transforms.Resize((108, 192)),
])

image = transform(image)  # Apply the transformations

# Move the image to the correct device
image = image.to(device)

## 3: Perform Inference
# Pass the image through the model
with torch.no_grad():  # No need to track gradients
    output = model(image.unsqueeze(0))

# Convert output tensor to a list of values
predicted_parameters = output.squeeze(0).cpu().numpy() 
print("Predicted Shape Parameters:", predicted_parameters)