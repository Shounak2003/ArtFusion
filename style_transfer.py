import torch
import torch.optim as optim
from torch import nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image

# Assuming your device is CUDA-enabled. Otherwise, use 'cpu'.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Helper function to load images
def load_image(img_path, max_size=512):
    img = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((max_size, max_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = transform(img).unsqueeze(0)
    return img.to(device)

# Load images
content_img = load_image('/root/.cache/kagglehub/datasets/anonymous132423/molp2442/versions/1/abstract/811769e22a886ffbc2ab750ab2cbfd03c.jpg')
style_img = load_image('/root/.cache/kagglehub/datasets/anonymous132423/molp2442/versions/1/abstract/e967f5f00fa249c02572546a3c3f59d2c.jpg')

# Load pre-trained VGG19 model
vgg19 = models.vgg19(pretrained=True).features.to(device).eval()

# Define the layers where content and style will be extracted
content_layers = [21]  # VGG19 layer 21 gives a good content representation
style_layers = [0, 5, 10, 19, 28]  # Multiple layers for style representation

# Function to extract features from specific layers of VGG
def get_features(image, model, layers):
    features = {}
    x = image
    for name, layer in enumerate(model):
        x = layer(x)
        if name in layers:
            features[name] = x
    return features

# Loss functions for content and style
def compute_content_loss(target, content):
    return torch.mean((target - content) ** 2)

def gram_matrix(x):
    b, c, h, w = x.size()
    features = x.view(b * c, h * w)
    gram = torch.mm(features, features.t())
    return gram.div(b * c * h * w)

def compute_style_loss(target, style):
    gram_target = gram_matrix(target)
    gram_style = gram_matrix(style)
    return torch.mean((gram_target - gram_style) ** 2)

# Initialize the content and style targets
content_features = get_features(content_img, vgg19, content_layers)
style_features = get_features(style_img, vgg19, style_layers)

# Create a target image (start with the content image)
target_img = content_img.clone().requires_grad_(True).to(device)

# Define weights for content and style loss
content_weight = 1e5
style_weight = 1e10

# Optimizer
optimizer = optim.Adam([target_img], lr=0.003)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    target_features = get_features(target_img, vgg19, content_layers + style_layers)
    
    # Compute the content loss
    content_loss = 0
    for content_layer in content_layers:
        content_loss += compute_content_loss(target_features[content_layer], content_features[content_layer])

    # Compute the style loss
    style_loss = 0
    for style_layer in style_layers:
        style_loss += compute_style_loss(target_features[style_layer], style_features[style_layer])
    
    # Total loss
    total_loss = content_weight * content_loss + style_weight * style_loss
    
    # Backward pass
    optimizer.zero_grad()
    total_loss.backward(retain_graph=True)  # Add retain_graph=True to avoid freeing the graph
    optimizer.step()
    
    # Print loss every few steps
    if epoch % 10 == 0:
        print(f"Epoch {epoch}/{num_epochs}, Total Loss: {total_loss.item():.4f}, Content Loss: {content_loss.item():.4f}, Style Loss: {style_loss.item():.4f}")

# Save the result
final_img = target_img.clone().detach().cpu().squeeze(0)
final_img = transforms.ToPILImage()(final_img)
final_img.save("output_image.jpg")
