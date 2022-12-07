import torch
import torch.nn as nn
from torchvision import datasets, models
import torchvision.transforms as T
from PIL import Image
import io
from helpers import draw_box, url_to_img, img_to_bytes


device = torch.device('cpu')


class PythonPredictor:

    def __init__(self, config):
        self.model = torch.load('resnet50_framework.pt', map_location = device)
        num_ftrs = self.model.fc.in_features
        self.model.fc = torch.nn.Sequential(torch.nn.Linear(num_ftrs,101))
        # checkpoint = torch.load("app/model_final.pt", map_location = device)
        checkpoint = torch.load("model_final.pt", map_location = device)

        self.model.load_state_dict(checkpoint['model_state_dict'])

    def predict(self, payload):

        img = url_to_img(payload["url"])
        output = self.model(img)
        _, preds = torch.max(output.data, dim=1)

        return preds

    



# # Transfer Learning
# #model = models.resnet50(pretrained=False)
# model = torch.load('resnet50_framework.pt', map_location = device)
# num_ftrs = model.fc.in_features
# model.fc = torch.nn.Sequential(torch.nn.Linear(num_ftrs,101))

# # checkpoint = torch.load("app/model_final.pt", map_location = device)
# checkpoint = torch.load("model_final.pt", map_location = device)

# model.load_state_dict(checkpoint['model_state_dict'])

# model.eval()

# def transform_image(image_bytes):
#     transform = T.transforms.Compose([
# #         T.transforms.ColorJitter(brightness=0.1,contrast=0.1,saturation=0.1),
#         T.transforms.RandomAffine(15),
#         T.transforms.RandomHorizontalFlip(),
#         T.transforms.RandomRotation(15),
#         T.transforms.Resize((224,224)),
#         T.transforms.ToTensor(),
#         T.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
#     ])

#     image  = Image.open(io.BytesIO(image_bytes))

#     return transform(image).unsqueeze(0)


