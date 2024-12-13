import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.ops import RoIPool

class RCNN(nn.Module):
    def __init__(self, resnet_model, num_classes=2):
        super(RCNN, self).__init__()

        # Utiliser le ResNet50 pour l'extraction de caractéristiques (sans la dernière couche FC)
        self.resnet = nn.Sequential(*list(resnet_model.children())[:-2])  # On enlève la couche FC et le pooling final

        # RoI Pooling : taille de sortie fixée à (7, 7)
        self.roi_pool = RoIPool(output_size=(7, 7), spatial_scale=1.0 / 16)  # ResNet réduit l'image par un facteur de 16

        # Classifieur : prend les caractéristiques de taille (7, 7, 2048) et prédit les classes
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes)
        )

    def forward(self, images, rois):
        """
        images : Tensor de forme [batch_size, 3, H, W]
        rois : Tensor de forme [num_rois, 5] avec [batch_index, x1, y1, x2, y2]
        """
        # Extraire les caractéristiques avec ResNet
        features = self.resnet(images)  # [batch_size, 2048, H/16, W/16]

        # Appliquer le RoI Pooling sur les régions d'intérêt
        pooled_features = self.roi_pool(features, rois)  # [num_rois, 2048, 7, 7]

        # Classifier chaque région
        outputs = self.classifier(pooled_features)  # [num_rois, num_classes]

        return outputs

def create_model(num_classes=2):
    resnet = models.resnet50(pretrained=True)
    return RCNN(resnet, num_classes)

# Fonction main pour tester le modèle
def main():
    # Créer une instance du modèle
    model = create_model(num_classes=2)
    model.eval()

    # Exemple de données : une image fictive et des RoIs
    images = torch.randn(1, 3, 224, 224)  # Batch de 1 image de taille 224x224
    rois = torch.tensor([[0, 30, 30, 180, 180]], dtype=torch.float)  # Un RoI

    # Effectuer une prédiction
    with torch.no_grad():
        outputs = model(images, rois)
        print(f"Prédictions : {outputs}")

if __name__ == "__main__":
    main()