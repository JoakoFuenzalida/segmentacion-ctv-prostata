"""Variante de nnUNetTrainerNoDA (sin aumento de datos) con 250 epocas.

Copiar este archivo en:
    <entorno>/Lib/site-packages/nnunetv2/training/nnUNetTrainer/variants/data_augmentation/
nnU-Net encuentra el entrenador por su nombre de clase. Si se reinstala nnU-Net, hay que volver a copiarlo.
"""
import torch
from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerNoDA import nnUNetTrainerNoDA


class nnUNetTrainerNoDA_250epochs(nnUNetTrainerNoDA):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250
