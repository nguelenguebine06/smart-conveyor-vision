from ultralytics.data.utils import check_det_dataset

data = check_det_dataset("dataset/data.yaml")

print("Dataset chargé avec succès !")
print("Classes :", data["names"])
print("Images d'entraînement :", data["train"])
print("Images de validation :", data["val"])