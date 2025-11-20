import bentoml
from bentoml.io import NumpyNdarray
import numpy as np
import torch

model = bentoml.pytorch.load_model("resnet_cifake:latest")
model.eval()

svc = bentoml.Service("resnet_cifake_service")

@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def predict(input_array: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        tensor = torch.tensor(input_array).float()
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        return probs.numpy()
