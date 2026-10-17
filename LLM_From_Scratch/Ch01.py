import torch

# print(torch.__version__)
from importlib.metadata import version

print("Pytorch Version: ", version("torch"))
print("TikToken Version: ", version("tiktoken"))