
def run():
    import numpy as np
    print("NumPy imported.")
    import matplotlib.pyplot as plt
    print("PyPlot imported.")
    import scipy.ndimage as nd
    print("SciPy imported.")
    from sklearn import metrics
    print("Sklearn imported.")
    import pandas as pd
    print("Pandas imported.")
    import SimpleITK as sitk
    print("SimpleITK imported.")
    import torch as tc
    print("PyTorch imported.")
    import torchvision as tv
    print("Torchvision imported.")
    import lightning
    print("Lightning imported.")
    from monai import metrics
    print("MONAI imported.")
    import torchio as tio
    print("TorchIO imported.")
    from torchmetrics import F1Score
    print("TorchMetrics imported.")
    import torchsummary
    print("TorchSummary imported.")

    print("All imports completed successfully.")

if __name__ == "__main__":
    run()