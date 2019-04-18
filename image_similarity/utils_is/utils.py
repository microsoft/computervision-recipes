from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def plot_similars(similars, rows=2, cols=5):
    for num, (image, score) in enumerate(similars):
        img = Image.open(image)
        plt.rcParams["figure.dpi"] = 200
        plt.subplot(rows, cols, num + 1)
        title = "{} - {:0.3f}".format(image.split("/")[-1], score)
        plt.title(title, fontsize=6)
        plt.axis("off")
        plt.imshow(img)


class SaveFeatures:
    """Hook to save the features in the intermediate layers
    
    Source: https://forums.fast.ai/t/how-to-find-similar-images-based-on-final-embedding-layer/16903/13
    
    Args:
        model_layer (nn.Module): Model layer
    """
    features=None
    def __init__(self, model_layer): 
        self.hook = model_layer.register_forward_hook(self.hook_fn)
        self.features = None
    
    def hook_fn(self, module, input, output): 
        out = output.detach().cpu().numpy()
        if isinstance(self.features, type(None)):
            self.features = out
        else:
            self.features = np.row_stack((self.features, out))
    
    def remove(self): 
        self.hook.remove()
    