from PIL import Image
import matplotlib.pyplot as plt


def plot_similars(similars, rows=2, cols=5):

    for num, (image, score) in enumerate(similars):
        img = Image.open(image)
        plt.rcParams["figure.dpi"] = 200
        plt.subplot(rows, cols, num + 1)
        title = "{} - {:0.3f}".format(image.split("/")[-1], score)
        plt.title(title, fontsize=6)
        plt.axis("off")
        plt.imshow(img)
