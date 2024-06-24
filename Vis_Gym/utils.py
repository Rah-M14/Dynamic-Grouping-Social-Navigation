
import matplotlib.pyplot as plt

def display_image(title, image, cmap=None, axis_switch = 'off'):
    plt.figure(figsize=(6, 6))
    plt.title(title)
    plt.imshow(image, cmap=cmap)
    plt.axis(axis_switch)
    plt.show()