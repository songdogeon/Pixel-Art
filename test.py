from PIL import Image
import imageio
import os

path = [f'C:/Users/SONG/Desktop/nft/monday sickness/Disc Brake/{i}' for i in os.listdir('C:/Users/SONG/Desktop/nft/monday sickness/Disc Brake')]
imgs = [Image.open(i) for i in path]
imageio.mimsave('C:/coding/pixel_is_image', imgs, fps = 2)