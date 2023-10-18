import math
from PIL import Image, ImageDraw  
import numpy as np
from pathlib import Path
import imageio

output_path = Path('/home/siyuan/research/SpatialTransformer/data/imgs/Ouchi')
# Define the size of the image  
width, height =300, 300
# Define the size of the rectangles and the space between them  
rectangle_width = 4 
rectangle_height = 32  
radius = 36

bg_color = 100
fg_color = 140
output_file_name = f'small_ouchi_{bg_color}_{fg_color}.png'
  
# Create a new image with white background  
img = Image.new('RGB', (width, height), (bg_color, bg_color, bg_color))  
d = ImageDraw.Draw(img)  

# Draw the circle in the center with opposite pattern direction  
center_x, center_y = width // 2, height // 2  
  
# Draw the rectangles  
for i in range(0, width, rectangle_height):  
    for j in range(0, height, rectangle_width):  
        if (i // rectangle_height + j // rectangle_width) % 2 == 0:
            d.rectangle([i, j, i + rectangle_height-1, j + rectangle_width-1], fill=(fg_color, fg_color, fg_color)) 

# Create a mask for the circle  
mask = Image.new('L', (width, height), 0)  
mask_d = ImageDraw.Draw(mask)  
mask_d.ellipse([center_x - radius, center_y - radius, center_x + radius, center_y + radius], fill='white')  

# Apply the mask to the image  
circle = Image.new('RGB', (width, height))  
circle.paste(img, mask=mask)  
  
# Rotate the circle by 90 degrees  
circle = circle.rotate(90)  
  
# Paste the rotated circle back into the image  
img.paste(circle, (0, 0), mask=mask)  

# convert pil image to numpy array
img = np.array(img)
# get the location where the pixel is 0
loc = np.where(img == 0)
# set the pixel to gb_color
img[loc] = bg_color
# save image as 0-255
img = img.astype(np.uint8)
# save image
imageio.imwrite(output_path/output_file_name, img)
