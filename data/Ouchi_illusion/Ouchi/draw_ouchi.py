import math
from PIL import Image, ImageDraw  
import numpy as np
# Define the size of the image  
width, height =256, 256
# Define the size of the rectangles and the space between them  
rectangle_width = 5 
rectangle_height = 15  
radius = 45

bg_color = 40
fg_color = 200
  
# Create a new image with white background  
img = Image.new('RGB', (width, height), (bg_color, bg_color, bg_color))  
d = ImageDraw.Draw(img)  

# Draw the circle in the center with opposite pattern direction  
center_x, center_y = width // 2, height // 2  
  
# Draw the rectangles  
for i in range(0, width, rectangle_width):  
    for j in range(0, height, rectangle_height):  
        if (i // rectangle_width + j // rectangle_height) % 2 == 0:
            d.rectangle([i, j, i + rectangle_width-1, j + rectangle_height-1], fill=(fg_color, fg_color, fg_color)) 

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
# convert numpy array to pil image
img = Image.fromarray(img)
img.save('variant_ouchi_1.png')
# Save the image  
# img.save('variant_ouchi_2.png')  
