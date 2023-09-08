from calendar import c
from PIL import Image, ImageDraw  
  
# Define the size of the image  
width, height =256, 256
# Define the size of the rectangles and the space between them  
rectangle_width = 5 
rectangle_height = 15  
space = 0  
radius = 45

bg_color = 0
fg_color = 255
  
# Create a new image with white background  
img = Image.new('RGB', (width, height), (bg_color, bg_color, bg_color))  
d = ImageDraw.Draw(img)  

# Draw the circle in the center with opposite pattern direction  
center_x, center_y = width // 2, height // 2  
  
# Draw the rectangles  
for i in range(0, width, rectangle_width + space):  
    for j in range(0, height, rectangle_height + space):  
        if (i // (rectangle_width + space)) % 2 == (j // (rectangle_height + space)) % 2:  
            d.rectangle([i, j, i + rectangle_width, j + rectangle_height], fill=(fg_color, fg_color, fg_color)) 

# Create a mask for the circle  
mask = Image.new('L', (width, height), 0)  
mask_d = ImageDraw.Draw(mask)  
mask_d.ellipse([center_x - radius, center_y - radius, center_x + radius, center_y + radius], fill=fg_color)  
  
# Apply the mask to the image  
circle = Image.new('RGB', (width, height))  
circle.paste(img, mask=mask)  
  
# Rotate the circle by 90 degrees  
circle = circle.rotate(90)  
  
# Paste the rotated circle back into the image  
img.paste(circle, (0, 0), mask=mask)  
 
# Save the image  
img.save('variant_ouchi_1.png')  
