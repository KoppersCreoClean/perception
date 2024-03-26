from PIL import Image, ImageDraw
import numpy as np
import json

# Load the image
image = Image.open('./data/test1.jpg')

# Load the JSON annotation
with open('./data/test1.json') as f:
    annotation = json.load(f)

print(annotation)

# Create a blank mask image with the same size as the original image
mask = Image.new('L', image.size, 0)

# Draw the annotation as a mask on the blank image
draw = ImageDraw.Draw(mask)
for shape in annotation:
    points = shape['points']
    polygon = [(int(x), int(y)) for x, y in points]
    draw.polygon(polygon, fill=255)

# Apply the mask to the original image
masked_image = Image.composite(image, Image.new('RGB', image.size), mask)

# Display the masked image
masked_image.show()