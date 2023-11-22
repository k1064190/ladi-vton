import os
import cv2

img_input = 'dataset/replace/'
orig_height = 1024
orig_width = 768
orig_ratio = 768/1024

imgs = os.listdir(img_input)

# target to the original size
for img_path in imgs:
    img = cv2.imread(os.path.join(img_input, img_path))
    height, width, _ = img.shape
    if height == orig_height and width == orig_width:
        continue
    ratio = width/height
    if ratio > orig_ratio:  # width is larger than original
        new_height = height
        new_width = int(height * orig_ratio)
    else:
        new_width = width
        new_height = int(width / orig_ratio)
    # crop the image
    x_start = int((width - new_width) / 2)
    y_start = int((height - new_height) / 2)
    crop_img = img[y_start:height-y_start, x_start:width-x_start, :]
    # resize the image
    img = cv2.resize(crop_img, (orig_width, orig_height))

    cv2.imwrite(os.path.join(img_input, img_path), img)
