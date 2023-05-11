import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
import os
import glob
from segment_anything import sam_model_registry, SamPredictor

input_dir = "./center_views"
output_dir = "./mask"
vis_dir = "./masked_image"
image_names = sorted(glob.glob(f"{input_dir}/*.png"))
n_images = len(image_names)

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)

fig = plt.figure(figsize=(12,9))
ax = fig.add_subplot(111)

coords = []
labels = []
current_img_idx = 0
img = None
init_mask = None
cur_mask = None

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=50):
    if len(coords) == 0:
        return
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25) 

def onclick(event):
    print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          ('double' if event.dblclick else 'single', event.button,
           event.x, event.y, event.xdata, event.ydata))
    
    global ix, iy
    ix, iy = event.xdata, event.ydata
    print (f'x = {ix}, y = {iy}')

    global coords, labels
    coords.append(np.array([ix, iy]))
    if event.button == 1:
        labels.append(1)
    elif event.button == 3:
        labels.append(0)
    coords_np = np.stack(coords, 0)
    labels_np = np.stack(labels, 0)

    ax.clear()
    ax.imshow(img)
    ax.set_title(image_names[current_img_idx])
    show_points(coords_np, labels_np, ax)

    global init_mask, cur_mask
    if len(coords_np) > 0:
        if len(coords_np) == 1:
            masks, scores, logits = predictor.predict(
                point_coords=coords_np,
                point_labels=labels_np,
                multimask_output=True,
            )
            init_mask = logits[np.argmax(scores), :, :]
            cur_mask = masks[np.argmax(scores)]
            show_mask(cur_mask, plt.gca())
        else:
            masks, _, _ = predictor.predict(
                point_coords=coords_np,
                point_labels=labels_np,
                mask_input=init_mask[None, :, :],
                multimask_output=False,
            )
            cur_mask = masks[0]
            show_mask(masks, plt.gca())

    fig.canvas.draw_idle()

def on_press(event):
    global coords, labels
    global img, current_img_idx
    global init_mask, cur_mask
    sys.stdout.flush()
    if event.key == 'r':
        coords = []
        labels = []
        ax.clear()
        ax.imshow(img)
        fig.canvas.draw()
    elif event.key == 'escape' or event.key == 'q':
        exit(0)
    elif event.key == ' ':
        if cur_mask is not None:
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(vis_dir, exist_ok=True)
            filename = os.path.basename(image_names[current_img_idx])
            out = cur_mask[..., np.newaxis].astype(np.float32)
            cv2.imwrite(f"{output_dir}/{filename}", out * 255)
            cv2.imwrite(f"{vis_dir}/{filename}", img * out)
            
        current_img_idx += 1
        if current_img_idx >= n_images:
            print("All done.")
            exit(0)
        img = imread(image_names[current_img_idx])
        predictor.set_image(img)
        ax.clear()
        coords = []
        labels = []
        init_mask = None
        cur_mask = None
        ax.imshow(img)
        ax.set_title(image_names[current_img_idx])
        fig.canvas.draw()

def imread(fp):
    img = cv2.imread(fp)
    img = img / 255
    img = np.power(img, 1/2.2) * 255
    img = img.astype(np.uint8)
    return img

cid = fig.canvas.mpl_connect('button_press_event', onclick)
cid = fig.canvas.mpl_connect('key_press_event', on_press)

if __name__ == "__main__":
    img = imread(image_names[current_img_idx])
    predictor.set_image(img)
    ax.imshow(img)
    ax.set_title(image_names[current_img_idx])
    plt.show()