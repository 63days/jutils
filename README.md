# A personal package for deep learning.
## Install
```shell
conda install -c conda-forge fresnel igl
pip install mitsuba # optional for mitsuba rendering
pip install -e .
```

## Mesh & Point Cloud rendering
```
from jutils import visutil, imageutil

verts # np.ndarray of [V,3]
faces # np.ndaaray of [F,3]
pc # np.ndarray of [N,3]

mesh_img = visutil.render_mesh(verts, faces)
pc_img = visutil.render_mesh(pc)

img = imageutil.merge_images([mesh_img, pc_img]) # horizontally concatenate images.
img.save("render.png")
```

## Image Concatenation
```
from jutils import imageutil

img_grid = [[img1, img2],
				    [img3, img4]]

merged_img = imageutil.merge_images(img_grid)
merged_img.save("merged.png")
```
