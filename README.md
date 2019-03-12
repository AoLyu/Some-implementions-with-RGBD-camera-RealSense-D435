# Save_pointcloud_with_realsenseD435
This script is written accoding to the issue [Convert Realsense poincloud in Open3D pointcloud](https://github.com/IntelVCL/Open3D/issues/473)

## Install
[Open3D](https://github.com/IntelVCL/Open3D) is required.

```python
pip install open3d-python
```
## Usage:
```python
python capt_pt.py
```

press ‘s’ on the image(depth or color) window，the current rgb-d images and pointcloud will be saved.

press 'q' ,the pipeline will be stopped.

![result](doc/result1.png)
![result](doc/result2.png)

 