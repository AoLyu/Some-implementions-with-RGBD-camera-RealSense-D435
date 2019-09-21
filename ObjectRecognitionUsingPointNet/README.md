# Obect Recognition Using PointNet
The code of PointNet is highly borrowed from https://github.com/fxia22/pointnet.pytorch
This code is an implementation of capturing point cloud on the client computer and processing it on the sever.
## Usage
You need prepare your own dataset,and then modify the `datasets.py`
To train your network
```
python train_classification.py
```
To test it on the real object,change the IP adress and port in the `server.py`,then start it.
```
python server.py --model yourmodel_path
```
then start the client and your realsense D435.
```
python client.py
```
this code detect the plane(green) by chessboard,so you need put chessboard(6x9)  on the table plane first,the table plane will turn into green from its origin color.


## Screenshotsr
<img src="./doc/screencut.gif" height="350" width="" >

## Platform
Ubuntu 16.04


