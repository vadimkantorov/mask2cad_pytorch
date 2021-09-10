```shell
wget https://ftp.nluug.nl/pub/graphics/blender/release/Blender2.93/blender-2.93.2-linux-x64.tar.xz
tar -xf blender-2.93.2-linux-x64.tar.xz
blender=$PWD/../blender/blender-2.93.2-linux-x64/blender

bash scripts/pix3d.sh

pip install pyclustering
python preprocess_pix3d.py
$blender -noaudio --background --python vis_pix3d.py

$blender -noaudio --background --python render_pix3d.py
```

### Training
```shell
# finetune Mask-RCNN on Pix3D
python train.py --loss-weights loss_classifier=1 --loss-weights loss_box_reg=1 --loss-weights loss_mask=1 --loss-weights loss_objectness=1 --loss-weights loss_rpn_box_reg=1

# train Mask2CAD
python train.py --loss-weights shape_embedding=0.5 --loss-weights pose_classification=0.25 --loss-weights pose_regression=5.0 --loss-weights center_regression=5.0
```

### Evaluation
```shell
wget https://raw.githubusercontent.com/facebookresearch/meshrcnn/c89886f46a0f02871f3bc83a770d907f41a5624b/meshrcnn/evaluation/pix3d_evaluation.py
```
