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

### Evaluation
```shell
wget https://raw.githubusercontent.com/facebookresearch/meshrcnn/c89886f46a0f02871f3bc83a770d907f41a5624b/meshrcnn/evaluation/pix3d_evaluation.py
```
