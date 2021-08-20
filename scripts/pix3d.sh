mkdir -p data/common/pix3d
wget -nc -P data/common http://pix3d.csail.mit.edu/data/pix3d.zip
unzip data/common/pix3d.zip -d data/common/pix3d -q
rm data/common/pix3d.zip

mkdir -p data/common/pix3d_splits
# https://github.com/chengzhag/Implicit3DUnderstanding/tree/main/data/pix3d/splits
wget -P data/common/pix3d_splits $(printf "https://github.com/chengzhag/Implicit3DUnderstanding/raw/main/data/pix3d/splits/%s " train.json test.json)
# https://github.com/facebookresearch/meshrcnn/tree/master/datasets/pix3d
wget -P data/common/pix3d_splits $(printf "https://dl.fbaipublicfiles.com/meshrcnn/pix3d/%s " pix3d_s1_train.json pix3d_s1_test.json pix3d_s2_train.json pix3d_s2_test.json)
