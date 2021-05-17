#!/bin/bash

# parameters
DATA=/home/chcp/Datasets
#NAMEDATASET='U2OS_1_0_1'
NAMEDATASET='Seg1009_0.3.1'
#NAMEDATASET='Kaggle2018_1_0_0'

PROJECT='../out/SEG1009'
EPOCHS=500
BATCHSIZETRAIN=1
BATCHSIZETEST=1
LEARNING_RATE=0.00001
MOMENTUM=0.5
PRINT_FREQ=100
WORKERS=0
RESUME='model_best.pth.tar' #model_best, chk000000
GPU=0
#ARCH='albunet'
#ARCH='unetvgg16'
#ARCH='segnet'
ARCH='unetpad'
#ARCH='unetresnet101'

#POST_METHOD="th"
POST_METHOD="map"
#POST_METHOD="wts"


LOSS='jreg'
WMAP=''
OPT='adam'
SCHEDULER='fixed'
SNAPSHOT=20 #20 #5
COUNTTRAIN=500 #1000
COUNTTEST=32 #10 #7
IMAGECROP=1010
IMAGESIZE=1010 #256 #64
IMAGEPAD=0
NUMCHANNELS=3
NUMCLASSES=2
NUMSEGS=60
LOAD_SEGS=1
CASCADE='ransac'
USE_ORI=1

EXP_NAME='Segments_1009_60_unetpad_jreg__adam_Seg1009_0.3.2_map_ransac_0000_7'
NAMEDATASET='Seg33_1.0.3'
# rm -rf $PROJECT/$EXP_NAME/$EXP_NAME.log
# rm -rf $PROJECT/$EXP_NAME/
# mkdir $PROJECT
# mkdir $PROJECT/$EXP_NAME


## execute
python ../ISBI_eval.py \
$DATA/$NAMEDATASET \
--project=$PROJECT \
--name=$EXP_NAME \
--epochs=$EPOCHS \
--batch-size-train=$BATCHSIZETRAIN \
--batch-size-test=$BATCHSIZETEST \
--count-train=$COUNTTRAIN \
--count-test=$COUNTTEST \
--num-classes=$NUMCLASSES \
--num-channels=$NUMCHANNELS \
--learning-rate=$LEARNING_RATE \
--momentum=$MOMENTUM \
--print-freq=$PRINT_FREQ \
--workers=$WORKERS \
--resume=$RESUME \
--gpu=$GPU \
--loss=$LOSS \
--opt=$OPT \
--snapshot=$SNAPSHOT \
--scheduler=$SCHEDULER \
--arch=$ARCH \
--image-size=$IMAGESIZE \
--image-crop=$IMAGECROP \
--post-method=$POST_METHOD \
--weight=$WMAP \
--pad=$IMAGEPAD \
--numsegs=$NUMSEGS \
--cascade=$CASCADE \
--load-segments=$LOAD_SEGS \
--use-ori=$USE_ORI \
--finetuning \
2>&1 | tee -a $PROJECT/$EXP_NAME/$EXP_NAME.log \


#--parallel \
