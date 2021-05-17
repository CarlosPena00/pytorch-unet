#!/bin/bash

# parameters
DATA=/home/chcp/Datasets
#NAMEDATASET='U2OS_1_0_1'
#NAMEDATASET='Seg1009_0.3.1'
NAMEDATASET='FluoC2DLMSC_0.1.1'
#NAMEDATASET='Kaggle2018_1_0_0'

PROJECT='../out/Fluo'
EPOCHS=500
BATCHSIZETRAIN=1
BATCHSIZETEST=1
LEARNING_RATE=0.00001
MOMENTUM=0.5
PRINT_FREQ=100
WORKERS=0
RESUME='model_best.pth.tar' #model_best, chk000000
GPU=2

#ARCH='albunet'
#ARCH='unetvgg16'
#ARCH='segnet'
#ARCH='unetpad'
ARCH='unetpad'
#ARCH='unetresnet101'

#POST_METHOD="th"
POST_METHOD="map"
#POST_METHOD="wts"


LOSS='mce'
WMAP=''
OPT='adam'
SCHEDULER='fixed'
SNAPSHOT=20 #20 #5
COUNTTRAIN=500 #1000
COUNTTEST=14 #10 #7
IMAGECROP=1010
IMAGESIZE=1010 #256 #64
IMAGEPAD=0
NUMCHANNELS=3
NUMCLASSES=4

EXP_NAME='baseline_'$ARCH'_'$LOSS'_'$WMAP'_'$OPT'_'$NAMEDATASET'_'$POST_METHOD'_0011'

# rm -rf $PROJECT/$EXP_NAME/$EXP_NAME.log
# rm -rf $PROJECT/$EXP_NAME/
# mkdir $PROJECT
# mkdir $PROJECT/$EXP_NAME


## execute
python ../ISBI_train.py \
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
--load-segs=0 \
--finetuning \
2>&1 | tee -a $PROJECT/$EXP_NAME/$EXP_NAME.log \


#--parallel \
