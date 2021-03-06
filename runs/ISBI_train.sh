#!/bin/bash

# parameters
DATA=/home/chcp/Datasets
#NAMEDATASET='U2OS_1_0_1'
NAMEDATASET='Bfhsc_1.0.0'
#NAMEDATASET='Kaggle2018_1_0_0'

PROJECT='../out/ISBI'
EPOCHS=100
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
COUNTTRAIN=1000 #1000
COUNTTEST=7 #10 #7
IMAGECROP=1010
IMAGESIZE=1010 #256 #64
NUMCHANNELS=3
NUMCLASSES=4

EXP_NAME='baseline_ISBI_'$ARCH'_'$LOSS'_'$WMAP'_'$OPT'_'$NAMEDATASET'_'$POST_METHOD'_0000'

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
--finetuning \
2>&1 | tee -a $PROJECT/$EXP_NAME/$EXP_NAME.log \


#--parallel \
