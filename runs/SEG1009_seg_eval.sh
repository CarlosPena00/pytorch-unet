#!/bin/bash

# parameters
DATA=/home/chcp/Datasets

PROJECT='../out/SEG1009'
EPOCHS=5
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
COUNTTEST=500 #10 #7
IMAGECROP=1010
IMAGESIZE=1010 #256 #64
IMAGEPAD=0
NUMCHANNELS=3
NUMCLASSES=2
NUMSEGS=60
COUNT_SEGS=60
LOADSEGS=1

#NAMEDATASET='U2OS_1_0_1'
NAMEDATASET='Seg33_1.0.4'
#NAMEDATASET='Seg1009_0.3.2'
#NAMEDATASET='Kaggle2018_1_0_0'

#CASCADE='none'
CASCADE='ransac'
#CASCADE='ransac2'
#CASCADE='simple'

SEGSPERFORWARD=3
USEORI=1

EXP_NAME='Segments_Seg1009_0.3.2_'$ARCH'_'$LOSS'_'$WMAP'_'$OPT'_'$POST_METHOD'_'$CASCADE'_'$LOADSEGS'_'$SEGSPERFORWARD'_'$USEORI

## execute
python ../ISBI_train.py \
$DATA/$NAMEDATASET --project=$PROJECT --name=$EXP_NAME --epochs=$EPOCHS --batch-size-train=$BATCHSIZETRAIN \
--batch-size-test=$BATCHSIZETEST --count-train=$COUNTTRAIN --count-test=$COUNTTEST --num-classes=$NUMCLASSES \
--num-channels=$NUMCHANNELS --learning-rate=$LEARNING_RATE --momentum=$MOMENTUM --print-freq=$PRINT_FREQ \
--workers=$WORKERS --resume=$RESUME --gpu=$GPU --loss=$LOSS --opt=$OPT --snapshot=$SNAPSHOT \
--scheduler=$SCHEDULER --arch=$ARCH --image-size=$IMAGESIZE --image-crop=$IMAGECROP --post-method=$POST_METHOD \
--weight=$WMAP --pad=$IMAGEPAD --cascade=$CASCADE --load-segs=$LOADSEGS --segs-per-forward=$SEGSPERFORWARD \
--use-ori=$USEORI --just-eval=1 --finetuning \
2>&1 | tee -a $PROJECT/$EXP_NAME/$EXP_NAME.log \


SEGSPERFORWARD=3
USEORI=0

EXP_NAME='Segments_Seg1009_0.3.2_'$ARCH'_'$LOSS'_'$WMAP'_'$OPT'_'$POST_METHOD'_'$CASCADE'_'$LOADSEGS'_'$SEGSPERFORWARD'_'$USEORI

## execute
python ../ISBI_train.py \
$DATA/$NAMEDATASET --project=$PROJECT --name=$EXP_NAME --epochs=$EPOCHS --batch-size-train=$BATCHSIZETRAIN \
--batch-size-test=$BATCHSIZETEST --count-train=$COUNTTRAIN --count-test=$COUNTTEST --num-classes=$NUMCLASSES \
--num-channels=$NUMCHANNELS --learning-rate=$LEARNING_RATE --momentum=$MOMENTUM --print-freq=$PRINT_FREQ \
--workers=$WORKERS --resume=$RESUME --gpu=$GPU --loss=$LOSS --opt=$OPT --snapshot=$SNAPSHOT \
--scheduler=$SCHEDULER --arch=$ARCH --image-size=$IMAGESIZE --image-crop=$IMAGECROP --post-method=$POST_METHOD \
--weight=$WMAP --pad=$IMAGEPAD --cascade=$CASCADE --load-segs=$LOADSEGS --segs-per-forward=$SEGSPERFORWARD \
--use-ori=$USEORI --just-eval=1 --finetuning \
2>&1 | tee -a $PROJECT/$EXP_NAME/$EXP_NAME.log \


SEGSPERFORWARD=7
USEORI=1

EXP_NAME='Segments_Seg1009_0.3.2_'$ARCH'_'$LOSS'_'$WMAP'_'$OPT'_'$POST_METHOD'_'$CASCADE'_'$LOADSEGS'_'$SEGSPERFORWARD'_'$USEORI

## execute
python ../ISBI_train.py \
$DATA/$NAMEDATASET --project=$PROJECT --name=$EXP_NAME --epochs=$EPOCHS --batch-size-train=$BATCHSIZETRAIN \
--batch-size-test=$BATCHSIZETEST --count-train=$COUNTTRAIN --count-test=$COUNTTEST --num-classes=$NUMCLASSES \
--num-channels=$NUMCHANNELS --learning-rate=$LEARNING_RATE --momentum=$MOMENTUM --print-freq=$PRINT_FREQ \
--workers=$WORKERS --resume=$RESUME --gpu=$GPU --loss=$LOSS --opt=$OPT --snapshot=$SNAPSHOT \
--scheduler=$SCHEDULER --arch=$ARCH --image-size=$IMAGESIZE --image-crop=$IMAGECROP --post-method=$POST_METHOD \
--weight=$WMAP --pad=$IMAGEPAD --cascade=$CASCADE --load-segs=$LOADSEGS --segs-per-forward=$SEGSPERFORWARD \
--use-ori=$USEORI --just-eval=1 --finetuning \
2>&1 | tee -a $PROJECT/$EXP_NAME/$EXP_NAME.log \
