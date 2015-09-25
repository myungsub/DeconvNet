LOGDIR=./training_log
CAFFE=./caffe/build/tools/caffe
SOLVER=./solver.prototxt
WEIGHTS=./model/VGG_conv/VGG_ILSVRC_16_layers_conv.caffemodel
<<<<<<< HEAD
#WEIGHTS=./snapshot/stage_1_train_iter_1500_from_scratch.caffemodel
=======
>>>>>>> origin/master

GLOG_log_dir=$LOGDIR $CAFFE train -solver $SOLVER -weights $WEIGHTS -gpu 0

#send_notify_mail "039_voc_single_object_finetune_from_031 train script is finished"
