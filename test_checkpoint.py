import sys
import os
import time
import PIL
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
from io import BytesIO
from PIL import Image
from PIL import ImageFile
import editdistance


from sar_model import SARModel
from data_provider import data_generator
from data_provider import lmdb_data_generator
from data_provider.data_utils import get_vocabulary
from utils.transcription_utils import idx2label, calc_metrics
from config import get_args
from train import get_data, get_batch_data


def main_test(args):
    voc, char2id, id2char = get_vocabulary(voc_type=args.test_voc_type)

    # Build graph. I did not change the name of the placeholders from train to test to not disrupt other parts of the code.
    input_test_images = tf.placeholder(dtype=tf.float32, shape=[args.test_batch_size, args.height, args.width, 3], name="input_train_images")
    input_test_images_width = tf.placeholder(dtype=tf.float32, shape=[args.test_batch_size], name="input_train_width")
    input_test_labels = tf.placeholder(dtype=tf.int32, shape=[args.test_batch_size, args.max_len], name="input_train_labels")
    input_test_labels_mask = tf.placeholder(dtype=tf.int32, shape=[args.test_batch_size, args.max_len], name="input_train_labels_mask")

    voc_size = args.max_voc_len
    #voc_size = len(voc)
    
    sar_model = SARModel(num_classes=voc_size,
                        encoder_dim=args.encoder_sdim,
                        encoder_layer=args.encoder_layers,
                        decoder_dim=args.decoder_sdim,
                        decoder_layer=args.decoder_layers,
                        decoder_embed_dim=args.decoder_edim,
                        seq_len=args.max_len,
                        is_training=False,
                        trainable_backbone = False)
    test_model_infer, test_attention_weights, test_pred = sar_model(input_test_images, input_test_labels,
                                                                       input_test_images_width,
                                                                       reuse=False)
    test_loss = sar_model.loss(test_model_infer, input_test_labels, input_test_labels_mask)


    test_data_list = get_data(args.test_data_dir,
                         args.test_data_gt,
                         args.voc_type,
                         args.max_len,
                         args.num_train,
                         args.height,
                         args.width,
                         args.val_batch_size,
                         args.workers,
                         args.keep_ratio,
                         with_aug=False)

    global_step = tf.get_variable(name='global_step', initializer=tf.constant(0), trainable=False)

    tot_edit_dist = 0
    
    tf.summary.scalar(name='val_loss', tensor=test_loss)
    merge_summary_op = tf.summary.merge_all()
    
    config = tf.ConfigProto() 
    with tf.Session(config=config) as sess:
        print('Loading model from {:s}'.format(args.test_pretrained))
        ckpt_state = tf.train.get_checkpoint_state(args.test_pretrained)
        restore_model_path = os.path.join(args.test_pretrained, os.path.basename(ckpt_state.model_checkpoint_path))
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        saver.restore(sess=sess, save_path=restore_model_path)
        for i in range(args.test_batches):    
          test_data = get_batch_data(test_data_list, args.val_batch_size)
          merge_summary_value, test_loss_value, test_pred_value  = sess.run([merge_summary_op, test_loss, test_pred], feed_dict={
                                                                                                 input_test_images: test_data[0],
                                                                                                 input_test_labels: test_data[1],
                                                                                                 input_test_labels_mask: test_data[2],
                                                                                                 input_test_images_width: test_data[4]})
          batch_edit_dist = 0
          for label, pred in zip(test_data[3], idx2label(test_pred_value, args.test_voc_type)):
            batch_edit_dist += float(editdistance.eval(label, pred)) / max(len(pred), len(label), 1)
            print(pred + " : " + label)
          tot_edit_dist += batch_edit_dist / args.test_batch_size
    
    print("Mean edit distance: ", tot_edit_dist / args.test_batches)
          


if __name__ == "__main__":
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    args = get_args(sys.argv[1:])
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    main_test(args)
