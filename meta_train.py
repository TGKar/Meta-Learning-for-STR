import sys
import os
import time
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 


from sar_model import SARModel
from data_provider import data_generator
from data_provider import lmdb_data_generator
from data_provider.data_utils import get_vocabulary
from utils.transcription_utils import idx2label, calc_metrics
from config import get_args
from train import get_data, get_batch_data


def meta_train(args):
    
    # Get vocabularies
    voc = [0]*len(args.languages)
    char2id = [0]*len(args.languages)
    id2char = [0]*len(args.languages)
    max_voc_len = 0
    for i, lang in enumerate(args.languages):
        voc[i], char2id[i], id2char[i] = get_vocabulary(voc_type=lang)
        if max_voc_len < len(voc[i]):
            max_voc_len = len(voc[i])
    
    # Build graph
    input_train_images = tf.placeholder(dtype=tf.float32, shape=[args.train_batch_size, args.height, args.width, 3], name="input_train_images")
    input_train_images_width = tf.placeholder(dtype=tf.float32, shape=[args.train_batch_size], name="input_train_width")
    input_train_labels = tf.placeholder(dtype=tf.int32, shape=[args.train_batch_size, args.max_len], name="input_train_labels")
    input_train_labels_mask = tf.placeholder(dtype=tf.int32, shape=[args.train_batch_size, args.max_len], name="input_train_labels_mask")

    input_val_images = tf.placeholder(dtype=tf.float32, shape=[args.val_batch_size, args.height, args.width, 3],name="input_val_images")
    input_val_images_width = tf.placeholder(dtype=tf.float32, shape=[args.val_batch_size], name="input_val_width")
    input_val_labels = tf.placeholder(dtype=tf.int32, shape=[args.val_batch_size, args.max_len], name="input_val_labels")
    input_val_labels_mask = tf.placeholder(dtype=tf.int32, shape=[args.val_batch_size, args.max_len], name="input_val_labels_mask")

    sar_model = SARModel(num_classes=max_voc_len,
                  encoder_dim=args.encoder_sdim,
                  encoder_layer=args.encoder_layers,
                  decoder_dim=args.decoder_sdim,
                  decoder_layer=args.decoder_layers,
                  decoder_embed_dim=args.decoder_edim,
                  seq_len=args.max_len,
                  is_training=True,
                  trainable_backbone=False)
                  
    train_model_infer, train_attention_weights, train_pred = sar_model(input_train_images, input_train_labels,
                                                                 input_train_images_width,
                                                                 reuse=False)
    train_loss = sar_model.loss(train_model_infer, input_train_labels, input_train_labels_mask)


    # Load datasets
    train_data_list = []
    #val_data_list = []
    for lang in args.languages:
        language_path = args.meta_train_data_dir + "/" + lang
        train_data_list.append(get_data([language_path],
                    [language_path + "/gt.txt"],
                    lang,
                    args.max_len,
                    args.num_train,
                    args.height,
                    args.width,
                    args.train_batch_size,
                    args.workers,
                    args.keep_ratio,
                    with_aug=args.aug))


    # Set up optimizer and related variables
    global_step = tf.get_variable(name='global_step', initializer=tf.constant(0), trainable=False)
    learning_rate = args.inner_lr

    batch_norm_updates_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    grads = optimizer.compute_gradients(train_loss)
    apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)

    # Save summary
    os.makedirs(args.checkpoints, exist_ok=True)
    tf.summary.scalar(name='train_loss', tensor=train_loss)
    #tf.summary.scalar(name='val_loss', tensor=val_loss)
    tf.summary.scalar(name='learning_rate', tensor=learning_rate)

    merge_summary_op = tf.summary.merge_all()

    train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    model_name = 'sar_{:s}.ckpt'.format(str(train_start_time))
    model_save_path = os.path.join(args.checkpoints, model_name)
    variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([variables_averages_op, apply_gradient_op, batch_norm_updates_op]):
        train_op = tf.no_op(name='train_op')

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
    summary_writer = tf.summary.FileWriter(args.checkpoints)
      
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        summary_writer.add_graph(sess.graph)
        start_iter_outer = 0  # Outer loop current iteration
         
        meta_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="sar/Decoder")
        meta_vars += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="sar/Encoder")
              
        if args.resume == True and args.pretrained != '':
            print('Restore model from {:s}'.format(args.checkpoints))
            ckpt_state = tf.train.get_checkpoint_state(args.checkpoints)
            saver_for_restore = tf.train.Saver(tf.global_variables(), max_to_keep=1)
            restore_model_path = os.path.join(args.checkpoints, os.path.basename(ckpt_state.model_checkpoint_path))
            saver.restore(sess, restore_model_path)
            start_iter = sess.run(tf.train.get_global_step())
        else:            
            print('Restoring backbone only')
            init = tf.global_variables_initializer()
            sess.run(init)
            vars_to_restore = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="sar/resnet_backbone")            
            ckpt_state = tf.train.get_checkpoint_state(args.pretrained)
            restore_model_path = os.path.join(args.pretrained, os.path.basename(ckpt_state.model_checkpoint_path))
            saver_for_restore = tf.train.Saver(vars_to_restore)  # Removed moving averages
            print('Restore from {}'.format(restore_model_path))
            saver_for_restore.restore(sess, restore_model_path)


        # Create list of variable values before last iteration 

        old_var_vals = []
        for var in meta_vars:
            old_var_vals.append(var.eval(session=sess))
              
        while start_iter_outer < args.outer_iters:
            start_iter_outer += 1
            lang_i = np.random.choice(np.arange(len(args.languages)), p=args.languages_p)
            start_iter = 0  # Inner loop current iteration
            print("Outer iteration: ", str(start_iter_outer), "  Language: ", args.languages[lang_i])
            while start_iter < args.inner_iters:
                start_iter += 1
                train_data = get_batch_data(train_data_list[lang_i], args.train_batch_size)                
                
                _, train_loss_value, train_pred_value = sess.run([train_op, train_loss, train_pred], feed_dict={
                                                                                input_train_images: train_data[0],
                                                                                input_train_labels: train_data[1],
                                                                                input_train_labels_mask: train_data[2],
                                                                                input_train_images_width: train_data[4]})  
                if start_iter % args.log_iter == 0:  
                    print(args.languages[lang_i] + ", iter {} train loss= {:3f}".format(start_iter, train_loss_value))
        
            # Meta-update variables
            for i, var in enumerate(meta_vars):
                var_inner_val = var.eval(session=sess)
                var_reptiled_val = old_var_vals[i] + args.reptile_lr*(var_inner_val - old_var_vals[i])
                var.load(var_reptiled_val, session=sess)
                old_var_vals[i] = var_reptiled_val
            if start_iter_outer % args.save_iter == 0:
                print("Iter {} save to checkpoint".format(start_iter))
                saver.save(sess, model_save_path, global_step=global_step)
            
        print("Finished meta training! Saving to checkpoint to " + model_save_path)
        saver.save(sess, model_save_path, global_step=global_step)
        print("Model saved!")

if __name__ == "__main__":
    args = get_args(sys.argv[1:])
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    meta_train(args)
