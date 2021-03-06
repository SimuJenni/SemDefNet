import tensorflow as tf

from tensorflow.python.framework import ops

import os

from utils import montage_tf, weights_montage
from constants import LOG_DIR

slim = tf.contrib.slim


class SDNetTester:
    def __init__(self, model, dataset, pre_processor, tag='default'):
        tf.logging.set_verbosity(tf.logging.DEBUG)
        self.sess = tf.Session()
        self.graph = tf.Graph()
        self.model = model
        self.dataset = dataset
        self.tag = tag
        self.additional_info = None
        self.im_per_smry = 4
        self.summaries = {}
        self.pre_processor = pre_processor
        self.num_eval_steps = None
        with self.sess.as_default():
            with self.graph.as_default():
                self.global_step = slim.create_global_step()

    def get_save_dir(self):
        fname = '{}_{}_{}_finetune'.format(self.dataset.name, self.model.name, self.tag)
        if self.additional_info:
            fname = '{}_{}'.format(fname, self.additional_info)
        return os.path.join(LOG_DIR, '{}/'.format(fname))

    def get_test_batch(self, dataset_id=None):
        with tf.device('/cpu:0'):
            if dataset_id:
                test_set = self.dataset.get_split(dataset_id)
                self.num_eval_steps = (self.dataset.get_num_dataset(dataset_id) / self.model.batch_size)
            else:
                test_set = self.dataset.get_testset()
                self.num_eval_steps = (self.dataset.get_num_test() / self.model.batch_size)
            print('Number of evaluation steps: {}'.format(self.num_eval_steps))
            provider = slim.dataset_data_provider.DatasetDataProvider(test_set, num_readers=1, shuffle=False,
                                                                      common_queue_capacity=20 * self.model.batch_size,
                                                                      common_queue_min=10 * self.model.batch_size)

            [img_test, label_test] = provider.get(['image', 'label'])
            label_test -= self.dataset.label_offset

            # Pre-process data
            img_test = self.pre_processor.process_test(img_test)

            # Make batches
            imgs_test, labels_test = tf.train.batch([img_test, label_test], batch_size=self.model.batch_size,
                                                    num_threads=8,
                                                    capacity=512)
            batch_queue = slim.prefetch_queue.prefetch_queue([imgs_test, labels_test])

            return batch_queue.dequeue()

    def get_random_test_crops(self):
        with tf.device('/cpu:0'):
            # Get eval-data
            test_set = self.dataset.get_testset()
            self.num_eval_steps = (self.dataset.get_num_test() / self.model.batch_size)
            test_provider = slim.dataset_data_provider.DatasetDataProvider(test_set, num_readers=1, shuffle=False)
            [img_test, label_test] = test_provider.get(['image', 'label'])
            labels_test = tf.tile(tf.expand_dims(label_test, axis=0), [10, 1])
            imgs_test_t = tf.tile(tf.expand_dims(img_test, axis=0), [10, 1, 1, 1])
            imgs_test_p = tf.unstack(imgs_test_t, axis=0, num=10)
            imgs_test_p = [self.pre_processor.process_transfer_test(im) for im in imgs_test_p]
            imgs_test = tf.stack(imgs_test_p, axis=0)

        return imgs_test, labels_test

    def test_classifier(self, num_conv_trained=None, dataset_id=None):
        if not self.additional_info:
            self.additional_info = 'conv_{}'.format(num_conv_trained)
        print('Restoring from: {}'.format(self.get_save_dir()))
        with self.sess.as_default():
            with self.graph.as_default():
                # Get training batches
                imgs_test, labels_test = self.get_test_batch(dataset_id)

                # Get predictions
                predictions = self.model.classifier(imgs_test, self.dataset.num_classes, training=False)

                # Compute predicted label for accuracy
                preds_test = tf.argmax(predictions, 1)

                # Choose the metrics to compute:
                names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
                    'accuracy': slim.metrics.streaming_accuracy(preds_test, labels_test),
                })
                summary_ops = self.make_summaries(names_to_values)

                # Start evaluation
                slim.evaluation.evaluation_loop('', self.get_save_dir(), self.get_save_dir(),
                                                num_evals=self.num_eval_steps,
                                                max_number_of_evaluations=10,
                                                eval_op=names_to_updates.values(),
                                                summary_op=tf.summary.merge(summary_ops))

    def test_classifier_voc(self, num_conv_trained=None):
        self.additional_info = 'conv_{}'.format(num_conv_trained)
        with self.sess.as_default():
            with self.graph.as_default():
                # Get training batches
                imgs_test, labels_test = self.get_random_test_crops()

                # Get predictions
                preds_test = self.model.classifier(imgs_test, self.dataset.num_classes, training=False)
                preds_test = tf.nn.sigmoid(preds_test)
                preds_test = tf.reduce_mean(preds_test, axis=0, keep_dims=True)

                # Compute mean Average Precision
                map_test, summary_ops, update_ops = self.compute_mAP(labels_test, preds_test)
                op = tf.summary.scalar('map_test', map_test)
                op = tf.Print(op, [map_test], 'map_test', summarize=30)
                summary_ops.append(op)

                # Start evaluation
                slim.evaluation.evaluation_loop('', self.get_save_dir(), self.get_save_dir(),
                                                num_evals=self.num_eval_steps,
                                                max_number_of_evaluations=100,
                                                eval_op=update_ops,
                                                summary_op=tf.summary.merge(summary_ops))

    def compute_mAP(self, labels_test, preds_test):
        summary_ops = []
        update_ops = []
        thresholds = [0.01 * i for i in range(101)]
        map_test = tf.Variable(0, dtype=tf.float32, collections=[ops.GraphKeys.LOCAL_VARIABLES])
        for c in range(20):
            class_pred_test = tf.slice(preds_test, [0, c], size=[self.model.batch_size, 1])
            class_label_test = tf.slice(labels_test, [0, c], size=[self.model.batch_size, 1])

            # Choose the metrics to compute:
            prec_test, update_prec_test = slim.metrics.streaming_precision_at_thresholds(
                class_pred_test, class_label_test, thresholds)
            rec_test, update_rec_test = slim.metrics.streaming_recall_at_thresholds(
                class_pred_test, class_label_test, thresholds)
            update_ops.append([update_prec_test, update_rec_test])

            ap_test = tf.Variable(0, dtype=tf.float32, collections=[ops.GraphKeys.LOCAL_VARIABLES])
            for i in range(11):
                ap_test += tf.reduce_max(
                    prec_test * tf.cast(tf.greater_equal(rec_test, 0.1 * i), tf.float32)) / 10
            map_test += tf.to_float(ap_test) / 20.

            op = tf.summary.scalar('ap_test_{}'.format(c), ap_test)
            op = tf.Print(op, [ap_test], 'ap_test_{}'.format(c), summarize=30)
            summary_ops.append(op)
        return map_test, summary_ops, update_ops

    def test_reconstruction(self):
        model_dir = os.path.join(LOG_DIR, '{}_{}_{}/'.format(self.dataset.name, self.model.name, self.tag))
        print('Restoring from: {}'.format(model_dir))
        with self.sess.as_default():
            with self.graph.as_default():
                imgs_test, __ = self.get_test_batch()

                # Create the model
                dec_im, dec_pdrop = self.model.net(imgs_test, train=False)

                # Create the summary ops such that they also print out to std output:
                summary_ops = [tf.summary.image('images/autoencoded', montage_tf(dec_im[:24], 4, 6), max_outputs=1),
                               tf.summary.image('images/fake', montage_tf(dec_pdrop[:24], 4, 6), max_outputs=1),
                               tf.summary.image('images/original', montage_tf(imgs_test[:24], 4, 6), max_outputs=1)]
                with tf.variable_scope('discriminator', reuse=True):
                    weights_disc_1 = slim.variable('conv_1/weights')
                summary_ops.append(tf.summary.image('images/weights_disc_1', weights_montage(weights_disc_1, 6, 16),
                                                    max_outputs=1))

                slim.evaluation.evaluation_loop('', model_dir, model_dir,
                                                num_evals=1,
                                                max_number_of_evaluations=50,
                                                summary_op=tf.summary.merge(summary_ops))

    def make_summaries(self, names_to_values):
        # Create the summary ops such that they also print out to std output:
        summary_ops = []
        for metric_name, metric_value in names_to_values.iteritems():
            op = tf.summary.scalar(metric_name, metric_value)
            op = tf.Print(op, [metric_value], metric_name)
            summary_ops.append(op)
        return summary_ops

    def test_classifier_cv(self, num_conv_trained=None, fold=0):
        self.additional_info = 'conv_{}_fold_{}'.format(num_conv_trained, fold)
        self.test_classifier(num_conv_trained)

