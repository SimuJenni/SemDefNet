import tensorflow as tf
import tensorflow.contrib.slim as slim
from AlexNet_avgDisc_concat_noBN import AlexNet
from layers_new import up_conv2d, pixel_dropout, res_block_bottleneck
from utils import montage_tf


DEFAULT_FILTER_DIMS = [64, 128, 256, 512, 1024]


def toon_net_argscope(activation=tf.nn.elu, kernel_size=(3, 3), padding='SAME', training=True, center=True,
                      w_reg=0.0001, fix_bn=False):
    """Defines default parameter values for all the layers used in ToonNet.

    Args:
        activation: The default activation function
        kernel_size: The default kernel size for convolution layers
        padding: The default border mode
        training: Whether in train or eval mode
        center: Whether to use centering in batchnorm
        w_reg: Parameter for weight-decay

    Returns:
        An argscope
    """
    train_bn = training and not fix_bn
    batch_norm_params = {
        'is_training': train_bn,
        'decay': 0.99,
        'epsilon': 0.001,
        'center': center,
        'fused': True
    }
    he = tf.contrib.layers.variance_scaling_initializer(mode='FAN_AVG')
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.convolution2d_transpose],
                        activation_fn=activation,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params,
                        weights_regularizer=slim.l2_regularizer(w_reg),
                        biases_regularizer=slim.l2_regularizer(w_reg),
                        weights_initializer=he):
        with slim.arg_scope([slim.conv2d, slim.convolution2d_transpose],
                            kernel_size=kernel_size,
                            padding=padding):
            with slim.arg_scope([slim.batch_norm], **batch_norm_params):
                with slim.arg_scope([slim.dropout], is_training=training) as arg_sc:
                    return arg_sc


class SDNet:
    def __init__(self, num_layers, batch_size, target_shape, num_res=1, pool5=True, tag='default', fix_bn=False,
                 disc_pad='VALID'):
        """Initialises a ToonNet using the provided parameters.

        Args:
            num_layers: The number of convolutional down/upsampling layers to be used.
            batch_size: The batch-size used during training (used to generate training labels)
        """
        self.name = 'SDNet_avgDisc_concat_new_res{}_{}'.format(num_res, tag)
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.im_shape = target_shape
        self.num_res_layers = num_res
        self.discriminator = AlexNet(fix_bn=fix_bn, pool5=pool5, pad=disc_pad)
        self.dec_im = self.dec_drop = self.disc_out = self.drop_pred = self.drop_label = self.rec_weights = None

    def net(self, imgs, reuse=None, train=True):
        """Builds the full ToonNet architecture with the given inputs.

        Args:
            imgs: Placeholder for input images
            reuse: Whether to reuse already defined variables.
            train: Whether in train or eval mode

        Returns:
            dec_im: The autoencoded image
            dec_gen: The reconstructed image from cartoon and edge inputs
            disc_out: The discriminator output
            enc_im: Encoding of the image
            gen_enc: Output of the generator
        """
        # Concatenate cartoon and edge for input to generator
        enc_im = self.encoder(imgs, reuse=reuse, training=train)
        pixel_drop, drop_mask = pixel_dropout(enc_im, 0.5)
        self.rec_weights = tf.image.resize_nearest_neighbor(drop_mask, self.im_shape[:2])
        drop_label_fake = slim.flatten(drop_mask)
        drop_label_real = tf.ones_like(drop_label_fake)
        self.drop_label = tf.concat([drop_label_real, drop_label_fake], 0)
        tf.summary.image('imgs/drop_mask', montage_tf(drop_mask, 2, 8), max_outputs=1)

        # Decode both encoded images and generator output using the same decoder
        self.dec_im = self.decoder(enc_im, reuse=reuse, training=train)
        self.dec_drop = self.generator(pixel_drop, drop_mask, reuse=reuse, training=train)

        # Build input for discriminator (discriminator tries to guess order of real/fake)
        disc_in = tf.concat([self.dec_im, self.dec_drop], 0)
        self.disc_out, self.drop_pred, __ = self.discriminator.discriminate(disc_in, reuse=reuse, training=train)

        return self.dec_im, self.dec_drop

    def labels_real(self):
        labels = tf.concat([tf.ones((self.batch_size,), dtype=tf.int64), tf.zeros((self.batch_size,), dtype=tf.int64)], 0)
        return tf.one_hot(labels, 2)

    def labels_fake(self):
        labels = tf.concat([tf.zeros((self.batch_size,), dtype=tf.int64), tf.ones((self.batch_size,), dtype=tf.int64)], 0)
        return tf.one_hot(labels, 2)

    def classifier(self, img, num_classes, reuse=None, training=True):
        """Builds a classifier on top either the encoder, generator or discriminator trained in the AEGAN.

        Args:
            img: Input image
            num_classes: Number of output classes
            reuse: Whether to reuse already defined variables.
            training: Whether in train or eval mode

        Returns:
            Output logits from the classifier
        """
        _, _, model = self.discriminator.discriminate(img, reuse=reuse, training=training, with_fc=False)
        model = self.discriminator.classify(model, num_classes, reuse=reuse, training=training)
        return model

    def generator(self, net, drop_mask, reuse=None, training=True):
        """Builds a generator with the given inputs. Noise is induced in all convolutional layers.

        Args:
            net: Input to the generator (i.e. cartooned image and/or edge-map)
            reuse: Whether to reuse already defined variables
            training: Whether in train or eval mode.

        Returns:
            Encoding of the input.
        """
        f_dims = DEFAULT_FILTER_DIMS
        res_dim = DEFAULT_FILTER_DIMS[self.num_layers - 1]
        with tf.variable_scope('generator', reuse=reuse):
            with slim.arg_scope(toon_net_argscope(padding='SAME', training=training)):
                net_in = net
                for i in range(self.num_res_layers):
                    net = res_block_bottleneck(net, res_dim, res_dim / 4, noise_channels=32, scope='res_{}'.format(i))
                    net = net_in + (1.0-drop_mask)*net
                for l in range(0, self.num_layers - 1):
                    net = up_conv2d(net, num_outputs=f_dims[self.num_layers - l - 2], scope='deconv_{}'.format(l))
                net = tf.image.resize_images(net, (self.im_shape[0], self.im_shape[1]),
                                             tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                net = slim.conv2d(net, num_outputs=32, scope='deconv_{}'.format(self.num_layers), stride=1)
                net = slim.conv2d(net, num_outputs=3, scope='deconv_{}'.format(self.num_layers + 1), stride=1,
                                  activation_fn=tf.nn.tanh, normalizer_fn=None)
                return net

    def encoder(self, net, reuse=None, training=True):
        """Builds an encoder of the given inputs.

        Args:
            net: Input to the encoder (image)
            reuse: Whether to reuse already defined variables
            training: Whether in train or eval mode.

        Returns:
            Encoding of the input image.
        """
        f_dims = DEFAULT_FILTER_DIMS
        with tf.variable_scope('encoder', reuse=reuse):
            with slim.arg_scope(toon_net_argscope(padding='SAME', training=training)):
                net = slim.conv2d(net, num_outputs=32, stride=1, scope='conv_0')
                for l in range(0, self.num_layers):
                    net = slim.conv2d(net, num_outputs=f_dims[l], stride=2, scope='conv_{}'.format(l + 1))

                return net

    def decoder(self, net, reuse=None, training=True):
        """Builds a decoder on top of net.

        Args:
            net: Input to the decoder (output of encoder)
            reuse: Whether to reuse already defined variables
            training: Whether in train or eval mode.

        Returns:
            Decoded image with 3 channels.
        """
        f_dims = DEFAULT_FILTER_DIMS
        with tf.variable_scope('decoder', reuse=reuse):
            with slim.arg_scope(toon_net_argscope(padding='SAME', training=training)):
                for l in range(0, self.num_layers - 1):
                    net = up_conv2d(net, num_outputs=f_dims[self.num_layers - l - 2], scope='deconv_{}'.format(l))
                net = tf.image.resize_images(net, (self.im_shape[0], self.im_shape[1]),
                                             tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                net = slim.conv2d(net, num_outputs=32, scope='deconv_{}'.format(self.num_layers), stride=1)
                net = slim.conv2d(net, num_outputs=3, scope='deconv_{}'.format(self.num_layers + 1), stride=1,
                                  activation_fn=tf.nn.tanh, normalizer_fn=None)
                return net

    def ae_loss(self, imgs_train):
        # Define the losses for AE training
        ae_loss_scope = 'ae_loss'
        ae_loss = tf.losses.mean_squared_error(imgs_train, self.dec_im, scope=ae_loss_scope, weights=30.0)
        tf.summary.scalar('losses/autoencoder_mse', ae_loss)
        losses_ae = tf.losses.get_losses(ae_loss_scope)
        losses_ae += tf.losses.get_regularization_losses(ae_loss_scope)
        ae_total_loss = tf.add_n(losses_ae, name='ae_total_loss')
        return ae_total_loss

    def generator_loss(self, imgs_train):
        # Define the losses for generator training
        gen_loss_scope = 'gen_loss'
        fake_loss = tf.losses.softmax_cross_entropy(self.labels_fake(), self.disc_out, scope=gen_loss_scope,
                                                    weights=1.0)
        tf.summary.scalar('losses/generator', fake_loss)
        ae_loss = tf.losses.mean_squared_error(imgs_train, self.dec_drop, scope=gen_loss_scope,
                                               weights=30.0*self.rec_weights)
        tf.summary.scalar('losses/generator_mse', ae_loss)
        drop_pred_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(self.drop_label), self.drop_pred,
                                                         scope=gen_loss_scope, weights=0.1)
        tf.summary.scalar('losses/generator_drop', drop_pred_loss)
        losses_gen = tf.losses.get_losses(gen_loss_scope)
        losses_gen += tf.losses.get_regularization_losses(gen_loss_scope)
        gen_loss = tf.add_n(losses_gen, name='gen_total_loss')
        return gen_loss

    def discriminator_loss(self):
        # Define loss for discriminator training
        disc_loss_scope = 'disc_loss'
        real_loss = tf.losses.softmax_cross_entropy(self.labels_real(), self.disc_out, scope=disc_loss_scope,
                                                    weights=1.0)
        tf.summary.scalar('losses/discriminator', real_loss)
        drop_pred_loss = tf.losses.sigmoid_cross_entropy(self.drop_label, self.drop_pred, scope=disc_loss_scope,
                                                         weights=0.1)
        tf.summary.scalar('losses/discriminator_drop', drop_pred_loss)
        losses_disc = tf.losses.get_losses(disc_loss_scope)
        losses_disc += tf.losses.get_regularization_losses(disc_loss_scope)
        disc_total_loss = tf.add_n(losses_disc, name='disc_total_loss')

        real_pred = tf.arg_max(self.disc_out, 1)
        real_true = tf.arg_max(self.labels_real(), 1)
        tf.summary.scalar('accuracy/discriminator', slim.metrics.accuracy(real_pred, real_true))

        return disc_total_loss
