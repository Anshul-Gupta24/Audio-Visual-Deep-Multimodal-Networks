import keras
import keras.backend as K
from keras.models import Model, load_model, Sequential
#from keras.applications.vgg16 import VGG16
from keras.layers import Lambda, Layer, Dense, Input, Dropout, merge, Multiply, Add, Masking, LSTM, BatchNormalization, Activation, TimeDistributed
import tensorflow as tf
import numpy as np
import pickle
from keras_layer_normalization import LayerNormalization




def fro_norm(w):
    return K.sqrt(K.sum(K.square(K.abs(w))))

def cust_reg(w):
    print(w.shape)
    print(type(w.get_shape().as_list()[0]))
    shape = w.get_shape().as_list()
    m = K.dot(K.transpose(w), w) - tf.eye(shape[0], shape[1])
    return 0.01*fro_norm(m)



class MyLayer(Layer):

    def __init__(self, output_dim, init, **kwargs):
        self.output_dim = output_dim
        self.init = init
        super(MyLayer, self).__init__(**kwargs)


    def my_init(self, shape, dtype=None):
        return K.variable(self.init)


    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer=self.my_init,
                                    #   regularizer=keras.regularizers.l1(0.01),
                                    #   constraint=keras.constraints.MinMaxNorm(min_value=0.0, max_value=1.0, rate=1.0, axis=0),
                                      trainable=False)   
        # temperature term
        # self.sigma = self.add_weight(name='sigma', 
        #                               shape=(1, ),
        #                               initializer=keras.initializers.RandomUniform(minval=1, maxval=30, seed=None),
        #                               trainable=True)
        super(MyLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return 20 * K.dot(x, K.l2_normalize(self.kernel, axis=0)) 

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


class JointNet():

    def __init__(self, proxy_init):

        self.init = proxy_init
        self.audio_submodel = self.audio_submodel()
        self.audio_submodel.summary()
        self.image_submodel = self.image_submodel()
        self.image_submodel.summary()
        # self.image_submodel2 = self.image_submodel2()
        self.model = self.joint_model()
        self.model.summary()


    def identity_loss(self, y_true, y_pred):

        return K.mean(y_pred)


    def bpr_triplet_loss(self, X):

        distances, class_mask, class_mask_bar = X


        # COSINE SIMILARITY
        # distance_pos = K.sum(distances * class_mask, axis=-1, keepdims=True)
        d_pos = K.exp(K.sum(distances * class_mask, axis=-1, keepdims=True))
        d_neg = K.sum(K.exp(distances * class_mask_bar), axis=-1, keepdims=True) - 1
        # distance_neg = K.exp(K.sum(distances * class_mask_bar, axis=-1, keepdims=True))
        # num_pos = K.sum(tf.cast(tf.greater(distance_neg, 0.0), tf.float32), axis=-1) + 1

        loss = K.log(tf.divide(d_neg, d_pos))# * tf.divide(1, d_pos)
        # loss_actual = tf.gather(loss, tf.nn.top_k(tf.reshape(loss, [-1]), k=80)[1]) 

        # return loss_actual
        return loss


    # def one_hot(x):
    #     n_cls = 576
    #     return tf.one_hot(tf.cast(x, tf.int32), n_cls, axis=-1)


    # def nca_loss(y_true, y_pred):
    #     y_true = tf.reshape(y_true, [-1])
    #     y_true_oh = one_hot(y_true)
    #     y_true_oh_bar = 1 - y_true_oh
    #     nr = tf.multiply(y_pred, y_true_oh)
    #     nr = tf.exp(tf.reduce_sum(nr, axis=-1))
    #     dr = tf.multiply(y_pred, y_true_oh_bar)
    #     dr = tf.reduce_sum(tf.exp(dr), axis=-1) - 1
    #     loss = -1 * tf.log(tf.divide(nr, dr))
    #     return loss



    def audio_submodel(self):

        input_size = 80
        hidden_size = 128
       
        # inp = Input((None, 80))
        # old1 = Masking(mask_value=0.0, input_shape=(None, input_size), name='masking_1')(inp)	
        # old2 = LSTM(hidden_size, return_sequences=True, input_shape=(None, input_size), name='lstm_1', trainable=True)(old1)
        # old3 = LSTM(hidden_size, return_sequences=False, input_shape=(None, hidden_size), name='lstm_2', trainable=True)(old2)
        # old4 = Dense(2048, name='dense_2048', trainable=True)(old3)
        # old5 = BatchNormalization(name='batch_normalization_2048', trainable=True)(old4)
        # old6 = Activation('relu')(old5)



        inp = Input((2048, ))
        # op1 = Activation('softmax')(inp)
        # op1 = LayerNormalization(scale=False, center=False)(inp)
        # op1 = BatchNormalization()(inp)
        op1 = Dropout(0.5)(inp)
        op2 = Dense(576, activation='tanh', input_shape=(2048, ), name='denseY_jap', trainable=True)(op1)
        # op3 = LayerNormalization(scale=False, center=False)(op2)
        # op3 = BatchNormalization()(op2)
        # op4 = Activation('relu')(op2)
        # op3 = Dense(2048, activation='relu', name='dense_aud2')(op2)
        # op3 = Dropout(0.2)(op2)

        model = Model(inputs=inp, outputs=op2, name='sequential_1_jap')
        
        ################################

        # Remember to load!!!

        ################################
        
        # model.load_weights('/home/data1/anshulg/models/model_sepspk576_newdata_2048_noisy.h5', by_name=True)
        # model.load_weights('/home/data1/anshulg/models/model_sepspk576_jap_2048_subnet_scratch_LAST/saved-model-20.hdf5', by_name=True)

        # with open('/home/anshulg/WordNet/proxy_data/speech_weights.pkl', 'rb') as fp:
        #     weights = pickle.load(fp)

        # model.layers[-1].set_weights(weights)


        return model

    
    def image_submodel(self):

        # model = Sequential(name='sequential_2')
        # model.add(Dense(576, input_shape=(2048, ), name='dense_img1'))
        # model.add(LayerNormalization(trainable=True))
        # model.add(Dropout(0.2))
        # model.add(Dense(768, activation='relu', name='dense_img2'))
        # model.add(Dropout(0.2))
        # model.add(Dense(512, activation='relu', name='dense_img3'))


        inp = Input((2048, ))
        # op1 = Activation('softmax')(inp)
        # op1 = LayerNormalization(scale=False, center=False)(inp)
        # op1 = BatchNormalization()(inp)
        op1 = Dropout(0.5)(inp)
        op2 = Dense(576, activation='tanh', name='dense_img1', trainable=True)(op1)
        # op2 = Dense(576, activation='tanh', name='dense_img2', trainable=True)(op1)
        # op3 = LayerNormalization(scale=False, center=False)(op2)
        # op3 = BatchNormalization()(op2)
        # op4 = Activation('softmax')(op3)
        # op3 = Dense(2048, activation='relu', name='dense_img2')(op2)
        # op3 = Dropout(0.2)(op2)

        model = Model(inputs=inp, outputs=op2, name='sequential_2')
        

        ##########################
        #   Get xception weights
        ##########################

        # from keras.applications.xception import Xception
        # import sys
        # sys.path.insert(0, '/home/anshulg/WordNet/get_imagenet/functions')
        # import imagenet_posterior
        # classes = open('/home/anshulg/WordNet/labels_removed2.txt').read().split('\n')
        # classes = classes[:-1]
        # xinds = imagenet_posterior.xception_inds(classes)

        # xception_model = Xception(weights='imagenet')
        # weights = xception_model.layers[-1].get_weights()
        # kernel_weights = xception_model.layers[-1].get_weights()[0][:, xinds]
        # bias_weights = xception_model.layers[-1].get_weights()[1][xinds]

        # with open('/home/anshulg/WordNet/proxy_data/xception_weights.pkl', 'rb') as fp:
        #     weights = pickle.load(fp)

        # model.layers[-1].set_weights(weights)


        return model


    # def image_submodel2(self):

    #     inp = Input((576, ))
    #     op1 = LayerNormalization(scale=False, center=False)(inp)
    #     op2 = Dense(2048, activation='relu', name='dense_img1', trainable=True)(op1)
    #     op3 = LayerNormalization(scale=False, center=False)(op2)
    #     op2_2 = Dense(1024, activation='relu', name='dense_img2', trainable=True)(op3)
    #     op3_2 = LayerNormalization(scale=False, center=False)(op2_2)
    #     op2_3 = Dense(576, name='dense_img3', trainable=True)(op3_2)
    #     op3_3 = LayerNormalization(scale=False, center=False)(op2_3)

    #     model = Model(inputs=inp, outputs=op3_3, name='sequential_3')

    #     return model

    
    def joint_model(self):

        NUM_CLASSES = 90

        grounding = Input((1, ), name='grounding')
        grounding_bar = Input((1, ), name='grounding_bar')
        anchor_aud = Input((2048, ), name='anchor_aud')
        anchor_img = Input((2048, ), name='anchor_img')
        class_mask = Input((NUM_CLASSES, ), name='class_mask')
        class_mask_bar = Input((NUM_CLASSES, ), name='class_mask_bar')
        

        anchor_aud_latent = self.audio_submodel(anchor_aud)
        anchor_img_latent = self.image_submodel(anchor_img)
        # anchor_img_latent = self.image_submodel2(anchor_img_latent_old)
      

        anchor = Add()([Multiply()([grounding_bar, anchor_img_latent]), Multiply()([grounding, anchor_aud_latent])])     
        anchor_norm = Lambda(lambda x: K.l2_normalize(x, axis=-1))(anchor)


        # proxy_mat = Dense(576, input_shape=(2048, ), use_bias=False, kernel_initializer='uniform', name='proxy_mat', trainable=True)
        proxy_mat = MyLayer(NUM_CLASSES, self.init, name='my_layer_1_jap')
        distances = proxy_mat(anchor_norm)


        loss = merge(
            [distances, class_mask, class_mask_bar],
            mode=self.bpr_triplet_loss,
            name='loss',
            output_shape=(1, ))

        model = Model(
            input=[grounding, grounding_bar, anchor_aud, anchor_img, class_mask, class_mask_bar],
            output=loss)
        model.compile(loss=self.identity_loss, optimizer=keras.optimizers.Adam(lr=0.001, decay=1e-5))

        return model