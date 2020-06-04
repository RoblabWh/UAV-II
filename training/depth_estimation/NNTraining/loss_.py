
import tensorflow.keras.backend as K
import tensorflow as tf

def depth_loss_function(y_true, y_pred, theta=0.1, maxDepthVal=65536.0/655.0):
    #y_true_slice = y_true[:, 18:-9, 19:-18, :]
    # Point-wise depth
    #l_depth = K.mean(K.abs((y_pred - K.expand_dims(y_true[:, :, :, 0], -1)) * K.expand_dims(y_true[:, :, :, 1], -1)), axis=-1)
    l_depth = K.mean(K.abs(y_pred - y_true[:, :, :, 0:1]) * y_true[:, :, :, 1:2] * (y_true[:, :, :, 2:3] * 1 + 1))
    #print(y_pred.get_shape(), y_true.get_shape())
    #print(K.max(y_pred), K.max(y_true[:, :, :, 0:1]))
    #print(l_depth)
    #print(l_depth.shape)
    #print(K.mean(l_depth).shape)
    # Edges
    dy_true, dx_true = tf.image.image_gradients(K.expand_dims(y_true[:, :, :, 0], -1))
    dy_pred, dx_pred = tf.image.image_gradients(y_pred)
    l_edges = K.mean((K.abs(dy_pred - dy_true) + K.abs(dx_pred - dx_true)) * K.expand_dims(y_true[:, :, :, 1], -1), axis=-1)

    # Structural similarity (SSIM) index
    l_ssim = K.clip((1 - tf.image.ssim(K.expand_dims(y_true[:, :, :, 0], -1) * K.expand_dims(y_true[:, :, :, 1], -1), y_pred * K.expand_dims(y_true[:, :, :, 1], -1), maxDepthVal)) * 0.5, 0, 1)

    # Weights
    w1 = 1.0
    w2 = 1.0
    w3 = theta
    #print(l_ssim, K.mean(l_edges), l_depth)
    #print((w1 * l_ssim) + (w2 * K.mean(l_edges)) + (w3 * l_depth))


    return (w1 * l_ssim) + (w2 * K.mean(l_edges)) + (w3 * l_depth)