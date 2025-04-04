import tensorflow as tf
## this only creates loss functions for when nodata values are -1
def masked_mse_loss(no_data_value=-1.0):
    def loss(y_true, y_pred):
        # Create mask where y_true != no_data_value
        mask = tf.not_equal(y_true, no_data_value)
        mask = tf.cast(mask, tf.float32)

        # Apply mask
        squared_diff = tf.square(y_true - y_pred) * mask

        # Avoid division by zero
        loss_value = tf.reduce_sum(squared_diff) / tf.reduce_sum(mask)
        return loss_value
    return loss
