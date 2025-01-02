import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.regularizers import l2

def create_model(num_classes, input_shape=(224, 224, 3), base_model='VGG16', 
                dense_layers=[1024], dropout_rate=0.5, l2_reg=0.01, 
                activation_function='relu', kernel_initializer='glorot_uniform',
                use_batch_norm=True, batch_norm_momentum=0.99):  # Added batch norm parameters
    """
    Create a CNN model with configurable batch normalization.
    
    Args:
        ... (previous parameters) ...
        use_batch_norm: Whether to use batch normalization
        batch_norm_momentum: Momentum for batch normalization layer
    """
    if base_model == 'VGG16':
        base = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    elif base_model == 'ResNet50':
        base = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    else:
        raise ValueError(f"Unsupported base model: {base_model}")
    
    inputs = tf.keras.Input(shape=input_shape)
    x = base(inputs)
    x = GlobalAveragePooling2D()(x)
    
    # Add dense layers with optional batch normalization
    for units in dense_layers:
        if use_batch_norm:
            x = tf.keras.layers.BatchNormalization(momentum=batch_norm_momentum)(x)
        x = Dense(units, 
                 kernel_regularizer=l2(l2_reg),
                 kernel_initializer=kernel_initializer)(x)
        if use_batch_norm:
            x = tf.keras.layers.BatchNormalization(momentum=batch_norm_momentum)(x)
        x = tf.keras.layers.Activation(activation_function)(x)
        x = Dropout(dropout_rate)(x)
    
    # Output layer
    if use_batch_norm:
        x = tf.keras.layers.BatchNormalization(momentum=batch_norm_momentum)(x)
    outputs = Dense(num_classes,
                   activation='softmax',
                   kernel_regularizer=l2(l2_reg),
                   kernel_initializer=kernel_initializer)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    # Freeze base model layers
    for layer in base.layers:
        layer.trainable = False
    
    return model