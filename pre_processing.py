import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

"""
@author: Erik Dale
@date: 25.03.23
"""


def pre_process_cnn_input(image):
    """
    Pre-processes the given image for use as input to a convolutional neural network (CNN) model.
    :param image: the raw image as a NumPy array.
    :return: the pre-processed image as a grayscale NumPy array.
    """
    # Convert the image to a PIL Image object.
    image = Image.fromarray(image)

    # Crop the image to the desired size.
    image = image.crop((0, 0, 400, 288))

    # Resize the image to 102x86 pixels.
    image = image.resize((102, 86))

    # Resize the image to 84x84 pixels.
    image = image.resize((84, 84))

    # Grayscale the image
    gray_image = image.convert('L')

    # Apply adaptive thresholding
    thresholded_image = ImageOps.autocontrast(gray_image, cutoff=2).convert('1')

    thresholded_image = np.array(thresholded_image)

    thresholded_image = np.expand_dims(thresholded_image, axis=-1)

    # Normalize the image.
    normalized_image = tf.image.per_image_standardization(thresholded_image)

    return normalized_image


def pre_process_dnn_input(state_reward_struct):
    """
    Pre-processes the given state-reward structure for use as input to a deep neural network (DNN) model.
    :param state_reward_struct: a dictionary containing the state and reward information.
    :return: the pre-processed state information as a normalized NumPy array.
    """
    state = [float(state_reward_struct['y'])]

    # Append the y-coordinate of the bird to the state vector.

    # Append a binary flag indicating whether the bird has collided with the ground to the state vector.
    if state_reward_struct['groundCrash']:
        state.append(float(1))
    else:
        state.append(float(0))

    # Append the x-coordinate of the base to the state vector.
    state.append(float(state_reward_struct['basex']))

    # Append the x- and y-coordinates of the first upper pipe to the state vector.
    state.append(float(state_reward_struct['upperPipes'][0]['x']))
    state.append(float(state_reward_struct['upperPipes'][0]['y']))

    # Append the x- and y-coordinates of the second upper pipe to the state vector.
    state.append(float(state_reward_struct['upperPipes'][1]['x']))
    state.append(float(state_reward_struct['upperPipes'][1]['y']))

    # Append the x- and y-coordinates of the first lower pipe to the state vector.
    state.append(float(state_reward_struct['lowerPipes'][0]['x']))
    state.append(float(state_reward_struct['lowerPipes'][0]['y']))

    # Append the x- and y-coordinates of the second lower pipe to the state vector.
    state.append(float(state_reward_struct['lowerPipes'][1]['x']))
    state.append(float(state_reward_struct['lowerPipes'][1]['y']))

    # Append the vertical velocity of the bird to the state vector.
    state.append(float(state_reward_struct['playerVelY']))

    # Append the rotation of the bird to the state vector.
    state.append(float(state_reward_struct['playerRot']))

    # Normalize the state vector.
    state = tf.keras.utils.normalize([state], axis=1)

    return state

