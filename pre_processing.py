import tensorflow as tf
from PIL import Image
import numpy as np


def pre_process_cnn_input(image):
    image = Image.fromarray(image)
    image = image.crop((0, 0, 400, 288))
    image = image.resize((102, 86))
    image = image.resize((84, 84))
    image = np.array(image)
    normalized_image = tf.image.per_image_standardization(image)
    gray_image = tf.image.rgb_to_grayscale(normalized_image)
    return gray_image


def pre_process_dnn_input(state_reward_struct):
    state = []
    state.append(float(state_reward_struct['y']))
    if state_reward_struct['groundCrash']:
        state.append(float(1))
    else:
        state.append(float(0))

    state.append(float(state_reward_struct['basex']))

    state.append(float(state_reward_struct['upperPipes'][0]['x']))
    state.append(float(state_reward_struct['upperPipes'][0]['y']))
    state.append(float(state_reward_struct['upperPipes'][1]['x']))
    state.append(float(state_reward_struct['upperPipes'][1]['y']))

    state.append(float(state_reward_struct['lowerPipes'][0]['x']))
    state.append(float(state_reward_struct['lowerPipes'][0]['y']))
    state.append(float(state_reward_struct['lowerPipes'][1]['x']))
    state.append(float(state_reward_struct['lowerPipes'][1]['y']))

    state.append(float(state_reward_struct['playerVelY']))
    state.append(float(state_reward_struct['playerRot']))

    state = tf.keras.utils.normalize([state], axis=1)
    return state
