class hyperparameters:
    #global Hyperparameters
    ENV_NAME = 'Breakout-v0'
    RESIZE_IMAGE_SIZE = (84, 84)
    SHOW_PREVIEW = False
    SAVE_LOGS_PATH = '.\logs'
    SAVE_MODEL_PATH = 'models'
    LEARNINGRATE = 0.99
    NO_ACTION_STEPS = 30
    SKIP_FRAMES = 4

    #DQN Hyperparameters
    MAX_EPISODES_DQN = 50000
    STACKED_FRAME_SIZE = 4
    EXP_SIZE = 400000
    EXP_SAMPLE_SIZE = 32
    START_EPSILON = 1
    EPSILON_DECAY_STEPS = 1000000
    MIN_EPSILON = 0.1
    MIN_EXP = 50000
    UPDATE_TARGET_EVERY = 10000
    DQN_MODEL_NAME = 'dqn.h5'

    #A3C Hyperparameters
    ACTOR_LEARNINGRATE = 0.00025
    CRITIC_LEARNINGRATE = 0.00025
    NUMBER_WORKERS = 1
    MAX_EPISODES_A3C = 50000
    ACTION_SIZE = 4

    def __init__(self):
        pass