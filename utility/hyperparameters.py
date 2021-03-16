class hyperparameters:
    #global Hyperparameters
    ENV_NAME = 'BreakoutDeterministic-v4'
    RESIZE_IMAGE_SIZE = (84, 84)
    STACKED_FRAME_SIZE = 4
    DISCOUNTFACTOR = 0.99
    NO_ACTION_STEPS = 30
    ACTION_SIZE = 3

    #DQN Hyperparameters
    MAX_EPISODES_DQN = 50000
    EXP_SIZE = 400000
    EXP_SAMPLE_SIZE = 32
    START_EPSILON = 1
    EPSILON_DECAY_STEPS = 1000000
    MIN_EPSILON = 0.1
    MIN_EXP = 50000
    UPDATE_TARGET_EVERY = 10000
    RMS_LR = 2.5e-4
    RMS_RHO = 0.95
    RMS_EPS = 0.01
    DQN_MODEL_NAME = 'dqn.h5'
    SAVE_LOGS_PATH_DQN = "./logs/breakout_dqn"
    SAVE_MODEL_PATH_DQN = "./models/dqn"

    #A3C Hyperparameters
    ACTOR_LEARNINGRATE = 2.5e-4
    CRITIC_LEARNINGRATE = 2.5e-4
    NUMBER_WORKERS = 8
    MAX_EPISODES_A3C = 80000
    STEP_MAX = 20
    SAVE_LOGS_PATH_A3C = "./logs/breakout_a3c"
    SAVE_MODEL_PATH_A3C = "./models/a3c"

    def __init__(self):
        pass