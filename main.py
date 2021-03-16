import argparse
from dqn.agent import agent as DQNAgent
from A3C.agent import agent as A3CGlobalAgent

# argparse.add_argument("-m", "--model", type=str, action='store', help="Please specify the agent you wish to use, either DQN or A3C", required=True)
# argparse.add_argument("-n", "--mode", type=str, action='store', help="Please specify the mode you wish to run, either train or test", required=True)

# args = argparse.parse_args()
# print(args)

# if args.model == 'DQN':
# agent = DQNAgent()

#     if args.mode == 'train':
# agent.train()
    
#     if args.mode == 'test':
#         pass

# if args.model == 'A3C':
agent = A3CGlobalAgent()

    # if args.mode == 'train':
agent.train()
    
    # if args.mode == 'test':
    #     pass
