import argparse
from dqn.agent import DQNAgent as DQNAgent
from A3C.agent import A3CAgent as A3CGlobalAgent

parser = argparse.ArgumentParser(description='Run the Trainingsprocess or Evaluationprocess for a DQN- or A3C-Model.')
parser.add_argument("-m", "--model", type=str, action='store', help="Please specify the agent you wish to use, either DQN or A3C", required=True)
parser.add_argument("-n", "--mode", type=str, action='store', help="Please specify the mode you wish to run, either train or eval", required=True)

args = parser.parse_args()
print(args)

if args.model == 'DQN':
    agent = DQNAgent()

    if args.mode == 'train':
        agent.train()
    
    if args.mode == 'eval':
        agent.Evaluate()

if args.model == 'A3C':
    agent = A3CGlobalAgent()

    if args.mode == 'train':
        agent.train()
    
    if args.mode == 'eval':
        agent.Evaluate()
