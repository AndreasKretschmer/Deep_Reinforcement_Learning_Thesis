## Reinforcement Learning: Breakout
Implementation of 
- DQN - Agent
- A3C - Agent
to 

## Prerequistes
You can install all the prerequistes for code use using 

```text
pip install -r requirements.txt
```

## Instructions on Use

To run the code use
```python
  python main.py
```
with arguments where arguments are given by

```text
usage: main.py [-h] -n NETWORK -m MODE

Train and test different networks on Space Invaders

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model     MODEL
                        Please specify the model you wish to use, either DQN or A3C
  -n MODE, --mode MODE  Please specify the mode you wish to run, either train or eval
  ```
  
  For example, to test the pre-trained Deep Q Network architecture and view the network playing Breakout use
  
  ```text
    python main.py -n DQN -m train
  ```

