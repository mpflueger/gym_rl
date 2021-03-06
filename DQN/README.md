# DQN Implementation for OpenAI Gym

This is an implementation of Deep Q-Networks as described by [Mnih et. al. 2015](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) applied to the classic control problems from [OpenAI Gym](https://gym.openai.com/).  This implementation uses the [TensorFlow](https://www.tensorflow.org/) framework.  In some cases I've added some reward shaping, but that is pretty easy to turn off if you have a look. 

I created this implementation for educational purposes, if you want to use it for education or research, let me know so I can choose an appropriate license.

## Usage
To train a model:
```
python {problem.py} train
```

To test a trained model:
```
python {problem.py} test --restore [checkpoint location]
```

Or just check out the options:
```
python {problem.py} --help
```
