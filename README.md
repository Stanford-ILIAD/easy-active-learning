This code learns reward functions from human preferences in various tasks by actively generating queries to the human user based on maximum information gain.

Companion code to CoRL 2019 submission.

## Dependencies
You need to have the following libraries with [Python3](http://www.python.org/downloads):
- [MuJoCo 2.0](http://www.mujoco.org/index.html)
- [NumPy](https://www.numpy.org/)
- [OpenAI Gym](https://gym.openai.com)
- [pyglet](https://bitbucket.org/pyglet/pyglet/wiki/Home)
- [SciPy](https://www.scipy.org/)
- [theano](http://deeplearning.net/software/theano/)

### Sampling the input space
This is the preprocessing step, so you need to run it only once (subsequent runs will overwrite for each task). It is not interactive and necessary only if you will use batch active preference-based learning. For non-batch version and random querying, you can skip this step.

You simply run
```python
	python input_sampler.py [task_name] D
```
For quick (but highly suboptimal) results, we recommend D=1000. In the article, we used D=500000.

## Running
Throughout this demo,
- [task_name] should be selected as one of the following: Driver, LunarLander, MountainCar, Swimmer, Tosser, LDS
- [criterion] should be selected as one of the following: information, volume, random
- [query_type] should be selected as one of the following: weak, strong
For the details and positive integer parameters c, M, N; we refer to the publication.
You should run the codes in the following order:

### Learning preference reward function
You can simply run
```python
	python run.py [task_name] [criterion] [query_type] c M N
```
N is only effective if c=0.
After each query, the user will be showed the w-vector learned up to that point.

### Demonstration of learned parameters
This is just for demonstration purposes. run_optimizer.py starts with 3 parameter values. You can simply modify them to see optimized behavior for different tasks and different w-vectors.
