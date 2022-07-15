import argparse
import gym



def build_arg_parser():
    parser = argparse.ArgumentParser(description = 'Run an environment')
    parser.add_argument('--input-env', dest = 'input_env', required = True, choices = ['cartpole', 'mountaincar', 'pendulum', 'taxi', 'lake'],
                        help = 'Specify the name of the environment')
    
    return parser



if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    input_env = args.input_env
    
    #Create a mapping from the input input argument string to the environments in OpenAI gym package
    name_map = {'cartpole': 'CartPole-v1', 'mountaincar': 'MountainCar-v0', 'pendulum': 'Pendulum-v1', 'taxi': 'Taxi-v3', 'lake': 'FrozenLake-v1'}
    
    #Create the environment based on the input arg and reset it
    env = gym.make(name_map[input_env])
    env.reset()
    
    #Iterate 1000 times and take a random action during each step
    for _ in range(1000):
        #Render the environment
        env.render()
        
        #Take a random action
        env.step(env.action_space.sample())