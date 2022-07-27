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
    
    #Create the environment based on the input argument
    env = gym.make(name_map[input_env])
    
    #Start iterating by resetting the environment
    for _ in range(20):
        observation = env.reset()   #Reset the environment
        
        #For each reset, iterate 100 times
        for i in range(100):
            env.render()   #Render the environment
            
            print(observation)   #Print current observation and Take action
            action = env.action_space.sample()   
            
            #Extract the consequences of taking the current action
            observation, reward, done, info = env.step(action)
            
            #Check if goal has been achieved
            if done:
                print(f'Episode finished after {i+1} timesteps')
                break