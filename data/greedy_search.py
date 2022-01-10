from argparse import ArgumentParser
from data.simpleai.search import SearchProblem, greedy
from string import ascii_lowercase, ascii_uppercase



def build_arg_parser():
    parser = ArgumentParser(description = 'Creates the input string using the greedy search algorithm')
    
    parser.add_argument('--input-string', dest = 'input_string', required = True, help = 'Input string')
    parser.add_argument('--initial-state', dest = 'initial_state', required = False, default = '', help = 'Starting point for the search')
    
    return parser


class CustomProblem(SearchProblem):
    def set_target(self, target_string):
        self.target_string = target_string
        
    def actions(self, cur_state):
        #Check the current state and take the right action
        if len(cur_state) < len(self.target_string):
            return list(ascii_lowercase + ' ' + ascii_uppercase)
        else:
            return []
        
    def result(self, cur_state, action):
        #Concatenate state and action to get result
        return cur_state + action
    
    def is_goal(self, cur_state):
        #Check if the goal is achieved
        return cur_state == self.target_string
    
    def heuristic(self, cur_state):
        #Define the heuristic that will be used - by comparing current string with target string
        dist = sum([1 if cur_state[i] != self.target_string[i] else 0 for i in range(len(cur_state))])
        
        #Difference between the lengths
        diff = len(self.target_string) - len(cur_state)
        
        return dist + diff
    
    

if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    
    problem = CustomProblem()
    
    problem.set_target(args.input_string)
    problem.initial_state = args.initial_state
    
    #Solve the problem
    output = greedy(problem)
    
    print(f'Target string: {args.input_string}')
    
    print('\nPath to the solution')
    for item in output.path():
        print(item)