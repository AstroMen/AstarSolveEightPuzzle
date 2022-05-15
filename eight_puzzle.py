import math
import time
import random
import heapq
from heapq import heappush
from copy import deepcopy


class TreeNode(object):
    def __init__(self, val, parent=None):
        """
        Initialize an tree node with value
        :param val: the puzzle state
        :param parent: the parent node
        """
        super(TreeNode, self).__init__()
        self.val = val
        self.parent = parent
        self.child = list()
        self.visited = False

    def __lt__(self, other):
        """
        Customized comparison functions
        :param other: the other tree node
        :return: which one is smaller. if current one is smaller, return True; otherwise, return False
        """
        if self.val < other.val:
            return True
        else:
            return False


def generate_puzzle(size):
    """
    Generate an initial state of puzzle
    :param size: the size of two-dimensional puzzle
    :return: an initial state
    """
    x = [i for i in range(size * size)]
    random.shuffle(x)
    return x


def generate_goal(size):
    """
    Generate a goal state of puzzle based on the size
    :param size: the size of two-dimensional puzzle
    :return: the goal state
    """
    goal_arr = [i for i in range(1, size * size)]
    return goal_arr + [0]


def get_inversion_num(arr):
    """
    Get the number of inversion for an array
    :param arr: an array
    :return: the number of inversion
    """
    inversion = 0
    for i in range(1, len(arr)):
        for j in range(i):
            if arr[j] < arr[i]:
                inversion += 1
    return inversion


def is_valid_puzzle(init_arr, goal_arr):
    """
    Determine if the current initial state could get the solution to goal state.
    :param init_arr: the array of initial state
    :param goal_arr: the array of goal state
    :return: is valid or not
    """
    init_arr.remove(0)
    goal_arr.remove(0)
    return get_inversion_num(init_arr) % 2 == get_inversion_num(goal_arr) % 2


def is_safe_move(x, y, N):
    """
    Determine if the current move is legal
    :param x: x-axis value of coordinates
    :param y: y-axis value of coordinates
    :param N: the size of two-dimensional puzzle
    :return: is valid or not
    """
    return x >= 0 and x < N and y >= 0 and y < N


def get_corr_by_arr_index(idx, arr_size):
    """
    Get the two-dimensional puzzle coordinates corresponding to the index in the array.
    :param idx: the index in the array
    :param arr_size: the size of array
    :return: the two-dimensional puzzle coordinates
    """
    size = int(math.sqrt(arr_size))
    column = idx % size
    row = int(idx / size)
    return row, column


def get_arr_index_by_corr(x, y, size):
    """
    Get the index in the array corresponding to the two-dimensional puzzle coordinates.
    :param x: x-axis value of coordinates
    :param y: y-axis value of coordinates
    :param size: the size of two-dimensional puzzle
    :return: the index in the one-dimensional array
    """
    return x * size + y


def move(arr, direction):
    """
    Do a move operation on puzzle
    :param arr: the array of a state of puzzle
    :param direction: move direction (four options: left, right, up, down)
    :return: the state after moving
    """
    size = int(math.sqrt(len(arr)))
    index_blank = arr.index(0)
    corr_blank = (get_corr_by_arr_index(index_blank, len(arr)))
    operation = {
        'left': (0, -1),
        'right': (0, 1),
        'up': (-1, 0),
        'down': (1, 0)
    }
    ops_offset = operation[direction]
    corr_target = (corr_blank[0] + ops_offset[0], corr_blank[1] + ops_offset[1])
    if not is_safe_move(corr_target[0], corr_target[1], size):
        return None

    index_target = get_arr_index_by_corr(corr_target[0], corr_target[1], size)
    arr[index_blank], arr[index_target] = arr[index_target], arr[index_blank]
    return arr


def get_h_of_misplaced_tiles(cur_state, goal_state):
    """
    Get h(n) value when using misplaced tiles algorithm
    :param cur_state: current state of puzzle
    :param goal_state: goal state of puzzle
    :return: h(n) of current state
    """
    diff_cnt = 0
    for i in range(len(cur_state)):
        if cur_state[i] != goal_state[i] and cur_state[i] > 0:
            diff_cnt += 1
    return diff_cnt


def get_h_of_manhattan_distance(cur_state, goal_state):
    """
    Get h(n) value when using manhattan distance algorithm
    :param cur_state: current state of puzzle
    :param goal_state: goal state of puzzle
    :return: h(n) of current state
    """
    sum_diff_step = 0
    for i in range(len(cur_state)):
        if cur_state[i] != goal_state[i] and cur_state[i] > 0:
            idx_goal = goal_state.index(cur_state[i])
            cur_corr = get_corr_by_arr_index(i, len(cur_state))
            goal_corr = get_corr_by_arr_index(idx_goal, len(goal_state))
            sum_diff_step += int(math.fabs(cur_corr[0] - goal_corr[0]) + math.fabs(cur_corr[1] - goal_corr[1]))
    return sum_diff_step


def get_h_of_uniform_cost_search(cur_state, goal_state):
    """
    Get h(n) value when using uniform cost search algorithm
    :param cur_state: current state of puzzle
    :param goal_state: goal state of puzzle
    :return: 0
    """
    return 0


def heuristic_search(initial_state, goal_state, method='misplaced_tile'):
    """
    Do the search algorithm, use heap to store the TreeNode of state, use set to record the repeated state.
    :param initial_state: the array of initial puzzle
    :param goal_state: the array of goal puzzle
    :param method: the search algorithm name
    :return: If find the goal state, return the TreeNode of goal state. Otherwise, return None
    """
    total_step = 0
    repeated_hist = set()
    full_operation = ('right', 'down', 'up', 'left')
    __ops_method_forks = {
        'uniform_cost_search': get_h_of_uniform_cost_search,
        'misplaced_tile': get_h_of_misplaced_tiles,
        'manhattan_distance': get_h_of_manhattan_distance,
    }

    root = TreeNode(initial_state)
    state_heap = list()
    heappush(state_heap, (0, 0, root))
    while len(state_heap) > 0:
        cur_f_val, cur_g_val, cur_node = heapq.heappop(state_heap)
        cur_node.visited = True
        cur_arr = cur_node.val

        if str(cur_arr) in repeated_hist:
            continue

        repeated_hist.add(str(cur_arr))
        # print('{}: {}'.format(total_step, [cur_arr[0:3], cur_arr[3:6], cur_arr[6:9]]))
        g_val = cur_g_val + 1
        for ops in full_operation:
            res_arr = move(deepcopy(cur_arr), ops)
            if res_arr is None or str(res_arr) in repeated_hist:
                continue

            new_node = TreeNode(res_arr, parent=cur_node)
            cur_node.child.append(new_node)
            total_step += 1
            if res_arr == goal_state:
                new_node.visited = True
                return new_node, total_step
            h_val = __ops_method_forks[method](res_arr, goal_state)
            heappush(state_heap, (g_val + h_val, g_val, new_node))
    return None, total_cases


def beauty_print_puzzle(arr):
    """
    Beauty the printing puzzle
    :param arr: the array of one puzzle, store by row
    :return: None, print one puzzle like a matrix
    """
    arr_size = len(arr)
    size = int(math.sqrt(arr_size))
    print('  ┏' + '━━━┳' * (size - 1) + '━━━┓')
    for i in range(len(arr)):
        left_placeholder = '' if arr[i] > 99 else ' '
        right_placeholder = ' ' if arr[i] < 10 else ''
        if (i+1) % size == 0 and (i+1) != arr_size:  # rightmost
            print("{}{}{}┃".format(left_placeholder, arr[i], right_placeholder))
            print('  ┣' + '━━━╋' * (size - 1) + '━━━┫')
        elif (i+1) == arr_size:  # last line, rightmost
            print("{}{}{}┃".format(left_placeholder, arr[i], right_placeholder))
        elif i % size == 0:  # leftmost
            print('  ┃{}{}{}'.format(left_placeholder, arr[i], right_placeholder), end="┃")
        else:  # center
            print('{}{}{}'.format(left_placeholder, arr[i], right_placeholder), end="┃")
    print('  ┗' + '━━━┻' * (size - 1) + '━━━┛')


def print_path(leaf_node, goal_state, method='manhattan_distance'):
    """
    Print the expand path of state
    :param leaf_node: the TreeNode of goal state
    :param goal_state: the array of goal state
    :param method: the search algorithm name
    :return: None, print the g(n) and h(n) of every expanding puzzle
    """
    res_path = list()
    while leaf_node:
        res_path.append(leaf_node.val)
        leaf_node = leaf_node.parent
    res_path.reverse()

    depth = 0
    __ops_method_forks = {
        'uniform_cost_search': get_h_of_uniform_cost_search,
        'misplaced_tile': get_h_of_misplaced_tiles,
        'manhattan_distance': get_h_of_manhattan_distance,
    }
    for node in res_path:
        print('{}, g(n) = {}, h(n) = {}'.format(node, depth, __ops_method_forks[method](node, goal_state)))
        depth += 1
        beauty_print_puzzle(node)
    print('Depth: {}'.format(depth - 1))


if __name__ == "__main__":
    # Input or generate initial puzzle
    size_puzzle = int(input('Please input the dimension of puzzle matrix: ').strip())
    is_generate = input('Do you want to generate puzzle automatically? (YES/NO/DEFAULT) Otherwise, you could input specific puzzle. ').strip()
    if is_generate.upper() == 'YES' or is_generate.upper() == 'Y':
        initial_state = generate_puzzle(size_puzzle)
    elif is_generate.upper() == 'NO' or is_generate.upper() == 'N':
        init_str = input('Please input puzzle you want by row (separated by commas, press Enter to end the input): ')
        initial_state = [int(x.strip()) for x in init_str.split(',')]
    else:
        # initial_state = [4, 1, 2, 5, 3, 0, 7, 8, 6]
        # initial_state = [1, 2, 3, 4, 0, 6, 7, 5, 8]
        # # initial_state = [1, 6, 7, 5, 0, 3, 4, 8, 2]
        initial_state = [0, 7, 2, 4, 6, 1, 3, 5, 8]
        # initial_state = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 0, 14, 15]
    print('Initial state: ')
    beauty_print_puzzle(initial_state)

    # Generate goal puzzle
    goal_state = generate_goal(size_puzzle)
    print('Goal state: ')
    beauty_print_puzzle(goal_state)

    # Verify that the initial puzzle has a solution, if not, exit program
    valid_flag = is_valid_puzzle(deepcopy(initial_state), deepcopy(goal_state))
    if not valid_flag:
        print('Invalid initial puzzle, no solution.')
        exit(0)

    # Choose the search algorithm
    algorithm = int(input("Please select an algorithm to search (1/2/3):" + '\n' + "1. uniform cost search 2. misplaced tile heuristic 3. manhattan distance heuristic." + '\n'))
    search_mode = ('uniform_cost_search', 'misplaced_tile', 'manhattan_distance')

    # Do the search operation
    time_start = time.time()
    goal_tree_node, total_cases = heuristic_search(initial_state, goal_state, method=search_mode[algorithm-1])
    time_end = time.time()
    if goal_tree_node:
        print_path(goal_tree_node, goal_state, method=search_mode[algorithm-1])
        print('Find, expand node: {}'.format(total_cases))
    print('Search running time: {}'.format(time_end - time_start))
