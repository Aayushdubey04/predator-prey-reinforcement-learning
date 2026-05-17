import random

actions = {
    0: (0, 0),
    1: (-1, 0),
    2: (1, 0),
    3: (0, -1),
    4: (0, 1)
}

def move(pos, action, N):
    x, y = pos
    dx, dy = action
    nx, ny = x + dx, y + dy
    
    if 1 <= nx <= N and 1 <= ny <= N:
        return (nx, ny)
    return (x, y)

def simulator(N, predator_pos, prey_pos, action):
    predator_next = move(predator_pos, actions[action], N)
    
    possible_moves = [move(prey_pos, a, N) for a in actions.values()]
    prey_next = random.choice(possible_moves)
    
    if predator_next == prey_next:
        reward = 1
        
        all_cells = [(i, j) for i in range(1, N+1) for j in range(1, N+1)]
        all_cells.remove(predator_next)
        prey_next = random.choice(all_cells)
    else:
        reward = 0
    
    return predator_next, prey_next, reward