import numpy as np

class GridWorld:
    def __init__(self, size):
        self.size = size;
        self.reset();
        
    def reset(self):
        # row, col
        self.a_pos = np.array([0,0]);
        while True:
            self.g_pos = np.random.randint(0, self.size, size=2);
            if not np.array_equal(self.g_pos, self.a_pos):
                break;
        return self.get_state();
            
    def get_state(self):
        return np.concatenate([self.a_pos, self.g_pos]).astype(np.float32)
    
    def step(self, dec):
        #row change, col change
        moves = {
            0: np.array([-1, 0]), #up
            1: np.array([1, 0]), #down
            2: np.array([0, -1]), #left
            3: np.array([0,1]), #right
        }
        new_pos = self.a_pos + moves[dec];
        
        if 0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size:
            self.a_pos = new_pos;
            reward = -1 / self.size;
            done = False;
        else:
            reward = -5;
            done = False;
        
        if np.array_equal(self.a_pos, self.g_pos):
            reward = 4 * self.size;
            done = True;
        
        return self.get_state(), reward, done;

    def render(self):
        a = tuple(self.a_pos);
        g = tuple(self.g_pos);
        grid = {};
        
        if a == g:
            grid[a] = 'X';
        else:
            grid[a] = "A"
            grid[g] = "G"
        
        for i in range(self.size):
            print("".join(grid.get((i, j), ".") for j in range(self.size)));
        
        
