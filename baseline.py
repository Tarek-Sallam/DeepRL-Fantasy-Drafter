from Environment import DraftEnv
import os
import numpy as np

env = DraftEnv(12, 15, os.path.join(os.getcwd(), 'data', 'projection_data.csv'), os.path.join(os.getcwd(), 'data', 'adp_data.npy'))

iterations = 200
pt_history = []

for iteration in range(iterations):
    env.reset()
    positions = [0, 1, 2, 3, 1, 2, [1, 2]]
    totalPts = 0
    while positions:
        proj = env.draftBoard.get_top_projections()
        pos = positions[np.random.randint(0, len(positions))]
        positions.remove(pos)
        if isinstance(pos, int):
            totalPts += proj[pos]
            env.step(pos)
        else:
            pos = pos[np.random.randint(0, 2)]
            totalPts += proj[pos]
            env.step(pos)

    pt_history.append(totalPts)

baseline = np.mean(pt_history)
print(baseline) ## 4.847968 after 1000 iterations
     

        
