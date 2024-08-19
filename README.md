# DeepRL-Fantasy-Drafter
Deep reinforcement learning project to draft players in an upcoming NFL fantasy draft

Author - Tarek Sallam
Info: Dependencies are listed in 'requirements.txt'
    To install Dependencies -> 'pip install -r requirements.txt'

Project Files:
    Agent.py: a class for monte carlo policy gradient   
        reinforcment learning agent with a deep neural network
        as a policy
    
    DraftBoard.py: a class that manages everything to do with
        the players available and picked in a draft simulation
        -Requires a npy file of adp data and a csv of the projected draft picks
    
    Environment.py: a class that extends the open ai gym
        environment class that acts as the environment in the training process. Handles the changing of state and rewards through stepping functions and resetting functions.

    modelTrain.py: main for training the model. Initiates the  
        environment and agent, and has an iteration loop to train the agent

    DraftMVC.py: model, view, and controller classes to display a 
        GUI for actually deploying the draft agent. This GUI allows for the initiation of a draft and the search and removal of players (for the other teams), as well as the picking from the agent 

    modelDeploy.py: main for deploying the model. Initiates the 
        MVC classes and sets up a root to display the GUI. For deploying the model for an actual NFL Fantasy Draft 