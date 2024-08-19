from DraftBoard import DraftBoard
from Agent import PolicyGradientAgent
import os
import customtkinter as ctk

class DraftGUIView():
    def __init__(self, root, width, height):
        self.root = root
        self.width = width
        self.height = height
        root.title("Draft Board")
        self.menu_page = ctk.CTkFrame(root, width=width, height=height, fg_color="transparent")
        self.menu_page.grid(row=0, column=0)
        self.draft_board_page = ctk.CTkFrame(root, width=width, height=height, fg_color="transparent")

        self.create_draft_button = ctk.CTkButton(self.menu_page, text="Create Draft", width=0.1*width, height=0.05*width)
        self.create_draft_button.pack(pady=10)
        self.teams_input = ctk.CTkEntry(self.menu_page, placeholder_text="", width=0.05*width, height=0.05*width)
        self.agent_pick_input = ctk.CTkEntry(self.menu_page, placeholder_text="", width=0.05*width, height=0.05*width)
        self.rounds_input = ctk.CTkEntry(self.menu_page, placeholder_text="", width=0.05*width, height=0.05*width)
        self.teams_input.pack(pady=10)
        self.agent_pick_input.pack(pady=10)
        self.rounds_input.pack(pady=10)
        
    def bind_create_draft_button(self, command):
        self.create_draft_button.configure(command=command)
    
    def get_teams_input(self):
        return self.teams_input.get()
    
    def get_agent_pick_input(self):
        return self.agent_pick_input.get()
    
    def get_rounds_input(self):
        return self.rounds_input.get()
    
    def clear_input_values(self):
       self.teams_input.delete(0, ctk.END)
       self.agent_pick_input.delete(0, ctk.END)
       self.rounds_input.delete(0, ctk.END)

    def switch_to_draft_view(self):
        self.menu_page.grid_forget()
        self.draft_board_page.grid(row=0, column=0)

class DraftGUIModel():
    def __init__(self):
        pass

    def create_draft_board(self, teams, agent_pick, rounds, projection_data, adp_data, keras_model):
        self.draftboard = DraftBoard(teams=teams,agent_pick=agent_pick,rounds=rounds,is_training=False,
                                     projection_data_path=projection_data, adp_data_path=adp_data)
        self.keras_model = PolicyGradientAgent(epsilon=0)
        self.keras_model.load_model(keras_model)

class DraftGUIController():
    def __init__(self, model, view, projection_data, adp_data, keras_model):
        self.model = model
        self.view = view
        self.projection_data = projection_data
        self.adp_data = adp_data
        self.keras_model = keras_model
        view.bind_create_draft_button(command=self.create_draft_board)

    def create_draft_board(self):
        teams = self.view.get_teams_input()
        agent_pick = self.view.get_agent_pick_input()
        rounds = self.view.get_rounds_input()
        
        try:
            teams = int(teams)
            agent_pick = int(agent_pick)
            rounds = int(rounds)
        except:
            print("At least one of the inputs was not an integer.")
        else:
            if agent_pick > teams or agent_pick < 1:
                print("The agent's pick is out of bounds")
            elif rounds > 20 or rounds < 10:
                print("Number of Rounds is out of bounds")
            elif teams < 6 or teams > 32:
                print("Number of Teams is out of bounds")
            else:
                self.model.create_draft_board(teams, agent_pick, rounds, self.projection_data, self.adp_data, self.keras_model)
                self.switch_to_draft_view()
                
    def switch_to_draft_view(self):
        self.view.switch_to_draft_view()