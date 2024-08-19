from DraftBoard import DraftBoard
from Agent import PolicyGradientAgent
import os
import customtkinter as ctk
from tkinter import ttk

class DraftGUIView():
    def __init__(self, root, width, height):
        self.root = root
        self.width = width
        self.height = height
        root.title("Draft Board")

        ## create the menu page frame and display on the root
        self.menu_page = ctk.CTkFrame(root, width=width, height=height, fg_color="transparent")
        self.menu_page.place(relx=0.5, rely=0.5, anchor=ctk.CENTER)
        self.menu_page.columnconfigure(list(range(4)), minsize=width/4)
        self.menu_page.rowconfigure(list(range(5)), minsize=height/5)

        ## create the draft board page and do not display on the root
        self.draft_board_page = ctk.CTkFrame(root, width=width, height=height, fg_color="transparent")

        ## create the buttons and display onto the menu page
        self.create_draft_button = ctk.CTkButton(self.menu_page, text="Create Draft", width=0.1*width, height=0.05*width)
        self.create_draft_button.grid(column=2, row=4)

        ## create the inputs and display onto the menu page
        self.teams_input = ctk.CTkEntry(self.menu_page, justify="center", placeholder_text="", width=0.05*width, height=0.05*width)
        self.agent_pick_input = ctk.CTkEntry(self.menu_page, justify="center", placeholder_text="", width=0.05*width, height=0.05*width)
        self.rounds_input = ctk.CTkEntry(self.menu_page, justify="center", placeholder_text="", width=0.05*width, height=0.05*width)
        self.teams_input.grid(column=2, row=1)
        self.agent_pick_input.grid(column=2, row=2)
        self.rounds_input.grid(column=2, row=3)

        ## create the labels and display onto the menu page
        self.teams_label = ctk.CTkLabel(self.menu_page, text="Number of Teams: ")
        self.agent_pick_label = ctk.CTkLabel(self.menu_page, text="Agent's Pick: ")
        self.rounds_label = ctk.CTkLabel(self.menu_page, text="Number of Rounds: ")
        self.teams_label.grid(column=1, row=1)
        self.agent_pick_label.grid(column=1, row=2)
        self.rounds_label.grid(column=1, row=3)

        ## create the search label, input, and search and clear buttons and place onto draft board page
        self.search_label = ctk.CTkLabel(self.draft_board_page, text="Search: ")
        self.search_label.grid(row=0, column=0)
        
        self.search_input = ctk.CTkEntry(self.draft_board_page, placeholder_text="Name")
        self.search_input.grid(row=0, column=1)

        self.search_button = ctk.CTkButton(self.draft_board_page, text="Search")
        self.search_button.grid(row=0, column=2)

        self.clear_button = ctk.CTkButton(self.draft_board_page, text="Clear")
        self.clear_button.grid(row=0, column=3)

        self.select_player_button = ctk.CTkButton(self.draft_board_page, text="Select Player")
        self.select_player_button.grid(row=2, column=0)

        self.player_list = ttk.Treeview(self.draft_board_page, columns=(1, 2, 3), show="headings")
        self.player_list.grid(row=1, column = 0, columnspan=4)
        self.player_list.heading(1, text="Name")
        self.player_list.heading(2, text="Position")
        self.player_list.heading(3, text="Projected")

    def bind_create_draft_button(self, command):
        self.create_draft_button.configure(command=command)
    
    def get_teams_input(self):
        return self.teams_input.get()
    
    def get_agent_pick_input(self):
        return self.agent_pick_input.get()
    
    def get_rounds_input(self):
        return self.rounds_input.get()

    def bind_search_button(self, command):
        self.search_button.configure(command=command)

    def bind_clear_button(self, command):
        self.clear_button.configure(command=command)

    def bind_select_player_button(self, command):
        self.select_player_button.configure(command=command)

    def clear_search_input(self):
        self.search_input.delete(0, ctk.END)

    def get_search_input(self):
        return self.search_input.get()
    
    def get_selected_player_id(self):
        return self.player_list.focus()
    
    def get_player_info(self, player_id):
        player = self.player_list.item(player_id)['values']
        player = {'display': player[0], 'position': player[1], 'proj': player[2]}
        return player
    
    def switch_to_draft_view(self):
        self.menu_page.destroy()
        self.draft_board_page.place(relx=0.5, rely=0.5, anchor=ctk.CENTER) ## place the draft board page

class DraftGUIModel():
    def __init__(self):
        pass

    def create_draft_board(self, teams, agent_pick, rounds, projection_data, adp_data, keras_model):
        self.draftboard = DraftBoard(teams=teams,agent_pick=agent_pick,rounds=rounds,is_training=False,
                                     projection_data_path=projection_data, adp_data_path=adp_data)
        self.keras_model = PolicyGradientAgent(epsilon=0)
        self.keras_model.load_model(keras_model)
        self.players_df = self.draftboard.get_players_df('')

    def remove_player(self, player):
        full_player = self.draftboard.getFullPlayer(player)
        self.draftboard.removePlayerByInfo(player)
        self.players_df = self.draftboard.get_players_df('')

class DraftGUIController():
    def __init__(self, model, view, projection_data, adp_data, keras_model):
        self.model = model
        self.view = view
        self.projection_data = projection_data
        self.adp_data = adp_data
        self.keras_model = keras_model
        view.bind_create_draft_button(command=self.create_draft_board)
        view.bind_search_button(command=self.search_button)
        view.bind_clear_button(command=self.clear_button)
        view.bind_select_player_button(command=self.make_selection)

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
    
    def search_button(self):
        query = self.view.get_search_input().strip()

    def clear_button(self):
        self.view.clear_search_input()
        
    def make_selection(self):
        id = self.view.get_selected_player_id()
        player = self.view.get_player_info(id)
        self.model.remove_player(player)
        self.view.player_list.delete(id)
        print(self.model.players_df)

    def switch_to_draft_view(self):
        self.view.switch_to_draft_view()
        for player in self.model.players_df.to_numpy().tolist():
            self.view.player_list.insert('', ctk.END, values=player)