from DraftBoard import DraftBoard
import tensorflow as tf
import os
import numpy as np
import customtkinter as ctk
from tkinter import ttk

## The View portion of the Model View Controller
class DraftGUIView():

    ## init all the stuff that goes on the pages
    def __init__(self, root, width, height):
        self.root = root ## set the root
        self.width = width ## set the width
        self.height = height ## set the height
        root.title("Draft Board") ## set the title of the window

        ## create the menu page frame and display on the root
        self.menu_page = ctk.CTkFrame(root, width=width, height=height, fg_color="transparent")
        self.menu_page.place(relx=0.5, rely=0.5, anchor=ctk.CENTER)
        self.menu_page.columnconfigure(list(range(4)), minsize=width/4)
        self.menu_page.rowconfigure(list(range(5)), minsize=height/5)
    

        ## create the draft board page and do not display on the root
        self.draft_board_page = ctk.CTkFrame(root, width=width, height=height, fg_color="transparent")
        self.draft_board_page.rowconfigure([0, 1, 3], minsize=height/9)
        self.draft_board_page.rowconfigure(2, minsize=(height*2)/3)
        self.draft_board_page.columnconfigure(list(range(6)), minsize=width/6)


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

        ## creates the labels for the draft board page and places onto the page
        self.round_label = ctk.CTkLabel(self.draft_board_page, text="", width = width/6, height = height / 18)
        self.pick_label = ctk.CTkLabel(self.draft_board_page, text="", width = width/6, height = height / 18)
        self.agent_pick_label = ctk.CTkLabel(self.draft_board_page, text="", width=width/6, height = height / 18)
        self.round_label.grid(row=0, column = 1)
        self.pick_label.grid(row=0, column = 2)
        self.agent_pick_label.grid(row=0, column = 3)

        ## creates the search input, search button, clear button, select player button and 
            # calculate player buttons and places them onto the page
        self.search_input = ctk.CTkEntry(self.draft_board_page, placeholder_text="Name", width = 2 * width / 6, height = height / 18)
        self.search_input.grid(row=1, column=1, columnspan=2)

        self.search_button = ctk.CTkButton(self.draft_board_page, text="Search", width = width/6, height = height / 18)
        self.search_button.grid(row=1, column=3)

        self.clear_button = ctk.CTkButton(self.draft_board_page, text="Clear", width = width/6, height = height / 18)
        self.clear_button.grid(row=1, column=4)

        self.select_player_button = ctk.CTkButton(self.draft_board_page, text="Select Player", width = width/6, height = height / 18)
        self.select_player_button.grid(row=3, column=4)

        self.calculate_player_button = ctk.CTkButton(self.draft_board_page, text="Calculate Selection", width = width/6, height = height / 18)
        self.calculate_player_button.grid(row=3, column=3)

        self.undo_button = ctk.CTkButton(self.draft_board_page, text="Undo", width = width/6, height = height / 18)
        self.undo_button.grid(row=3, column = 2)

        ## create a ttk style and configre the treeviews to have a certain row height
        s=ttk.Style()
        s.configure('Treeview', rowheight=(2 * height / 3) / 20)

        ## create the treeview to display the players, add the headings, and place onto the draft board page
        self.player_list = ttk.Treeview(self.draft_board_page, columns=(1, 2, 3), show="headings", height=15)
        self.player_list.grid(row=2, column = 1, columnspan=4)
        self.player_list.heading(1, text="Name")
        self.player_list.heading(2, text="Position")
        self.player_list.heading(3, text="Projected")

    ## binders for the buttons
    def bind_create_draft_button(self, command):
        self.create_draft_button.configure(command=command)
    
    def bind_clear_button(self, command):
        self.clear_button.configure(command=command)

    def bind_select_player_button(self, command):
        self.select_player_button.configure(command=command)

    def bind_calculate_player_button(self, command):
        self.calculate_player_button.configure(command=command)

    def bind_undo_button(self, command):
        self.undo_button.configure(command=command)

    def bind_search_button(self, command):
        self.search_button.configure(command=command)

    ## getters for the inputs 
    def get_teams_input(self):
        return self.teams_input.get()
    
    def get_agent_pick_input(self):
        return self.agent_pick_input.get()
    
    def get_rounds_input(self):
        return self.rounds_input.get()
    
    def get_search_input(self):
        return self.search_input.get()

    ## clears the search input
    def clear_search_input(self):
        self.search_input.delete(0, ctk.END)
    
    ## gets the selected players id in the tree view
    def get_selected_player_id(self):
        return self.player_list.focus()
    
    ## gets a players information from the treeview given an id
    def get_player_info(self, player_id):
        player = self.player_list.item(player_id)['values']
        player = {'display': player[0], 'position': player[1], 'proj': player[2]}
        return player
    
    ## function that removes the menu page and adds the draft board page onto the root
    def switch_to_draft_view(self):
        self.menu_page.destroy()
        self.draft_board_page.place(relx=0.5, rely=0.5, anchor=ctk.CENTER) ## place the draft board page

## Model portion of the MVC
class DraftGUIModel():

    def __init__(self):
        self.current_pick = 1 ## set the current pick
        self.current_round = 1 ## set the current round

        ## histories in case of undoing
        self.history = [] 
        self.roster_history = []

    ## creates the draft board class, loads the queried df, and creates the roster (for the state)
    def create_draft_board(self, teams, agent_pick, rounds, projection_data, adp_data, keras_model):
        self.teams = teams
        self.rounds = rounds
        self.agent_pick = agent_pick
        self.draftboard = DraftBoard(teams=teams,agent_pick=agent_pick,rounds=rounds,is_training=False,
                                     projection_data_path=projection_data, adp_data_path=adp_data)
        self.agent = tf.keras.models.load_model(keras_model)
        self.players_df = self.draftboard.get_players_df('')
        self.roster = [0] * 9

    ## removes a player from the draftboard adds them to the history, and increments the pick
    def remove_player(self, player):
        full_player = self.draftboard.getFullPlayer(player)
        self.draftboard.removePlayerByInfo(player)
        self.players_df = self.draftboard.get_players_df('')
        self.increment_pick()
        self.history.append(full_player)

    ## increments the pick (or round)
    def increment_pick(self):
        self.current_pick +=1
        if (self.current_pick > self.teams):
            self.current_pick = 1
            self.current_round += 1

    ## decrements the pick (or round)
    def decrement_pick(self):
        if self.current_round == 1 and self.current_pick == 1:
            return
        self.current_pick -=1
        if self.current_pick < 1:
            self.current_pick = 1
            self.current_round -=1

    ## function that adds a player to the roster list
    def addToRoster(self, position):

        ## get the roster
        roster = self.roster

        ## get a list of the player information
        player = self.draftboard.getPlayer(position, 0).to_list()

        ## if it is a RB or WR
        if position == 1 or position == 2:

            ## if the RB1 or WR1 is open then set it to 1 to fill it, and append the position to the list
            if roster[position] == 0:
                roster[position] = 1
                self.roster_history.append(position)
            
            ## if the RB2 or WR2 is open then set it to 1 to fill it, and append the position to the list
            elif roster[position + 5] == 0:
                roster[position + 5] = 1
                self.roster_history.append(position + 5)
            ## if the WR/RB slot it open then set it to 1 to fill it, and append the position to the list
            elif roster[8] == 0:
                roster[8] = 1
                self.roster_history.append(8)
            else:
                self.roster_history.append(-1)
        
        ## otherwise for QB, TE, K, DEF, check if the spot has been filled, if it has fill it, and append the position to the list
        else:
            if roster[position] == 0:
                roster[position] = 1
                self.roster_history.append(position)
            else:
                self.roster_history.append(-1)
            
        self.roster = roster
            
    ## make the pick based on the keras model's calculations
    def make_pick(self):
        state = []
        state.extend(self.roster)
        state.extend(self.draftboard.get_top_projections())
        print("State is: ")
        print(state)
        actionProbs = self.agent.predict(np.array([state]))[0]
        action = np.argmax(actionProbs)
        print(self.draftboard.getPlayer(action, 0))

    ## removes the player from the history, and adds them back into the draftboard and decrements the pick
    ## also removes the player from the agent's roster in case it was their pick to revert the state
    def undo(self):
        if not (self.current_pick == 1 and self.current_round == 1):
            self.decrement_pick()
            last_player = self.history.pop()
            self.draftboard.undo(last_player)
            if self.is_agent_pick():
                roster_slot = self.roster_history.pop()
                if (roster_slot != -1) {
                    self.roster[roster_slot] = 0
                }
            self.players_df = self.draftboard.get_players_df('')

    ## checks if it is currently the agents pick (given the snake draft)
    def is_agent_pick(self):
        if self.current_round % 2 == 1 and self.agent_pick == self.current_pick:
                return True
        elif self.current_round % 2 == 0 and self.rounds - self.agent_pick + 1 == self.current_pick:
                return True
        else:
            return False

## controller portion of the MVC
class DraftGUIController():

    # set the attributes and bind the functions
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
        view.bind_calculate_player_button(command=self.agent_pick)
        view.bind_undo_button(command=self.undo)

    ## function that creates the draft board portion of the view and the corresponding portion of the model
    def create_draft_board(self):

        ## get the inputs
        teams = self.view.get_teams_input()
        agent_pick = self.view.get_agent_pick_input()
        rounds = self.view.get_rounds_input()
        
        ## if there are any errors print them, else create the draft board and switch the view
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
    
    ## get the query, clear the search input, get the updated query dataframe
    ## delete all children of the tree view and add the dataframe into the tree view
    def search_button(self):
        query = self.view.get_search_input().strip().upper().replace(' ', '')
        self.view.clear_search_input()
        self.model.players_df = self.model.draftboard.get_players_df(query)
        for i in self.view.player_list.get_children():
            self.view.player_list.delete(i)
        for player in self.model.players_df.to_numpy().tolist():
            self.view.player_list.insert('', ctk.END, values=player)
        
    ## clears the query. then deletes the children of the tree view and adds all the players back
    ## into the treeview
    def clear_button(self):
        self.view.clear_search_input()
        self.model.players_df = self.model.draftboard.get_players_df('')
        for i in self.view.player_list.get_children():
            self.view.player_list.delete(i)

        for player in self.model.players_df.to_numpy().tolist():
            self.view.player_list.insert('', ctk.END, values=player)
        
    ## get the selected player, if it is currently the agent's pick update the state
    ## then remove the player from the treeview and from the model. Update the labels
    def make_selection(self):
        id = self.view.get_selected_player_id()
        positions = ['QB', 'RB', 'WR', 'TE', 'K', 'DEF']
        if (not id == ''):
            player = self.view.get_player_info(id)
            if self.model.is_agent_pick():
                self.model.addToRoster(positions.index(player['position']))
                
            self.model.remove_player(player)
            self.view.player_list.delete(id)
            self.update_labels()
        
    ## calculate pick button handler, call the models calculation to print it out
    def agent_pick(self):
        self.model.make_pick()
        
    ## undo button handler, call undo on the model, clear the query, update the labels
    def undo(self):
        self.model.undo()
        self.clear_button()
        self.update_labels()

    ## updates the round, pick and is agent pick labels corresponding to the models state
    def update_labels(self):
        self.view.round_label.configure(text="Round: " + str(self.model.current_round))
        self.view.pick_label.configure(text="Pick: " + str(self.model.current_pick))
        if self.model.is_agent_pick():
            self.view.agent_pick_label.configure(text="Agent's Pick")
        else:
            self.view.agent_pick_label.configure(text="Not Agent's Pick")

    ## switches from the first page to the draft board page by switching the view, and updating the model
    def switch_to_draft_view(self):
        self.view.switch_to_draft_view()
        for player in self.model.players_df.to_numpy().tolist():
            self.view.player_list.insert('', ctk.END, values=player)
        self.update_labels()
        
