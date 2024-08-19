from DraftMVC import DraftGUIModel, DraftGUIController, DraftGUIView
import customtkinter as ctk
import os

root = ctk.CTk()
width = 800
height = 600
root.geometry(str(width) + "x" + str(height))
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")
view = DraftGUIView(root, width, height)
model = DraftGUIModel()
controller = DraftGUIController(model, view, 
                                os.path.join(os.getcwd(), 'data', 'projection_data.csv'), 
                                os.path.join(os.getcwd(), 'data', 'adp_data.npy'),
                                os.path.join(os.getcwd(), 'keras', 'fantasyDrafter.keras'))

root.mainloop()