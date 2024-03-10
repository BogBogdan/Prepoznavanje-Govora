import tkinter as tk
import random
import tkinter as tk

class DrawingApp:
    def __init__(self, width=400, height=400):
        self.canvas = tk.Canvas(width=width, height=height)
        self.canvas.pack()

        self.entry = tk.Entry()
        self.entry.pack()
        self.entry.bind("<Return>", self.process_command)

        self.current_shape = None

    def process_command(self, string, event=None):
        command = string
        self.entry.delete(0, tk.END)
        print(command)
        if command == "kvadrat":
            self.draw_square()
        elif command == "krug":
            self.draw_circle()
        elif command == "trougao":
            self.draw_circle()
        elif command == "izbrisi":
            self.clear_canvas()
        elif command == "oboji":
            self.color_random()

    def draw_square(self):
        self.clear_canvas()
        self.current_shape = self.canvas.create_rectangle(100, 100, 300, 300, fill="", outline="black")

    def draw_circle(self):
        self.clear_canvas()  # Pretpostavljam da ova funkcija čisti platno
        self.current_shape = self.canvas.create_oval(50, 150, 350, 300, fill="", outline="black")


    def draw_triangle(self):
        self.clear_canvas()
        self.current_shape = self.canvas.create_polygon(200, 100, 300, 300, 100, 300, fill="", outline="black")

    def clear_canvas(self):
        self.canvas.delete("all")
        self.current_shape = None

    def color_random(self):
        if self.current_shape:
            random_color = "#%06x" % random.randint(0, 0xFFFFFF)
            self.canvas.itemconfig(self.current_shape, fill=random_color)

