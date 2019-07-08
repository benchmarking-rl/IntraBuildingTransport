from time import sleep
import threading
import pyglet
from pyglet.gl import *

pyglet.resource.path = ['./resources']
pyglet.resource.reindex()

game_window = pyglet.window.Window(resizable=True, visible=False)
man_image, background, line, elevator, circle, up, down = None, None, None, None, None, None, None
level_label, up_label, down_label = None, None, None

line_batch = None
line_ele = []
elevator_batch = []
elevator_ele = []
waiting_people_batch = None
waiting_people_ele = []

elevatorNumber = None
numberOfFloor = None

class Courier(threading.Thread):

    def __init__(self, shared, *args, **kwargs):
        super(Courier,self).__init__(*args, **kwargs)
        global elevatorNumber, numberOfFloor
        self.shared_mansion = shared
        self.floor_height = int(self.shared_mansion.attribute.FloorHeight)
        elevatorNumber = self.shared_mansion.attribute.ElevatorNumber 
        numberOfFloor = self.shared_mansion.attribute.NumberOfFloor
        self.screen_x = elevatorNumber * 100 + 600
        self.screen_y = (numberOfFloor) * 25 * self.floor_height + 80 # 20 for 1m
        self.create_window()

    def run(self):
        pyglet.clock.schedule_interval(self.update, 1/100)
        pyglet.app.run()

    def create_window(self):
        game_window.set_size(self.screen_x, self.screen_y)
        game_window.set_visible()
        self.load_images()
        self.init_batch()
    
    def center_image(self, image):
        image.anchor_x = image.width//2
        image.anchor_y = image.height//2
    
    def load_images(self):
        global man_image, background, line, elevator, circle, up, down
        global level_label, up_label, down_label
        man_image = pyglet.resource.image('matchstick_man.png')
        background = pyglet.resource.image("white_bg.png")
        line = pyglet.resource.image("line.png")
        elevator = pyglet.resource.image("elevator.png")
        circle = pyglet.resource.image("circle.png")
        up = pyglet.resource.image("up.png")
        down = pyglet.resource.image("down.png")

        # modify the images
        elevator.width, elevator.height = 60, 90
        self.center_image(elevator)
        # elevator = pyglet.sprite.Sprite(img = elevator, x=200, y=65)

        line.width, line.height = self.screen_y, 10
        self.center_image(line)
        # line = pyglet.sprite.Sprite(img = line, x=400, y=110)

        background.width, background.height = self.screen_x, self.screen_y
        self.center_image(background)
        background = pyglet.sprite.Sprite(img = background, x=self.screen_x//2, y=self.screen_y//2)

        man_image.width, man_image.height = 40, 60
        self.center_image(man_image)
        # man_image = pyglet.sprite.Sprite(img = man_image, x=50, y=50)

        circle.width, circle.height = 15, 15
        self.center_image(circle)
        # circle = pyglet.sprite.Sprite(img = circle, x=200, y=65)

        up.width, up.height = 30, 30
        self.center_image(up)

        down.width, down.height = 30, 30
        self.center_image(down)

        # text
        up_label = pyglet.text.Label(text = "Going Up", x=self.screen_x//2-200, y=self.screen_y-50, anchor_x='center', color=(0,0,0,255))
        down_label = pyglet.text.Label(text = "Going Down", x=self.screen_x//2+200, y=self.screen_y-50, anchor_x='center', color=(0,0,0,255))
        level_label = pyglet.text.Label(text = "Elevator Simulator", x=self.screen_x//2, y=self.screen_y-30, anchor_x='center', color=(0,0,0,255))

    def init_batch(self):
        global line_batch, elevator_batch, waiting_people_batch, line_ele
        global line
        global elevatorNumber, numberOfFloor

        line_batch = pyglet.graphics.Batch()
        waiting_people_batch = pyglet.graphics.Batch()
        for i in range(numberOfFloor):
            elevator_batch.append(pyglet.graphics.Batch())
        for i in range(numberOfFloor):
            line_ele.append(pyglet.sprite.Sprite(img = line, x=self.screen_x//2, y=25*self.floor_height*i+100, batch = line_batch))

    def update(self, dt):
        global elevator_batch, waiting_people_batch
        global elevator_ele, waiting_people_ele
        global man_image, elevator, up, down
        global elevatorNumber, numberOfFloor

        waiting_up, waiting_down = self.shared_mansion.waiting_queue
        waiting_people_ele = []
        for i in range(numberOfFloor):
            for j in range(len(waiting_up[i])):
                waiting_people_ele.append(pyglet.sprite.Sprite(img = man_image, x=300-30*j, y=25*self.floor_height*i+30, batch = waiting_people_batch))
            for j in range(len(waiting_down[i])):
                waiting_people_ele.append(pyglet.sprite.Sprite(img = man_image, x=self.screen_x-300+30*j, y=25*self.floor_height*i+30, batch = waiting_people_batch))

        elevator_ele = []
        for i in range(elevatorNumber):
            elevator_floor = self.shared_mansion.state.ElevatorStates[i].Floor
            elevator_ele.append(pyglet.sprite.Sprite(img = elevator, 
                            x=350+i*100, y=elevator_floor*self.floor_height*25-50, batch = elevator_batch[i]))
            if self.shared_mansion._elevators[i]._direction == 1:
                elevator_ele.append(pyglet.sprite.Sprite(img = up, x=325+i*100, y=elevator_floor*self.floor_height*25-50, batch = elevator_batch[i]))
            elif self.shared_mansion._elevators[i]._direction == -1:
                elevator_ele.append(pyglet.sprite.Sprite(img = down, x=325+i*100, y=elevator_floor*self.floor_height*25-50, batch = elevator_batch[i]))        
            for j in range(self.shared_mansion.loaded_people[i]):
                elevator_ele.append(pyglet.sprite.Sprite(img = circle, 
                            x=320+i*100+(j%3+1)*15, y=20-(j//3)*30+elevator_floor*self.floor_height*25-50, batch = elevator_batch[i]))

@game_window.event
def on_draw():
    global man_image, background, line, elevator, circle
    global level_label, up_label, down_label
    global line_batch
    global elevatorNumber, numberOfFloor

    game_window.clear()
    
    background.draw()
    level_label.draw()
    up_label.draw()
    down_label.draw()
    waiting_people_batch.draw()
    for i in range(elevatorNumber):
        elevator_batch[i].draw()
    line_batch.draw()
