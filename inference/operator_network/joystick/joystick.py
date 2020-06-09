import sys

import pygame
import pygame.locals
import drone
import time
import traceback


class JoystickPS3:
    # d-pad
    UP = 4  # UP
    DOWN = 6  # DOWN
    ROTATE_LEFT = 7  # LEFT
    ROTATE_RIGHT = 5  # RIGHT

    # bumper triggers
    TAKEOFF = 11  # R1
    LAND = 10  # L1
    # UNUSED = 9 #R2
    # UNUSED = 8 #L2

    # buttons
    FORWARD = 12  # TRIANGLE
    BACKWARD = 14  # CROSS
    LEFT = 15  # SQUARE
    RIGHT = 13  # CIRCLE

    # axis
    LEFT_X = 0
    LEFT_Y = 1
    RIGHT_X = 2
    RIGHT_Y = 3
    LEFT_X_REVERSE = 1.0
    LEFT_Y_REVERSE = -1.0
    RIGHT_X_REVERSE = 1.0
    RIGHT_Y_REVERSE = -1.0
    DEADZONE = 0.1


class JoystickPS4:
    # d-pad
    UP = -1  # UP
    DOWN = -1  # DOWN
    ROTATE_LEFT = -1  # LEFT
    ROTATE_RIGHT = -1  # RIGHT

    # bumper triggers
    TAKEOFF = 5  # R1
    LAND = 4  # L1
    # UNUSED = 7 #R2
    # UNUSED = 6 #L2

    # buttons
    FORWARD = 3  # TRIANGLE
    BACKWARD = 1  # CROSS
    LEFT = 0  # SQUARE
    RIGHT = 2  # CIRCLE

    # axis
    LEFT_X = 0
    LEFT_Y = 1
    RIGHT_X = 2
    RIGHT_Y = 3
    LEFT_X_REVERSE = 1.0
    LEFT_Y_REVERSE = -1.0
    RIGHT_X_REVERSE = 1.0
    RIGHT_Y_REVERSE = -1.0
    DEADZONE = 0.08


class JoystickPS4ALT:
    # d-pad
    UP = -1  # UP
    DOWN = -1  # DOWN
    ROTATE_LEFT = -1  # LEFT
    ROTATE_RIGHT = -1  # RIGHT

    # bumper triggers
    TAKEOFF = 5  # R1
    LAND = 4  # L1
    # UNUSED = 7 #R2
    # UNUSED = 6 #L2

    # buttons
    FORWARD = 3  # TRIANGLE
    BACKWARD = 1  # CROSS
    LEFT = 0  # SQUARE
    RIGHT = 2  # CIRCLE

    # axis
    LEFT_X = 0
    LEFT_Y = 1
    RIGHT_X = 3
    RIGHT_Y = 4
    LEFT_X_REVERSE = 1.0
    LEFT_Y_REVERSE = -1.0
    RIGHT_X_REVERSE = 1.0
    RIGHT_Y_REVERSE = -1.0
    DEADZONE = 0.08

class JoystickF310:
    # d-pad
    UP = -1  # UP
    DOWN = -1  # DOWN
    ROTATE_LEFT = -1  # LEFT
    ROTATE_RIGHT = -1  # RIGHT

    # bumper triggers
    TAKEOFF = 5  # R1
    LAND = 4  # L1
    # UNUSED = 7 #R2
    # UNUSED = 6 #L2

    # buttons
    FORWARD = 3  # Y
    BACKWARD = 0  # B
    LEFT = 2  # X
    RIGHT = 1  # A

    # axis
    LEFT_X = 0
    LEFT_Y = 1
    RIGHT_X = 3
    RIGHT_Y = 4
    LEFT_X_REVERSE = 1.0
    LEFT_Y_REVERSE = -1.0
    RIGHT_X_REVERSE = 1.0
    RIGHT_Y_REVERSE = -1.0
    DEADZONE = 0.08

class JoystickXONE:
    # d-pad
    UP = 0  # UP
    DOWN = 1  # DOWN
    ROTATE_LEFT = 2  # LEFT
    ROTATE_RIGHT = 3  # RIGHT

    # bumper triggers
    TAKEOFF = 9  # RB
    LAND = 8  # LB
    # UNUSED = 7 #RT
    # UNUSED = 6 #LT

    # buttons
    FORWARD = 14  # Y
    BACKWARD = 11  # A
    LEFT = 13  # X
    RIGHT = 12  # B

    # axis
    LEFT_X = 0
    LEFT_Y = 1
    RIGHT_X = 2
    RIGHT_Y = 3
    LEFT_X_REVERSE = 1.0
    LEFT_Y_REVERSE = -1.0
    RIGHT_X_REVERSE = 1.0
    RIGHT_Y_REVERSE = -1.0
    DEADZONE = 0.09


class JoystickTARANIS:
    # d-pad
    UP = -1  # UP
    DOWN = -1  # DOWN
    ROTATE_LEFT = -1  # LEFT
    ROTATE_RIGHT = -1  # RIGHT

    # bumper triggers
    TAKEOFF = 12  # left switch
    LAND = 12  # left switch
    # UNUSED = 7 #RT
    # UNUSED = 6 #LT

    # buttons
    FORWARD = -1
    BACKWARD = -1
    LEFT = -1
    RIGHT = -1

    # axis
    LEFT_X = 3
    LEFT_Y = 0
    RIGHT_X = 1
    RIGHT_Y = 2
    LEFT_X_REVERSE = 1.0
    LEFT_Y_REVERSE = 1.0
    RIGHT_X_REVERSE = 1.0
    RIGHT_Y_REVERSE = 1.0
    DEADZONE = 0.01

class joystick:
    def __init__(self, drone):
        self.buttons = None
        self.scale = 0.5
        self.speed = 100 * self.scale
        self.throttle = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.roll =0.0
        self.drone = drone
        self.running = True
        self.left_x = 0.0
        self.left_y = 0.0
        self.right_x = 0.0
        self.right_y = 0.0
        pygame.init()
        pygame.joystick.init()
        #self._verifyJostick()

    """
    Prüft welcher Joystick angeschlossen Ist
    """
    def _verifyJostick(self):
        try:
            js = pygame.joystick.Joystick(0)
            js.init()
            js_name = js.get_name()
            print('Joystick name: ' + js_name)
            if js_name in ('Wireless Controller', 'Sony Computer Entertainment Wireless Controller'):
                self.buttons = JoystickPS4
            elif js_name == 'Sony Interactive Entertainment Wireless Controller':
                self.buttons = JoystickPS4ALT
            elif js_name in ('PLAYSTATION(R)3 Controller', 'Sony PLAYSTATION(R)3 Controller'):
                self.buttons = JoystickPS3
            elif js_name in ('Logitech Gamepad F310'):
                self.buttons = JoystickF310
            elif js_name == 'Xbox One Wired Controller':
                self.buttons = JoystickXONE
            elif js_name == 'FrSky Taranis Joystick':
                self.buttons = JoystickTARANIS
        except pygame.error:
            pass

        if self.buttons is None:
            print('no supported joystick found')
            self.running = False
            return

    def __del__(self):
        self.running = False

    """
    Wird benötigt, um den Timeout der Verbindung zur Drohne zu ändern.
    Ist der Timeout nämlich zu hoch gewählt, kann das zur Intabilität führen!
    """
    def turnoff(self):
        self.running = False
        self.drone.set_joystick(False)

    def set_throttle(self, throttle):
        """
                Set_throttle controls the vertical up and down motion of the drone.
                Pass in an int from -1.0 ~ 1.0. (positive value means upward)
                """
        #print("in throttle %f " , throttle)
        self.left_y = self.__fix_range(throttle)

    def set_yaw(self, yaw):
        """
        Set_yaw controls the left and right rotation of the drone.
        Pass in an int from -1.0 ~ 1.0. (positive value will make the drone turn to the right)
        """
        self.left_x = self.__fix_range(yaw)

    def set_pitch(self, pitch):
        """
        Set_pitch controls the forward and backward tilt of the drone.
        Pass in an int from -1.0 ~ 1.0. (positive value will make the drone move forward)
        """
        self.right_y = self.__fix_range(pitch)

    def set_roll(self, roll):
        """
        Set_roll controls the the side to side tilt of the drone.
        Pass in an int from -1.0 ~ 1.0. (positive value will make the drone move to the right)
        """
        self.right_x = self.__fix_range(roll)

    def _normalize(self):
        left_x = self.left_x * 100 * self.scale
        right_x = self.right_x * 100 * self.scale
        left_y = self.left_y * 100 * self.scale
        right_y = self.right_y * 100 * self.scale
        return left_x, left_y, right_x, right_y

    """
            Stick Range form -1.0 to 1.0
    """

    def __fix_range(self, val, min=-1.0, max=1.0):
        if val < min:
            val = min
        elif val > max:
            val = max
        return val

    def update(self, old, new, max_delta=3):
        if abs(old - new) <= max_delta:
            res = new
        else:
            res = 0.0
        return res

    def run(self):
        self._verifyJostick()
        if(self.running):
            try:
                self.drone.set_joystick(True)
                while self.running:
                    # loop with pygame.event.get() is too much tight w/o some sleep
                    time.sleep(0.01)
                    #print("drin")
                    for e in pygame.event.get():
                        #print(e)
                        if e.type == pygame.locals.JOYAXISMOTION:
                            # ignore small input values (Deadzone)
                            if -self.buttons.DEADZONE <= e.value and e.value <= self.buttons.DEADZONE:
                                #print("drin")W
                                #print(e.value)
                                e.value = 0.0
                            if e.axis == self.buttons.LEFT_Y:
                                self.throttle = self.update(self.throttle, e.value * self.buttons.LEFT_Y_REVERSE)
                                self.set_throttle(self.throttle)
                                #print(self.throttle)
                            if e.axis == self.buttons.LEFT_X:
                                self.yaw = self.update(self.yaw, e.value * self.buttons.LEFT_X_REVERSE)
                                self.set_yaw((self.yaw))
                                #self.drone.set_yaw(yaw)
                            if e.axis == self.buttons.RIGHT_Y:
                                self.pitch = self.update(self.pitch, e.value * self.buttons.RIGHT_Y_REVERSE)
                                self.set_pitch(self.pitch)
                                #self.drone.set_pitch(pitch)
                            if e.axis == self.buttons.RIGHT_X:
                                self.roll = self.update(self.roll, e.value * self.buttons.RIGHT_X_REVERSE)
                                self.set_roll(self.roll)
                                #self.drone.set_roll(roll)
                            left_x, left_y, right_x, right_y = self._normalize()
                            self.drone.cmdRC(left_x, left_y, right_x, right_y)
                        elif e.type == pygame.locals.JOYHATMOTION:
                            #print("e[0] {}, e[1] {}".format(e.value[0], e.value[1]))
                            if e.value[0] < 0:
                                self.drone.rc_counter_clockwise(self.speed)
                            if e.value[0] == 0:
                                self.drone.rc_clockwise(0)
                            if e.value[0] > 0:
                                self.drone.rc_clockwise(self.speed)
                            if e.value[1] < 0:
                                self.drone.rc_down(self.speed)
                            if e.value[1] == 0:
                                self.drone.rc_up(0)
                            if e.value[1] > 0:
                                self.drone.rc_up(self.speed)
                        elif e.type == pygame.locals.JOYBUTTONDOWN:
                            print("Joybutton down")
                            print(e.button)
                            if e.button == self.buttons.LAND:
                                self.drone.cmdLand()
                            elif e.button == self.buttons.UP:
                                self.drone.rc_up(self.speed)
                            elif e.button == self.buttons.DOWN:
                                self.drone.rc_down(self.speed)
                            elif e.button == self.buttons.ROTATE_RIGHT:
                                self.drone.rc_right(self.speed)
                            elif e.button == self.buttons.ROTATE_LEFT:
                                self.drone.r_left(self.speed)
                            elif e.button == self.buttons.FORWARD:
                                self.drone.rc_forward(self.speed)
                            elif e.button == self.buttons.BACKWARD:
                                self.drone.rc_backward(self.speed)
                            elif e.button == self.buttons.RIGHT:
                                self.drone.rc_right(self.speed)
                            elif e.button == self.buttons.LEFT:
                                self.drone.rc_left(self.speed)
                        elif e.type == pygame.locals.JOYBUTTONUP:
                            print("Joybutton up")
                            print(e.button)
                            if e.button == self.buttons.TAKEOFF:
                                if self.throttle != 0.0:
                                    print('###')
                                    print('### throttle != 0.0 (This may hinder the drone from taking off)')
                                    print('###')
                                self.drone.cmdTakeoff()
                            elif e.button == self.buttons.UP:
                                self.drone.rc_up(0)
                            elif e.button == self.buttons.DOWN:
                                self.drone.rc_down(0)
                            elif e.button == self.buttons.ROTATE_RIGHT:
                                self.drone.rc_clockwise(0)
                            elif e.button == self.buttons.ROTATE_LEFT:
                                self.drone.rc_counter_clockwise(0)
                            elif e.button == self.buttons.FORWARD:
                                self.drone.rc_forward(0)
                            elif e.button == self.buttons.BACKWARD:
                                self.drone.rc_backward(0)
                            elif e.button == self.buttons.RIGHT:
                                self.drone.rc_right(0)
                            elif e.button == self.buttons.LEFT:
                                self.drone.rc_left(0)
            except KeyboardInterrupt as e:
                print(e)
            except Exception as e:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback.print_exception(exc_type, exc_value, exc_traceback)
                print(e)
        print("Joystick offline")
        """
        except e:
            pass"""
