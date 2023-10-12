from __future__ import annotations

import time

import numpy as np
from numba import jit


@jit(nopython=True)
def find_first(arr: np.array, value1: int, value2: int):
    """return the index of the first occurrence of value in arr"""
    for count, item in enumerate(arr):
        if item == value1 or item == value2:
            return count


def find_first_zero(arr: np.array):
    """return the index of the first zero element in arr"""
    for count, item in enumerate(arr):
        if not item:
            return count
    return len(arr)


class KaraError(Exception):
    pass


class BasicInterpreter:
    """
    basic interpreter for main process use
    """

    def __init__(self,
                 colour_layer: np.array,
                 leaf_layer: np.array,
                 object_layer: np.array,
                 kara_position: np.array,
                 kara_rotation: np.array):
        self.colour_layer = colour_layer
        self.leaf_layer = leaf_layer
        self.object_layer = object_layer
        self.kara_position = kara_position
        self.kara_rotation = kara_rotation
        self.world_size = object_layer.shape
        self.vector_list = tuple(np.array(i) for i in [(1, 0), (0, 1), (-1, 0), (0, -1)])
        self.kara_vector = self.vector_list[self.kara_rotation[0]]
        self.next_filed = np.mod(self.kara_position + self.kara_vector, self.world_size)
        self.TREE = np.uint32(1)
        self.LIGHT_MUSHROOM = np.uint32(2)
        self.HEAVY_MUSHROOM = np.uint32(3)
        self.moving_functions = (None, self.__tree_motion, self.__light_mushroom_motion, self.__heavy_mushroom_motion)

    def on_leaf(self) -> bool:
        return self.leaf_layer[self.kara_position[0], self.kara_position[1]]

    def tree_front(self) -> bool:
        return self.object_layer[self.next_filed[0], self.next_filed[1]] == self.TREE

    def light_mushroom_front(self) -> bool:
        return self.object_layer[self.next_filed[0], self.next_filed[1]] == self.LIGHT_MUSHROOM

    def heavy_mushroom_front(self) -> bool:
        return self.object_layer[self.next_filed[0], self.next_filed[1]] == self.HEAVY_MUSHROOM

    def colour_on(self) -> np.array:
        return self.colour_layer[self.kara_position[0], self.kara_position[1]]

    def set_colour(self, colour: tuple | list | np.array):
        self.colour_layer[self.kara_position[0], self.kara_position[1]] = np.mod(colour, 256)

    def put_leaf(self):
        if self.leaf_layer[self.kara_position[0], self.kara_position[1]]:
            raise KaraError("leaf already there")
        self.leaf_layer[self.kara_position[0], self.kara_position[1]] = 1

    def remove_leaf(self):
        if not self.leaf_layer[self.kara_position[0], self.kara_position[1]]:
            raise KaraError("no leaf to remove here")
        self.leaf_layer[self.kara_position[0], self.kara_position[1]] = 0

    def left_turn(self, steps: int = 1):
        self.kara_rotation[0] = (self.kara_rotation[0] - steps) % 4
        self.kara_vector = self.vector_list[self.kara_rotation[0]]
        self.next_filed = (self.kara_position + self.kara_vector) % self.world_size

    def right_turn(self, steps: int = 1):
        self.kara_rotation[0] = (self.kara_rotation[0] + steps) % 4
        self.kara_vector = self.vector_list[self.kara_rotation[0]]
        self.next_filed = (self.kara_position + self.kara_vector) % self.world_size

    def move(self, steps: int = 1):
        for _ in range(steps):
            if not self.object_layer[self.next_filed[0], self.next_filed[1]]:
                self.kara_position[:] = self.next_filed
                self.next_filed = np.mod(self.kara_position + self.kara_vector, self.world_size)

            else:
                self.moving_functions[self.object_layer[self.next_filed[0], self.next_filed[1]]]()

    def __heavy_mushroom_motion(self):
        self.kara_position[:] = self.next_filed
        self.next_filed = np.mod(self.next_filed + self.kara_vector, self.world_size)
        if self.object_layer[self.next_filed[0], self.next_filed[1]]:
            self.kara_position[:] = np.mod(self.kara_position + self.kara_vector * -1, self.world_size)
            self.next_filed = np.mod(self.kara_position + self.kara_vector, self.world_size)
            raise KaraError("unable to push mushroom because of obstacle in the way")
        self.object_layer[self.kara_position[0], self.kara_position[1]] = 0
        self.object_layer[self.next_filed[0], self.next_filed[1]] = self.HEAVY_MUSHROOM

    def __light_mushroom_motion(self):
        if self.kara_rotation == 0 or self.kara_rotation == 2:
            row = np.roll(self.object_layer[:, self.kara_position[1]], self.kara_position[0])

            nearest_space = find_first_zero(row[self.kara_vector[0]:][::self.kara_vector[0]])
            nearest_obstacle = find_first(row[self.kara_vector[0]:][::self.kara_vector[0]],
                                          self.TREE, self.HEAVY_MUSHROOM)

        else:
            column = np.roll(self.object_layer[self.kara_position[0], :], self.kara_position[1])

            nearest_space = find_first_zero(column[self.kara_vector[1]:][::self.kara_vector[1]])
            nearest_obstacle = find_first(column[self.kara_vector[1]:][::self.kara_vector[1]],
                                          self.TREE, self.HEAVY_MUSHROOM)

        if nearest_space is None or nearest_obstacle < nearest_space:
            raise KaraError("unable to push mushroom because of obstacle in the way")

        self.object_layer[tuple(np.mod(self.next_filed + self.kara_vector * nearest_space,
                                       self.world_size))] = self.LIGHT_MUSHROOM
        self.object_layer[self.next_filed[0], self.next_filed[1]] = 0
        self.kara_position[:] = self.next_filed
        self.next_filed = np.mod(self.next_filed + self.kara_vector)

    def __tree_motion(self):
        raise KaraError("kara is unable to move because of tree in front")


def sub_process_waiting(func):
    def func_with_control(*args, **kwargs):
        if args[0].control_array[0]:
            start = time.perf_counter()

            if args[0].control_array[6]:
                exit()

            if args[0].control_array[3]:
                while args[0].control_array[3]:
                    if args[0].control_array[6]:
                        exit()
                start = time.perf_counter()

            if args[0].control_array[2]:
                time_aim = start + args[0].control_array[1]
                while time.perf_counter() < time_aim:
                    if args[0].control_array[3]:
                        remaining_time = time_aim - time.perf_counter()
                        while args[0].control_array[3]:
                            if args[0].control_array[6]:
                                exit()
                        time_aim = time.perf_counter() + remaining_time

            else:
                time_aim = start + args[0].control_array[1]
                while time.perf_counter() < time_aim:
                    pass

        return func(*args, **kwargs)

    return func_with_control


class SubProcessInterpreterAddon:
    """
    class to add sub process control functionality to a simple interpreter fast but not perfectly performant
    """

    def __init__(self, interpreter: BasicInterpreter, control_array: np.array):
        self.interpreter = interpreter
        self.control_array = control_array

    @sub_process_waiting
    def on_leaf(self) -> bool:
        return self.interpreter.on_leaf()

    @sub_process_waiting
    def tree_front(self) -> bool:
        return self.interpreter.tree_front()

    @sub_process_waiting
    def light_mushroom_front(self) -> bool:
        return self.interpreter.light_mushroom_front()

    @sub_process_waiting
    def heavy_mushroom_front(self) -> bool:
        return self.interpreter.heavy_mushroom_front()

    @sub_process_waiting
    def colour_on(self) -> np.array:
        return self.interpreter.colour_on()

    @sub_process_waiting
    def set_colour(self, colour):
        self.interpreter.set_colour(colour)

    @sub_process_waiting
    def put_leaf(self):
        self.interpreter.put_leaf()

    @sub_process_waiting
    def remove_leaf(self):
        self.interpreter.remove_leaf()

    @sub_process_waiting
    def left_turn(self, steps: int = 1):
        self.interpreter.left_turn(steps)

    @sub_process_waiting
    def right_turn(self, steps: int = 1):
        self.interpreter.right_turn(steps)

    @sub_process_waiting
    def __single_move(self):
        self.interpreter.move()

    @sub_process_waiting
    def move(self, steps: int = 1):
        if steps <= 10:
            self.interpreter.move(steps)
        else:
            for _ in range(steps):
                self.__single_move()


def add_further_syntax(interpreter):
    class FurtherSyntaxInterpreter(interpreter):
        def __int__(self,
                    colour_layer: np.array,
                    leaf_layer: np.array,
                    object_layer: np.array,
                    kara_position: np.array,
                    kara_rotation: np.array):
            super().__init__(colour_layer, leaf_layer, object_layer, kara_position, kara_rotation)

        def turnLeft(self, steps: int = 1):
            self.left_turn(steps)

        def turnRight(self, steps: int = 1):
            self.right_turn(steps)

        def treeFront(self) -> bool:
            return self.tree_front()

        def mushroomFront(self) -> bool:
            return self.heavy_mushroom_front()

        def onLeaf(self) -> bool:
            return self.on_leaf()

        def putLeaf(self):
            self.put_leaf()

        def removeLeaf(self):
            self.remove_leaf()

        def turn_left(self, steps: int = 1):
            self.left_turn(steps)

        def turn_right(self, steps: int = 1):
            self.right_turn(steps)

    return FurtherSyntaxInterpreter
