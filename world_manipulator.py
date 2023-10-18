from __future__ import annotations

from dataclasses import dataclass

import pygame

import world_display
import world
import numpy as np


# TODO: implement WorldManipulator
class WorldManipulator:
    KARA = "kara_mode"
    COLOUR = "colour_mode"
    LEAF = "leaf_mode"
    LIGHT_MUSHROOM = "light_mushroom_mode"
    HEAVY_MUSHROOM = "heavy_mushroom_mode"
    TRASH = "trash_mode"

    def __init__(self,
                 world_object: world.World,
                 display_object: world_display.WorldDisplay,
                 display_position: np.ndarray[int, int]
                 ):
        self.world = world_object
        self.display_object = display_object
        self.display_position = display_position

        self.kara_rotation = 0
        self.mode = self.LEAF
        self.colour = (255, 0, 0)
        self.currently_pressed = False
        self.current_sequence_endpoint = None

    def draw_at_position(self, position: np.ndarray[int, int]):
        print(position)
        self.world.leaf_layer[position[0], position[1]] = 1

    def handle_drawn_lines(self, drawn_lines: list[list[np.ndarray[np.int32, np.int32], ...], ...]):
        pass
