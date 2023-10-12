from __future__ import annotations

from dataclasses import dataclass

import pygame

import world_display
import world
import numpy as np


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


@dataclass
class ColourModificationPoint:
    position: tuple[int, int] | np.ndarray[int, int]
    size: tuple[int, int] | np.ndarray[int, int]
    colour_matrix: np.ndarray[np.ndarray[np.ndarray[int, ...], ...], ...]


@dataclass
class ObjectModificationPoint:
    position: tuple[int, int] | np.ndarray[int, int]
    size: tuple[int, int] | np.ndarray[int, int]
    leaf_matrix: np.ndarray[np.ndarray[int, ...], ...]
    object_matrix: np.ndarray[np.ndarray[int, ...], ...]


class ModificationGroup:
    def __init__(self, reproducible: bool = True):
        self.modification_points = []
        self.reproducible = reproducible

    def append_point(self, modification_point: ColourModificationPoint | ObjectModificationPoint):
        self.modification_points.append(modification_point)


class ModificationHandler:
    COLOUR_MODIFICATION = "colour_modification"
    OBJECT_MODIFICATION = "object_modification"

    def __init__(self, world_object: world.World):
        self.world = world_object
        self.undo_depth = 0
        self.undo_modification_groups = []
        self.redo_modification_groups = []

    def _create_modification_point(self,
                                   position: tuple[int, int] | np.ndarray[int, int],
                                   size: tuple[int, int] | np.ndarray[int, int],
                                   modification_type: str) -> ColourModificationPoint | ObjectModificationPoint:
        if modification_type == self.COLOUR_MODIFICATION:
            return ColourModificationPoint(
                position=position,
                size=size,
                colour_matrix=self.world.colour_layer[position[0]:position[0] + size[0],
                                                      position[1]:position[1] + size[1]].copy()
            )

        if modification_type == self.OBJECT_MODIFICATION:
            return ObjectModificationPoint(
                position=position,
                size=size,
                leaf_matrix=self.world.leaf_layer[position[0]:position[0] + size[0],
                                                  position[1]:position[1] + size[1]].copy(),
                object_matrix=self.world.object_layer[position[0]:position[0] + size[0],
                                                      position[1]:position[1] + size[1]].copy()
            )

        raise ValueError(f"the given modification_type argument: '{modification_type}' is invalid")

    def add_modification_point(self,
                               position: tuple[int, int] | np.ndarray[int, int],
                               size: tuple[int, int] | np.ndarray[int, int],
                               modification_type: str):
        self.undo_modification_groups[-1].append_point(
            self._create_modification_point(position,
                                            size,
                                            modification_type)
        )

    def add_modification_group(self, reproducible: bool = True):
        self.undo_modification_groups.append(ModificationGroup(reproducible))

        self.redo_modification_groups = []
        self.redo_modification_groups = self.redo_modification_groups[:-1 - self.undo_depth]
        self.undo_depth = 0

    def _create_undo_modification_group(self, modification_group: ModificationGroup) -> ModificationGroup:
        undo_modification_group = ModificationGroup()

        for modification_point in modification_group.modification_points:

            undo_modification_group.modification_points.append(
                self._create_modification_point(
                    modification_point.position,
                    modification_point.size,
                    self.COLOUR_MODIFICATION if isinstance(modification_point, ColourModificationPoint)
                    else self.OBJECT_MODIFICATION
                                                )
            )

        return undo_modification_group

    def _apply_modification_group(self, modification_group: ModificationGroup):
        for modification_point in modification_group.modification_points:
            if isinstance(modification_point, ColourModificationPoint):

                self.world.colour_layer[modification_point.position[0]:
                                        modification_point.position[0] + modification_point.size[0],
                                        modification_point.position[1]:
                                        modification_point.position[1] + modification_point.size[1]] = (
                    modification_point.colour_matrix)

            elif isinstance(modification_point, ObjectModificationPoint):

                self.world.leaf_layer[modification_point.position[0]:
                                      modification_point.position[0] + modification_point.size[0],
                                      modification_point.position[1]:
                                      modification_point.position[1] + modification_point.size[1]] = (
                    modification_point.leaf_matrix)

                self.world.object_layer[modification_point.position[0]:
                                        modification_point.position[0] + modification_point.size[0],
                                        modification_point.position[1]:
                                        modification_point.position[1] + modification_point.size[1]] = (
                    modification_point.object_matrix)

            else:
                raise ValueError(f"modification point: '{modification_point}' in "
                                 f"modification group: '{modification_group}' is invalid")

    def undo(self):
        if self.undo_depth + 1 > len(self.undo_modification_groups):
            return None

        targeted_modification_group = self.undo_modification_groups[-1 - self.undo_depth]

        if targeted_modification_group.reproducible:
            self.redo_modification_groups.append(
                self._create_undo_modification_group(targeted_modification_group)
            )

            self.undo_depth += 1

        else:
            self.undo_modification_groups.pop()

        self._apply_modification_group(targeted_modification_group)

    def redo(self):
        if self.undo_depth == 0:
            return None

        targeted_modification_group = self.redo_modification_groups.pop(0)

        self._apply_modification_group(targeted_modification_group)
