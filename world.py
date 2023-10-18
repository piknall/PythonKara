from __future__ import annotations

import ctypes
import multiprocessing
import os
import time
from dataclasses import dataclass

import numpy as np


@dataclass
class ColourModificationPoint:
    """
    class to document changes made to a worlds colour_layer
    """
    position: tuple[int, int] | np.ndarray[int, int]
    size: tuple[int, int] | np.ndarray[int, int]
    colour_matrix: np.ndarray[np.ndarray[np.ndarray[int, ...], ...], ...]


@dataclass
class ObjectModificationPoint:
    """
    class to document changes made to a worlds colour_layer
    """
    position: tuple[int, int] | np.ndarray[int, int]
    size: tuple[int, int] | np.ndarray[int, int]
    leaf_matrix: np.ndarray[np.ndarray[int, ...], ...]
    object_matrix: np.ndarray[np.ndarray[int, ...], ...]


class ModificationGroup:
    def __init__(self, reproducible: bool = True):
        """
        class to bundle multiple ModificationPoints together
        :param reproducible: determines if the changes documented in the ModificationGroups ModificationPoints can be
                             redone if undone before
        """
        self.modification_points = []
        self.reproducible = reproducible

    def append_point(self, modification_point: ColourModificationPoint | ObjectModificationPoint) -> None:
        """
        add ModificationPoint to ModificationGroup
        :param modification_point: ModificationPoint to be added
        """
        self.modification_points.append(modification_point)


class BaseWorld:
    def __init__(self):
        """
        class to store and organise a worlds content and properties in a process sharable based format
        """
        self.size, self.raw_object_layer, self.object_layer, self.raw_leaf_layer, self.leaf_layer, \
            self.raw_colour_layer, self.colour_layer, self.placeholder_layer, self.raw_kara_position, \
            self.kara_position, self.raw_kara_rotation, self.kara_rotation, self.world_folder_path = (None,) * 13
        self.kara_having_position = False

    def create_arrays(self, size: tuple[int, int],
                      raw_object_layer: multiprocessing.Array = None,
                      raw_leaf_layer: multiprocessing.Array = None,
                      raw_colour_layer: multiprocessing.Array = None,
                      raw_kara_position: multiprocessing.Array = None,
                      kara_rotation: multiprocessing.Array = None,
                      create_placeholder_layer: bool = True,
                      inbetween_func=None):
        """
        :param size: size of the world (x: int, y: int)
        :param raw_object_layer: optional initial array, representing all trees and light/ heavy mushrooms
        :param raw_leaf_layer: optional initial array, representing all leafs
        :param raw_colour_layer: optional initial array, representing all background colours of the worlds fields
        :param raw_kara_position: optional initial array, representing karas position (2 ints inside array)
        :param kara_rotation: optional initial array, representing karas rotation (int 0-3 inside array)
        :param create_placeholder_layer: optional initial array, representing all future placeholders
        :param inbetween_func: function to be executed between every major part of the method (in total 4 times)
        :return: None
        """
        self.size = size
        array_length = int(self.size[0] * self.size[1])

        self.raw_object_layer = multiprocessing.Array(ctypes.c_int32, array_length) if raw_object_layer is None \
            else raw_object_layer

        self.object_layer = np.frombuffer(self.raw_object_layer.get_obj(), dtype=np.uint32)
        self.object_layer = self.object_layer.reshape(self.size)
        if inbetween_func is not None:
            inbetween_func()

        self.raw_leaf_layer = multiprocessing.Array(ctypes.c_int32, array_length) if raw_leaf_layer is None else\
            raw_leaf_layer

        self.leaf_layer = np.frombuffer(self.raw_leaf_layer.get_obj(), dtype=np.uint32)
        self.leaf_layer = self.leaf_layer.reshape(self.size)
        if inbetween_func is not None:
            inbetween_func()

        self.raw_colour_layer = multiprocessing.Array(ctypes.c_int8, array_length * 3) if raw_colour_layer is None\
            else raw_colour_layer

        self.colour_layer = np.frombuffer(self.raw_colour_layer.get_obj(), np.uint8)
        self.colour_layer.fill(255)
        self.colour_layer = self.colour_layer.reshape((self.size[0], self.size[1], 3))
        if inbetween_func is not None:
            inbetween_func()

        if create_placeholder_layer:
            self.placeholder_layer = np.zeros(size, dtype=np.uint32)

        self.raw_kara_position = multiprocessing.Array(ctypes.c_int32, 2) if raw_kara_position is None else \
            raw_kara_position
        self.kara_position = np.frombuffer(self.raw_kara_position.get_obj(), dtype=np.uint32)

        self.raw_kara_rotation = multiprocessing.Array(ctypes.c_int32, 1) if kara_rotation is None else \
            kara_rotation
        self.kara_rotation = np.frombuffer(self.raw_kara_rotation.get_obj(), dtype=np.uint32)
        if inbetween_func is not None:
            inbetween_func()

    @staticmethod
    def get_file_paths(world_folder_path):
        """
        :param world_folder_path: absolute path of the world folder
        :return: spec_path, colour_path, leaf_path, object_path, placeholder_path, kara_path
        """
        spec_path = os.path.join(world_folder_path, "KARA_WORLD_SPEC.txt")
        colour_path = os.path.join(world_folder_path, "KaraWorldColourLayer.npy")
        leaf_path = os.path.join(world_folder_path, "KaraWorldLeafLayer.npy")
        object_path = os.path.join(world_folder_path, "KaraWorldObjectLayer.npy")
        placeholder_path = os.path.join(world_folder_path, "KaraWorldPlaceholderLayer.npy")
        kara_path = os.path.join(world_folder_path, "KaraWorldKaraValues.csv")
        return spec_path, colour_path, leaf_path, object_path, placeholder_path, kara_path

    def create_world_folder(self, path: str, name: str):
        """
        :param path: absolute path of the directory, the world folder should be created in
        :param name: name of the world folder
        :return: None
        """
        world_path = os.path.join(path, name)
        if os.path.exists(world_path):
            raise ValueError(f"world_folder path '{world_path}' already exists")
        os.makedirs(world_path)

        spec_path = self.get_file_paths(world_path)[0]
        with open(spec_path, "w") as file:
            file.write(f"size: {', '.join(map(str, self.size))}\ncreated at: {time.ctime()}\nlastly saved: None")

    def save_to_world_folder(self, world_folder_path: str):
        """
        :param world_folder_path: absolute path of the world folder
        :return: None
        """
        spec_path, colour_path, leaf_path, object_path, placeholder_path, kara_path = \
            self.get_file_paths(world_folder_path)

        content = self.unpack_spec_file(spec_path)
        content["size"] = self.size
        content["lastly saved"] = time.ctime()
        with open(spec_path, "w") as spec_file:
            spec_file.write(self.pack_spec_file(content))

        with open(kara_path, "w") as kara_values_file:
            kara_values_file.write("\n".join((", ".join(map(str, self.kara_position)),
                                              str(self.kara_rotation[0]),
                                              str(int(self.kara_having_position)))))

        np.save(colour_path, self.colour_layer)
        np.save(leaf_path, self.leaf_layer)
        np.save(object_path, self.object_layer)
        np.save(placeholder_path, self.placeholder_layer)

    @staticmethod
    def unpack_spec_file(spec_file_path: str) -> dict:
        """
        :param spec_file_path: absolute path of the specification file ("...\\KARA_WORLD_SPEC.txt")
        :return: dictionary of all values specified in spec file, keys are headings, values are values in int / tuple
        format
        """
        with open(spec_file_path, "r") as spec_file:
            content = [tuple(i.strip() for i in line.split(":", 1)) for line in spec_file.read().splitlines()]
            content = dict([line for line in content if len(line) == 2])
            content["size"] = tuple(int(i) for i in content["size"].split(","))
            return content

    @staticmethod
    def pack_spec_file(spec_dict: dict) -> str:
        """
        :param: spec_dict: dictionary created by unpack_spec_file method
        :return: string representing the format, unpacked by unpack_spec_file method (to be saved in specification file)
        """
        return "\n".join(f"{key}: {', '.join(map(str, val)) if type(val) == tuple or type(val) == list else val}"
                         for key, val in zip(spec_dict.keys(), spec_dict.values()))

    def load_from_world_folder(self, world_folder_path: str, inbetween_func=None):
        """
        :param world_folder_path: absolute path of the world folder, data should be loaded from
        :param inbetween_func: inbetween_func argument for create_arrays method
        :return: None
        :raises ValueError: raises error if file is not in world format
        (checked with check_for_wrong_world_format method)
        """
        if self.check_for_wrong_world_format(world_folder_path):
            raise ValueError(self.check_for_wrong_world_format(world_folder_path))

        spec_path, colour_path, leaf_path, object_path, placeholder_path, kara_path = \
            self.get_file_paths(world_folder_path)

        self.world_folder_path = world_folder_path

        with open(spec_path, "r") as spec_file:
            self.size = tuple(int(i) for i in spec_file.read().splitlines()[0].split(": ")[1].split(","))

        self.create_arrays(self.size, create_placeholder_layer=False, inbetween_func=inbetween_func)

        with open(kara_path, "r") as kara_values_file:
            lines = kara_values_file.read().splitlines()
            self.kara_position[:] = np.array([int(i) for i in lines[0].split(",")])
            self.kara_rotation[0] = int(lines[1])
            self.kara_having_position = bool(int(lines[2]))

        self.colour_layer[:] = np.load(colour_path)
        self.leaf_layer[:] = np.load(leaf_path)
        self.object_layer[:] = np.load(object_path)
        self.placeholder_layer = np.load(placeholder_path)

    def check_for_wrong_world_format(self, world_folder_path: str) -> bool | str:
        """
        note that this method does not check the shape of all arrays, neither if any values are reasonable (just format)

        :param: world_folder_path: absolute world folder path
        :return: False if folder is right format, else str description
        """
        file_paths = self.get_file_paths(world_folder_path)

        if not all(map(os.path.exists, file_paths)):
            return f"file path(s): {', '.join(path for path in file_paths if os.path.exists(path))} do(es) not exist"

        spec_path, _, _, _, _, _ = file_paths
        if set(self.unpack_spec_file(spec_path).keys()) != {"size", "created at", "lastly saved"}:
            return f"found spec file keys: {self.unpack_spec_file(spec_path).keys()} does not math required set of: " \
                   f"'size', 'created at', 'lastly saved'"
        else:
            return False

    def resize(self, size: tuple, inbetween_func=None):
        """
        :param size: new size of world (x: int, y: int)
        :param inbetween_func: inbetween_func argument for create_arrays method
        :return: None
        """
        self.size = size
        temp_numpy_arrays = self.get_np_arrays()
        self.create_arrays(size, inbetween_func=inbetween_func)
        for arr, temp_arr in zip(self.get_np_arrays(), temp_numpy_arrays):
            arr[:size[0], :size[1]] = temp_arr

    def get_np_arrays(self, include_placeholder: bool = True):
        """
        :param include_placeholder: include placeholder layer in returned tuple
        :return: colour_layer, leaf_layer, object_layer (, placeholder_layer)
        """
        return (self.colour_layer, self.leaf_layer, self.object_layer, self.placeholder_layer) if include_placeholder \
            else (self.colour_layer, self.leaf_layer, self.object_layer)

    def get_multiprocessing_arrays(self):
        """
        note that the placeholder layer is not included for multiprocessing due to its different functionality
        :return: colour_layer, leaf_layer, object_layer
        """
        return self.raw_colour_layer, self.raw_leaf_layer, self.raw_object_layer

    def get_copy(self) -> WorldCopy:
        """
        :return: WorldCopy class with copied data in it
        """
        return WorldCopy(self.size,
                         self.colour_layer.copy(),
                         self.leaf_layer.copy(),
                         self.object_layer.copy(),
                         self.placeholder_layer.copy(),
                         self.kara_position.copy(),
                         self.kara_rotation[0],
                         self.kara_having_position)

    def load_from_copy(self, copy: WorldCopy):
        """
        :param copy: copy of an older version of the same world
        :return:
        """
        if copy.size != self.size:
            raise ValueError("copy size does not match current world size, create new world instance to load different"
                             " world")
        self.colour_layer[:] = copy.colour_layer
        self.leaf_layer[:] = copy.leaf_layer
        self.object_layer[:] = copy.object_layer
        self.placeholder_layer[:] = copy.placeholder_layer
        self.kara_position[:] = copy.kara_position
        self.kara_rotation[0] = copy.kara_rotation
        self.kara_having_position = copy.kara_having_position


class World(BaseWorld):
    COLOUR_MODIFICATION = "colour_modification"
    OBJECT_MODIFICATION = "object_modification"

    def __init__(self):
        """
        child class of BaseWorld enabling logging functionality
        """
        super().__init__()

        self.undo_depth = 0
        self.undo_modification_groups = []
        self.redo_modification_groups = []

    def _create_modification_point(self,
                                   position: tuple[int, int] | np.ndarray[int, int],
                                   size: tuple[int, int] | np.ndarray[int, int],
                                   modification_type: str) -> ColourModificationPoint | ObjectModificationPoint:
        """
        private method to create a Colour or ObjectModificationPoint of a given size at a given position of the world
        :param position: left up corner of the area in the world to be stored
        :param size: tuple of s and y size of the area in the world to be stored
        :param modification_type: modification type konstant indicating if the worlds colour_layer (COLOUR_MODIFICATION)
                                  or the worlds leaf or object_layer (OBJECT_MODIFICATION) is going to be modified
        :return: ModificationPoint object
        """
        if modification_type == self.COLOUR_MODIFICATION:
            return ColourModificationPoint(
                position=position,
                size=size,
                colour_matrix=self.colour_layer[position[0]:position[0] + size[0],
                                                position[1]:position[1] + size[1]].copy()
            )

        if modification_type == self.OBJECT_MODIFICATION:
            return ObjectModificationPoint(
                position=position,
                size=size,
                leaf_matrix=self.leaf_layer[position[0]:position[0] + size[0],
                                            position[1]:position[1] + size[1]].copy(),
                object_matrix=self.object_layer[position[0]:position[0] + size[0],
                                                position[1]:position[1] + size[1]].copy()
            )

        raise ValueError(f"the given modification_type argument: '{modification_type}' is invalid")

    def add_modification_point(self,
                               position: tuple[int, int] | np.ndarray[int, int],
                               size: tuple[int, int] | np.ndarray[int, int],
                               modification_type: str) -> None:
        """
        method to log the world's state by defining the position of the changes to be made soon
        :param position: left up corner of the area in the world to be stored
        :param size: tuple of s and y size of the area in the world to be stored
        :param modification_type: modification type konstant indicating if the worlds colour_layer (COLOUR_MODIFICATION)
                                  or the worlds leaf or object_layer (OBJECT_MODIFICATION) is going to be modified
        """
        self.undo_modification_groups[-1].append_point(
            self._create_modification_point(position,
                                            size,
                                            modification_type)
        )

    def add_modification_group(self, reproducible: bool = True) -> None:
        """
        method to add a modification group, modification groups are used to bundle multiple modification points together
        :param reproducible: if True once the created ModificationGroup is undone it can be redone else not
        """
        self.undo_modification_groups.append(ModificationGroup(reproducible))

        self.redo_modification_groups = []
        self.redo_modification_groups = self.redo_modification_groups[:-1 - self.undo_depth]
        self.undo_depth = 0

    def _create_redo_modification_group(self, modification_group: ModificationGroup) -> ModificationGroup:
        """
        creates a ModificationGroup based on the changes that are going to be made by another ModificationGroup soon
        :param modification_group: ModificationGroup to be inverted
        :return: ModificationGroup that can be used to undo the changes made by the specified ModificationGroup
        """
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

    def _apply_modification_group(self, modification_group: ModificationGroup) -> None:
        """
        restore the worlds status stored in the specified ModificationGroup
        :param modification_group: ModificationGroup to restore from
        """
        for modification_point in modification_group.modification_points:
            if isinstance(modification_point, ColourModificationPoint):

                self.colour_layer[modification_point.position[0]:
                                  modification_point.position[0] + modification_point.size[0],
                                  modification_point.position[1]:
                                  modification_point.position[1] + modification_point.size[1]] = (
                    modification_point.colour_matrix)

            elif isinstance(modification_point, ObjectModificationPoint):

                self.leaf_layer[modification_point.position[0]:
                                modification_point.position[0] + modification_point.size[0],
                                modification_point.position[1]:
                                modification_point.position[1] + modification_point.size[1]] = (
                    modification_point.leaf_matrix)

                self.object_layer[modification_point.position[0]:
                                  modification_point.position[0] + modification_point.size[0],
                                  modification_point.position[1]:
                                  modification_point.position[1] + modification_point.size[1]] = (
                    modification_point.object_matrix)

            else:
                raise ValueError(f"modification point: '{modification_point}' in "
                                 f"modification group: '{modification_group}' is invalid")

    def undo(self) -> None:
        """
        method to undo the latest not yet undone ModificationGroup
        """
        if self.undo_depth + 1 > len(self.undo_modification_groups):
            return None

        targeted_modification_group = self.undo_modification_groups[-1 - self.undo_depth]

        if targeted_modification_group.reproducible:
            self.redo_modification_groups.append(
                self._create_redo_modification_group(targeted_modification_group)
            )

            self.undo_depth += 1

        else:
            self.undo_modification_groups.pop()

        self._apply_modification_group(targeted_modification_group)

    def redo(self) -> None:
        """
        method to redo the latest not yet redone undone ModificationGroup (note that ModificationGroups that are
        specified to not be redoable are skipped)
        """
        if self.undo_depth == 0:
            return None

        targeted_modification_group = self.redo_modification_groups.pop(0)

        self._apply_modification_group(targeted_modification_group)


@dataclass
class WorldCopy:
    """
    class to store temporary copy of world-class
    """
    size: tuple
    colour_layer: np.array
    leaf_layer: np.array
    object_layer: np.array
    placeholder_layer: np.array
    kara_position: np.array
    kara_rotation: int
    kara_having_position: bool
