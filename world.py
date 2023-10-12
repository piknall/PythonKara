from __future__ import annotations

import ctypes
import multiprocessing
import os
import time
from dataclasses import dataclass

import numpy as np


class World:
    def __init__(self):
        """
        class to store and organise a worlds content and properties in a process sharable based format
        """
        self.size, self.raw_object_layer, self.object_layer, self.raw_leaf_layer, self.leaf_layer, \
            self.raw_colour_layer, self.colour_layer, self.placeholder_layer, self.raw_kara_position, \
            self.kara_position, self.raw_kara_rotation, self.kara_rotation, self.world_folder_path = (None,) * 13
        self.kara_having_position = False

    def create_arrays(self, size: np.ndarray,
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
