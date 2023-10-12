import ctypes
import multiprocessing
import time

import numpy as np

from command_interpreters import BasicInterpreter, SubProcessInterpreterAddon, add_further_syntax
from world import World


def subprocess_code(code,
                    size: np.array,
                    raw_colour_layer: multiprocessing.Array,
                    raw_leaf_layer: multiprocessing.Array,
                    raw_object_layer: multiprocessing.Array,
                    raw_kara_position: multiprocessing.Array,
                    raw_kara_rotation: multiprocessing.Array,
                    raw_communication_array: multiprocessing.Array):
    colour_layer = np.frombuffer(raw_colour_layer.get_obj(), np.uint8)
    colour_layer = colour_layer.reshape((size[0], size[1], 3))
    leaf_layer = np.frombuffer(raw_leaf_layer.get_obj(), np.uint32)
    leaf_layer = leaf_layer.reshape(size)
    object_layer = np.frombuffer(raw_object_layer.get_obj(), np.uint32)
    object_layer = object_layer.reshape(size)
    kara_position = np.frombuffer(raw_kara_position.get_obj(), np.uint32)
    kara_rotation = np.frombuffer(raw_kara_rotation.get_obj(), np.uint32)
    communication_array = np.frombuffer(raw_communication_array.get_obj(), np.float32)
    kara = add_further_syntax(SubProcessInterpreterAddon)(BasicInterpreter(colour_layer,
                                                                           leaf_layer,
                                                                           object_layer,
                                                                           kara_position,
                                                                           kara_rotation),
                                                          communication_array)
    print("starting code execution")
    exec(code)


class SubProcessControl:
    @property
    def direct_bypass(self):
        return self.communication_array[self.DIRECT_BYPASS]

    @direct_bypass.setter
    def direct_bypass(self, value: bool):
        self.communication_array[self.DIRECT_BYPASS] = value

    @property
    def paused(self):
        return self.communication_array[self.PAUSED_BY_MAINPROCESS]

    @paused.setter
    def paused(self, value: bool):
        self.communication_array[self.PAUSED_BY_MAINPROCESS] = value

    @property
    def waiting_time(self):
        return self.communication_array[self.WAITING_TIME]

    @waiting_time.setter
    def waiting_time(self, value: float):
        self.communication_array[self.WAITING_TIME] = value

    @property
    def sub_process_pausing(self):
        return self.communication_array[self.PAUSING]

    @property
    def long_waiting_time(self):
        return self.communication_array[self.LONG_WAITING_TIME]

    @long_waiting_time.setter
    def long_waiting_time(self, value: bool):
        self.communication_array[self.LONG_WAITING_TIME] = value

    @property
    def steps_back(self):
        return self.communication_array[self.PAUSING]

    @steps_back.setter
    def steps_back(self, value: int):
        self.communication_array[self.STEPS_BACK] = value

    def __init__(self, world: World, waiting_time: float = 0.25, long_waiting_barrier: float = 0.015625):
        self.world = world
        self.long_waiting_barrier = long_waiting_barrier
        self.process = None

        self.raw_communication_array = multiprocessing.Array(ctypes.c_float, 7)
        self.communication_array = np.frombuffer(self.raw_communication_array.get_obj(), np.float32)

        self.DIRECT_BYPASS = 0  # false for direct bypass
        self.WAITING_TIME = 1
        self.LONG_WAITING_TIME = 2
        self.PAUSED_BY_MAINPROCESS = 3
        self.PAUSING = 4
        self.STEPS_BACK = 5
        self.STOP = 6

        self.direct_bypass = 1 if waiting_time != 0 else 0
        self.waiting_time = waiting_time
        self.long_waiting_time = 0 if waiting_time <= long_waiting_barrier else 1

    def pause(self):
        self.paused = True
        self.direct_bypass = True

    def resume(self):
        self.paused = False
        if self.waiting_time == 0:
            self.direct_bypass = False

    def set_waiting_time(self, value: float):
        self.waiting_time = value
        self.direct_bypass = 1 if value != 0 else 0
        self.long_waiting_time = 0 if value <= self.long_waiting_barrier else 1
        print(value, self.waiting_time, self.direct_bypass, self.long_waiting_time)

    def start_subprocess(self, code):
        self.direct_bypass = 1 if self.waiting_time != 0 else 0
        self.communication_array[self.STOP] = False
        self.paused = False

        self.process = multiprocessing.Process(target=subprocess_code, args=(code,
                                                                             self.world.size,
                                                                             self.world.raw_colour_layer,
                                                                             self.world.raw_leaf_layer,
                                                                             self.world.raw_object_layer,
                                                                             self.world.raw_kara_position,
                                                                             self.world.raw_kara_rotation,
                                                                             self.raw_communication_array))
        self.process.start()

    def stop_subprocess(self):
        self.direct_bypass = True
        self.communication_array[self.STOP] = True

    def kill_subprocess(self):
        self.process.terminate()
