from __future__ import annotations

import math
import time
import typing

import numpy as np
import pygame


class PositionHint:
    RIGHT = "right_border"
    LEFT = "right_border"
    TOP = "right_border"
    BOTTOM = "right_border"

    def __init__(self,
                 direction: str,
                 position_node: int | GuiElement | typing.Callable,
                 offset: int = 0,
                 percentage: float = 1.0):
        self.direction = direction
        self.position_node = position_node
        self.offset = offset
        self.percentage = percentage

    def get_position(self) -> int:
        if type(self.position_node) is int:
            referenced_position = self.position_node

        elif isinstance(self.position_node, GuiElement):
            if self.direction == self.LEFT:
                referenced_position = self.position_node.get_left_position()
            elif self.direction == self.RIGHT:
                referenced_position = self.position_node.get_right_position()
            elif self.direction == self.TOP:
                referenced_position = self.position_node.get_top_position()
            elif self.direction == self.BOTTOM:
                referenced_position = self.position_node.get_bottom_position()
            else:
                raise ValueError(f"invalid position node: {self.position_node}")

        elif callable(self.position_node):
            referenced_position = self.position_node()

        else:
            raise ValueError(f"invalid position node: {self.position_node}")

        return int(referenced_position * self.percentage + self.offset)


class GuiElement:
    def __init__(self,
                 left_position: int | PositionHint | None,
                 right_position: int | PositionHint | None,
                 top_position: int | PositionHint | None,
                 bottom_position: int | PositionHint | None,
                 width: int | PositionHint | None,
                 height: int | PositionHint | None
                 ):
        self._left_position_node = left_position
        self._right_position_node = right_position
        self._top_position_node = top_position
        self._bottom_position_node = bottom_position
        self._width_node = width
        self._height_node = height

        self.check_position_arguments()

        self._position: tuple | None = None
        self._size: tuple | None = None

    @property
    def position(self):
        return self._position

    @property
    def size(self):
        return self._size

    @property
    def position_attr_dict(self):
        return {
            "left_position": self._left_position_node,
            "right_position": self._right_position_node,
            "top_position": self._top_position_node,
            "bottom_position": self._bottom_position_node,
            "width_node": self._width_node,
            "height_node": self._height_node,
        }

    def check_position_arguments(self) -> None:
        """
        check all given position and size nodes for different interpretations and recursive references
        """
        if sum((self._left_position_node is not None,
                self._right_position_node is not None,
                self._width_node is not None)) not in (1, 2):
            raise ValueError(f"position nodes {self._left_position_node=}, {self._right_position_node=}, "
                             f"{self._width_node=} can not be resolved")

        if sum((self._top_position_node is not None,
                self._bottom_position_node is not None,
                self._height_node is not None)) not in (1, 2):
            raise ValueError(f"position nodes {self._top_position_node=}, {self._bottom_position_node=}, "
                             f"{self._height_node=} can not be resolved")

        recursion_check = self._check_gui_element_recursion(self, self)

        if recursion_check is not None:
            error_message = ' -> '.join(
                [
                    str(element) if isinstance(element, GuiElement) else f'{element[0]} at attr {element[1]}'
                    for element in recursion_check
                ]
            )

            raise ValueError(f"position dependencies can not be resolved; faulty recursion path:\n{error_message}")

    def _check_gui_element_recursion(self,
                                     gui_element: GuiElement,
                                     forbidden_gui_element: GuiElement) \
            -> list[tuple[GuiElement, str] | GuiElement] | None:

        for key, value in gui_element.position_attr_dict.items():
            if isinstance(value, PositionHint) and isinstance(value.position_node, GuiElement):
                if value.position_node is forbidden_gui_element:
                    return [(gui_element, key), forbidden_gui_element]

                if ((recursion_check := self._check_gui_element_recursion(value.position_node, forbidden_gui_element))
                        is not None):
                    return [(gui_element, key)] + recursion_check

        return None

    def terminate_position(self) -> None:
        """
        calculate the GuiElement position based on the given nodes, elements with fixed width have to specify size prior
        to call
        """
        top_terminated_position = self._top_position_node.get_position() \
            if isinstance(self._top_position_node, PositionHint) else self._top_position_node

        bottom_terminated_position = self._bottom_position_node.get_position() \
            if isinstance(self._bottom_position_node, PositionHint) else self._bottom_position_node

        left_terminated_position = self._left_position_node.get_position() \
            if isinstance(self._left_position_node, PositionHint) else self._left_position_node

        right_terminated_position = self._right_position_node.get_position() \
            if isinstance(self._right_position_node, PositionHint) else self._right_position_node

        terminated_width = self._width_node.get_position() \
            if isinstance(self._width_node, PositionHint) else self._width_node

        terminated_height = self._height_node.get_position() \
            if isinstance(self._height_node, PositionHint) else self._height_node

        if terminated_width is None:
            if not self._size:
                raise ValueError("width has to be specified prior to calling this method or specified as node")
            terminated_width = self._size[0]

        if terminated_height is None:
            if not self._size:
                raise ValueError("height has to be specified prior to calling this method or specified as node")
            terminated_height = self._size[1]

        # horizontal position and width calculation:
        if right_terminated_position is None:
            new_x_position = left_terminated_position
            new_width = terminated_width
        elif left_terminated_position is None:
            new_x_position = right_terminated_position - terminated_width
            new_width = terminated_width
        else:
            new_x_position = left_terminated_position
            new_width = right_terminated_position - left_terminated_position

        # vertical position and width calculation:
        if bottom_terminated_position is None:
            new_y_position = top_terminated_position
            new_height = terminated_height
        elif top_terminated_position is None:
            new_y_position = bottom_terminated_position - terminated_height
            new_height = terminated_height
        else:
            new_y_position = top_terminated_position
            new_height = bottom_terminated_position - top_terminated_position

        self._position = (new_x_position, new_y_position)
        self.set_size((new_width, new_height))

    def get_left_position(self) -> int:
        """
        get x position of the GuiElements left side
        :return: x position
        """
        return self._position[0]

    def get_top_position(self) -> int:
        """
        get y position of the GuiElements top side
        :return: y position
        """
        return self._position[1]

    def get_right_position(self) -> int:
        """
        get x position of the GuiElements right side
        :return: x position + x size
        """
        return self._position[0] + self._size[0]

    def get_bottom_position(self) -> int:
        """
        get y position of the GuiElements bottom side
        :return: y position + y size
        """
        return self._position[1] + self._size[1]

    def set_height(self, new_height: int):
        """
        method to be overwritten to enable resizing functionality, the self._size attribute has to be set within the
        new method  or this parent method has to be called
        :param new_height: new height to be set
        """
        self._size = (self.size[0], new_height)

    def set_width(self, new_width: int):
        """
        method to be overwritten to enable resizing functionality, the self._size attribute has to be set within the
        new method or this parent method has to be called
        :param new_width: new height to be set
        """
        self._size = (new_width, self._size[1])

    def set_size(self, new_size: tuple[int, int]):
        """
        method to be overwritten to enable resizing functionality, the self._size attribute has to be set within the
        new method or this parent method has to be called
        :param new_size:
        :return:
        """
        self._size = new_size

    # methods to be defined:
    def blit(self, surface: pygame.Surface, force_blit: bool = False) -> None:
        pass

    def handel_events(self, events: list[pygame.Event] | tuple[pygame.Event, ...]):
        pass


class GuiManager:
    def __init__(self,
                 surface: pygame.Surface,
                 position: tuple[int, int] = (0, 0)):
        self.surface = surface
        self.position = position

        self.gui_elements: list[GuiElement] = []

    def add_element(self, gui_element: GuiElement):
        if not isinstance(gui_element, GuiElement):
            raise ValueError(f"gui_element argument has to be an instance of GuyElement not {type(gui_element)}")

        self.gui_elements.append(gui_element)

    def arrange_elements(self) -> None:
        for gui_element in self.gui_elements:
            gui_element.terminate_position()

    def handle_events(self, events: list[pygame.Event] | tuple[pygame.Event, ...]) -> None:
        for gui_element in self.gui_elements:
            gui_element.handel_events(events)

    def blit_elements(self, force_blit: bool = False):
        for gui_element in self.gui_elements:
            gui_element.blit(self.surface, force_blit)


class ButtonBackgroundAppearance:
    def __init__(self,
                 size_percentage: float = 1,
                 corner_radius_percentage: float = None,
                 colour: tuple | np.ndarray = (120, 120, 120),
                 line_width: int = 0,
                 smooth_scaling: bool = True):
        self.size_percentage = size_percentage
        self.corner_radius_percentage = corner_radius_percentage
        self.colour = colour
        self.line_width = line_width

        self.smooth_scaling = smooth_scaling

    def get_surface(self, size: int) -> pygame.Surface:
        surface_size = math.ceil(size * self.size_percentage)
        background_surface = pygame.Surface((surface_size,) * 2).convert_alpha()
        background_surface.fill((255, 255, 255, 0))
        pygame.draw.rect(background_surface,
                         self.colour,
                         (0, 0, surface_size, surface_size),
                         self.line_width,
                         math.ceil(self.corner_radius_percentage * size if
                                   self.corner_radius_percentage is not None else -1))
        return background_surface


class ButtonAppearance:
    def __init__(self,
                 size_percentage: float = 1,
                 alpha: int = None,
                 grayscale: bool = False,
                 background_appearance: ButtonBackgroundAppearance | tuple[ButtonBackgroundAppearance, ...] = None,
                 smooth_scaling: bool = True):
        self.size_percentage = size_percentage
        self.alpha = alpha
        self.grayscale = grayscale
        self.background_appearance = background_appearance

        self.smooth_scaling = smooth_scaling

    def get_appearance_applied_button(self, texture: pygame.Surface, size: int) -> pygame.Surface:
        # setting up surface:
        if type(self.background_appearance) is ButtonBackgroundAppearance:
            surface_size = math.ceil(size * max(self.size_percentage, self.background_appearance.size_percentage))
        elif hasattr(self.background_appearance, "__iter__"):
            surface_size = math.ceil(size * max(self.size_percentage,
                                                *tuple(appearance.size_percentage for appearance
                                                       in self.background_appearance)))
        else:
            surface_size = math.ceil(size * self.size_percentage)
        appearance_surface = pygame.Surface((surface_size,) * 2).convert_alpha()
        appearance_surface.fill((255, 255, 255, 0))

        # adding background surface:
        if type(self.background_appearance) is ButtonBackgroundAppearance:
            background_surface = self.background_appearance.get_surface(size)
            position_on_appearance_surface = ((surface_size - background_surface.get_width()) // 2,) * 2
            appearance_surface.blit(background_surface, position_on_appearance_surface)
        elif hasattr(self.background_appearance, "__iter__"):
            for background in self.background_appearance:
                background_surface = background.get_surface(size)
                position_on_appearance_surface = ((surface_size - background_surface.get_width()) // 2,) * 2
                appearance_surface.blit(background_surface, position_on_appearance_surface)

        # adding button:
        button_size = math.ceil(self.size_percentage * size)

        if self.smooth_scaling:
            button_surface = pygame.transform.smoothscale(texture, (button_size,) * 2)
        else:
            button_surface = pygame.transform.scale(texture, (button_size,) * 2)

        if self.alpha is not None:
            button_surface.set_alpha(self.alpha)

        appearance_surface.blit(button_surface, ((surface_size - button_size) // 2,) * 2)

        if self.grayscale:
            appearance_surface = pygame.transform.grayscale(appearance_surface)

        return appearance_surface


NORMAL_STATE = "normal_state"
PRESSED_STATE = "pressed_state"
HOVERED_STATE = "hovered_state"
SELECTED_STATE = "selected_state"
PASSIVE_STATE = "passive_state"


class Button:
    def __init__(self,
                 texture: pygame.Surface,
                 commands: tuple[typing.Callable, ...] | typing.Callable,
                 args: tuple[tuple] | tuple = None,
                 normal_appearance: ButtonAppearance = None,
                 pressed_appearance: ButtonAppearance = None,
                 hovered_appearance: ButtonAppearance = None,
                 selected_appearance: ButtonAppearance = None,
                 passive_appearance: ButtonAppearance = None
                 ):
        default_appearance = ButtonAppearance()
        self.texture = texture
        self.commands = commands
        self.args = args

        self.normal_appearance = normal_appearance if normal_appearance is not None else default_appearance
        self.pressed_appearance = pressed_appearance if pressed_appearance is not None else default_appearance
        self.hovered_appearance = hovered_appearance if hovered_appearance is not None else default_appearance
        self.selected_appearance = selected_appearance if selected_appearance is not None else default_appearance
        self.passive_appearance = passive_appearance if passive_appearance is not None else default_appearance

        self.appearance_surface_cages = {
            NORMAL_STATE: {},
            PRESSED_STATE: {},
            HOVERED_STATE: {},
            SELECTED_STATE: {},
            PASSIVE_STATE: {}
        }

    def get_appearance_by_state(self, state) -> ButtonAppearance:
        """
        method to get the appearance that matches the provided state
        :param state: state constant to be matched
        :return: the matching ButtonAppearance object
        """
        if state == NORMAL_STATE:
            return self.normal_appearance
        if state == PRESSED_STATE:
            return self.pressed_appearance
        if state == HOVERED_STATE:
            return self.hovered_appearance
        if state == SELECTED_STATE:
            return self.selected_appearance
        if state == PASSIVE_STATE:
            return self.passive_appearance
        raise ValueError("unknown state")

    def get_surface(self, button_size: int, state: str) -> pygame.Surface:
        if self.appearance_surface_cages[state].get(button_size) is None:
            self.appearance_surface_cages[state][button_size] = (self.get_appearance_by_state(state).
                                                                 get_appearance_applied_button(self.texture,
                                                                                               button_size))
        return self.appearance_surface_cages[state][button_size]

    def blit_button(self, surface: pygame.Surface, center: tuple | np.ndarray, button_size: int, state: str):
        button_surface = self.get_surface(button_size, state)
        position = tuple(cent - butt // 2 for cent, butt in zip(center, button_surface.get_size()))
        surface.blit(button_surface, position)

    def call_commands(self):
        """
        calls all functions or methods provided with the commands argument with the arguments specified in initial args
        make sure that the format of the provided callables is the same as for the args, meaning a tuple of a tuple of
        arguments if a tuple of callables with more than one argument is provided. If you get a to much or to less
        argument or zip error this is most likely for this reason
        """
        if callable(self.commands):
            if self.args is not None:
                if type(self.args) is tuple:
                    self.commands(*self.args)
                else:
                    self.commands(self.args)
            else:
                self.commands()
        else:
            if self.args is None:
                for command in self.commands:
                    command()
                return None

            for command, arguments in zip(self.commands, self.args):
                if arguments is not None:
                    if type(arguments) is tuple:
                        command(*arguments)
                    else:
                        command(arguments)
                else:
                    command()


class FunctionalButton:
    def __init__(self):
        pass


class ButtonArrangement:
    def __init__(self,
                 shape: tuple[int, int],
                 buttons: tuple[Button, ...],
                 arrangement_pointers: tuple[str, ...] | None,
                 initial_button_size: int,
                 button_padding_size: int = 15,
                 border_padding_size: int = None,
                 passive_buttons: tuple[bool, ...] = None,
                 background_colour: tuple | np.ndarray = (255, 255, 255)
                 ):
        """
        class to store a list of buttons in a given shape
        :param shape:
        :param buttons:
        :param arrangement_pointers:
        :param initial_button_size:
        :param button_padding_size:
        :param border_padding_size:
        :param passive_buttons:
        :param background_colour:
        """
        self.shape = shape
        self.button_size = initial_button_size
        self.button_padding_size = button_padding_size
        self.border_padding_size = border_padding_size if border_padding_size is not None \
            else math.ceil(button_padding_size / 2)
        self.background_colour = background_colour
        self.buttons = buttons
        self.arrangement_pointers = arrangement_pointers if arrangement_pointers is not None else ((None,)
                                                                                                   * len(self.buttons))

        self.surface = self.generate_surface()

        self.pressed_index = None
        self.hovered_index = None
        self.selected_index = None
        self.passive_button = passive_buttons if passive_buttons is not None else [False, ] * len(self.buttons)

        self.displayed_states = [None, ] * len(self.buttons)

    @property
    def combined_button_size(self):
        """
        equivalent to self.button_size + self.button_padding_size
        :return: space between positions on two separate buttons
        """
        return self.button_padding_size + self.button_size

    def generate_surface(self) -> pygame.Surface:
        surface_size = tuple(2 * self.border_padding_size - self.button_padding_size +
                             self.combined_button_size * axis for axis in self.shape)
        surface = pygame.Surface(surface_size)
        surface.fill(self.background_colour)
        return surface

    def _get_center_at_index(self, index: int) -> tuple[int, int]:
        return (self.border_padding_size + self.button_size // 2 + self.combined_button_size * (index % self.shape[0]),
                self.border_padding_size + self.button_size // 2 + self.combined_button_size * (index // self.shape[0]))

    def _draw_background_at_index(self, index: int):
        left_up_position = (self.border_padding_size - math.ceil(self.button_padding_size / 2) +
                            self.combined_button_size * (index % self.shape[0]),
                            self.border_padding_size - math.ceil(self.button_padding_size / 2) +
                            self.combined_button_size * (index // self.shape[0]))
        size = (self.combined_button_size,) * 2
        pygame.draw.rect(self.surface, self.background_colour, left_up_position + size)

    def _blit_button(self, index: int, state: str):
        self.buttons[index].blit_button(self.surface,
                                        self._get_center_at_index(index),
                                        self.button_size,
                                        state)

    def get_button_state(self, index: int):
        if self.passive_button[index]:
            return PASSIVE_STATE

        if self.pressed_index == index:
            return PRESSED_STATE

        if self.selected_index == index:
            return SELECTED_STATE

        if self.hovered_index == index:
            return HOVERED_STATE

        return NORMAL_STATE

    def terminate_surface(self) -> bool:
        updated = False
        for index, displayed_state in enumerate(self.displayed_states):
            button_state = self.get_button_state(index)
            if button_state != displayed_state:
                updated = True
                self._draw_background_at_index(index)
                self._blit_button(index, button_state)
                self.displayed_states[index] = button_state
        return updated

    def _set_hovered(self, index: int | None):
        if index is None:
            self.hovered_index = None
            return None

        if self.passive_button[index]:
            return None

        if self.pressed_index is not None:
            return None

        if self.selected_index == index:
            self.hovered_index = None
            return None

        self.hovered_index = index

    def set_hovered(self, index: int | None):
        if index is None or self.passive_button[index]:
            self.hovered_index = None
        else:
            self.hovered_index = index

    def _set_pressed(self, index: int):
        if self.passive_button[index]:
            return None
        self.hovered_index = None
        self.pressed_index = index

    def set_pressed(self, index: int | None):
        if index is None or self.passive_button[index]:
            self.pressed_index = None
        else:
            self.pressed_index = index

    def set_selected(self, index: int):
        if index is not None and not self.passive_button[index]:
            self.selected_index = index

    def _mouse_up(self, index: int | None, set_selected: bool = False):
        if set_selected and index == self.pressed_index and self.pressed_index is not None:
            self.selected_index = self.pressed_index
            self.pressed_index = None
            return None

        self.pressed_index = None

        if index is not None and self.selected_index != index:
            self.hovered_index = index

    def mouse_up(self, index: int | None, set_selected: bool = False, always_set_selected: bool = False):
        if set_selected and (self.pressed_index == index or (always_set_selected and self.pressed_index is not None)):
            self.set_selected(self.pressed_index)

        self.pressed_index = None

    def set_passive(self, index: int):
        if self.pressed_index == index:
            self.pressed_index = None
        if self.hovered_index == index:
            self.hovered_index = None

        self.passive_button[index] = True

    def set_active(self, index: int):
        self.passive_button[index] = False

    def set_all_passive(self):
        self.pressed_index = None
        self.hovered_index = None

        self.passive_button = [True, ] * len(self.passive_button)

    def set_all_active(self):
        self.passive_button = [False, ] * len(self.passive_button)

    def get_button_at_index(self, index: int) -> Button | None:
        if index >= len(self.buttons):
            return None
        return self.buttons[index]

    def get_arrangement_pointer_at_index(self, index: int) -> str | None:
        if index >= len(self.buttons):
            return None
        return self.arrangement_pointers[index]

    def get_surface_size(self):
        return self.surface.get_size()


class ButtonBox(GuiElement):
    def __init__(self,
                 left_position: int | PositionHint | None,
                 right_position: int | PositionHint | None,
                 top_position: int | PositionHint | None,
                 bottom_position: int | PositionHint | None,
                 button_layout_size: tuple[int, int],
                 button_size: int,
                 button_padding_size: int = 10,
                 border_padding_size: int = None,
                 selected_mode: bool = False,
                 background_colour: tuple[int, ...] | np.ndarray = (255, 255, 255),
                 process_not_longer_touched_buttons: bool = False
                 ):
        """
        container for pressable buttons
        :param button_layout_size: number of buttons along x- and y-axis in this order
        :param button_size: size of the buttons inside the ButtonBox as an integer
        :param button_padding_size: empty space between buttons
        :param border_padding_size: empty space between other buttons and the ButtonBoxes edge
        :param selected_mode: boolean specifying if a pressed button should be displayed as selected
        :param background_colour: background colour inside the ButtonBox
        :param process_not_longer_touched_buttons: if argument is truthy pressed buttons that are no longer hovered
                                                   over will be processed anyway, meaning that their commands and
                                                   pointers will be called
        """
        super().__init__(
            left_position,
            right_position,
            top_position,
            bottom_position,
            None,
            None,
        )

        self.button_layout_size = button_layout_size
        self.button_size = button_size
        self.button_padding_size = button_padding_size
        self.border_padding_size = border_padding_size if border_padding_size is not None \
            else math.ceil(button_padding_size / 2)
        self.selected_mode = selected_mode
        self.background_colour = background_colour
        self._size = self._get_surface_size(self.button_layout_size)

        self.all_buttons = []
        self.button_arrangements = {}
        self.current_button_arrangement: ButtonArrangement | None = None

        self.process_not_longer_touched_buttons = process_not_longer_touched_buttons

        self.reload_surface = True
        self.updated_buttons = False

    @property
    def combined_button_size(self) -> int:
        """
        equivalent to self.button_size + self.button_padding_size
        :return: space between positions on two separate buttons
        """
        return self.button_size + self.button_padding_size

    @property
    def arrangement_shape(self) -> tuple[int, int]:
        """
        get shape of the current button arrangements buttons
        :return: number of buttons along x and y-Axis
        """
        return self.current_button_arrangement.shape

    def _get_surface_size(self, shape: tuple[int, int]) -> tuple[int, int]:
        """
        calculate required surface size for a given arrangement shape
        :param shape: shape of an arrangement
        :return: required size for given shape
        """
        return tuple(self.border_padding_size * 2 - self.button_padding_size + axis * self.combined_button_size
                     for axis in shape)

    def get_size(self) -> tuple[int, int]:
        """
        get (surface) size of the ButtonBox (equivalent to self.size)
        :return: size attribute of ButtonBox
        """
        return self.size

    def add_button_arrangement(self,
                               name: str,
                               arrangement_shape: tuple[int, int],
                               buttons: tuple[Button, ...],
                               arrangement_pointers: tuple[str | None, ...] | None = None,
                               passive_buttons: tuple[bool, ...] = None):
        """
        method to create a new ButtonArrangement within ButtonBox
        :param name: name of the added ButtonArrangement that can be referred to in arrangement pointers
                     (used as dict key)
        :param arrangement_shape: shape of the buttons in the added ButtonArrangement, meaning number of buttons fitting
                                  along x and y-axis
        :param buttons: tuple of Buttons to fill the provided arrangement_shape
        :param arrangement_pointers: tuple of equal length as the provided buttons argument so that every button can
                                     have a pointer to another ButtonArrangement, meaning that the ButtonBox will switch
                                     to this other arrangement if the button associated with the pointer is pressed.
                                     If a tuple is provided each element within has to be the name of another
                                     arrangement (see in the name argument) or None for no pointer associated with that
                                     button, if arrangement_pointers argument is let to be None no pointers will be
                                     created
        :param passive_buttons: tuple of equal length as the buttons argument or None. If a tuple is provided each
                                boolean value within this tuple determines whether the associated button should be
                                passive
        """
        if any(shape > layout for shape, layout in zip(arrangement_shape, self.button_layout_size)):
            raise ValueError(f"ButtonArrangement shape {arrangement_shape} "
                             f"is not compatible with layout size {self.button_layout_size}")

        self.button_arrangements[name] = ButtonArrangement(shape=arrangement_shape,
                                                           buttons=buttons,
                                                           arrangement_pointers=arrangement_pointers,
                                                           initial_button_size=self.button_size,
                                                           button_padding_size=self.button_padding_size,
                                                           border_padding_size=self.border_padding_size,
                                                           passive_buttons=passive_buttons,
                                                           background_colour=self.background_colour)

        if self.current_button_arrangement is None:
            self.current_button_arrangement = self.button_arrangements[name]

    def set_current_arrangement(self, name: str):
        """
        set the arrangement with the given name to be the current arrangement used
        :param name: name of the arrangement (see in add_button_arrangement name argument documentation)
        """
        self.current_button_arrangement = self.button_arrangements[name]

    def get_index_at_position(self, position: tuple[int, ...]) -> int | None:
        """
        get index of the button at a given position of the ButtonBoxes surface
        :param position: position on the ButtonBoxes surface (including all border paddings)
        :return: index of the button at the given position or None if there is no button at the given position
        """
        pos_without_border = tuple(axis - self.border_padding_size for axis in position)
        button_position = tuple(axis // self.combined_button_size if
                                axis % self.combined_button_size <= self.button_size else None
                                for axis in pos_without_border)

        if button_position[0] is None or button_position[1] is None:
            return None
        if not all(0 <= axis < shape for axis, shape in zip(button_position, self.arrangement_shape)):
            return None

        index = button_position[0] + button_position[1] * self.arrangement_shape[0]
        return index if index < len(self.current_button_arrangement.buttons) else None

    def _logic(self, events: list[pygame.Event, ...] | tuple[pygame.Event, ...], position: tuple):
        """
        method to input the users mouse inputs in form of the associated pygame events
        :param events: list or tuple of events to be handled (MOUSEMOTION, MOUSEBUTTONDOWN and MOUSEBUTTONUP event types
                       are handled)
        :param position: position of the ButtonBox on the display
        """
        for event in events:
            if event.type == pygame.MOUSEMOTION:
                # get pressed down button index:
                position_on_surface = tuple(event_pos - pos for event_pos, pos in zip(event.pos, position))
                button_index = self.get_index_at_position(position_on_surface)

                self.current_button_arrangement.set_hovered(button_index)

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # get pressed down button index:
                position_on_surface = tuple(event_pos - pos for event_pos, pos in zip(event.pos, position))

                button_index = self.get_index_at_position(position_on_surface)

                self.current_button_arrangement.set_pressed(button_index)

            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                # get pressed down button index:
                position_on_surface = tuple(event_pos - pos for event_pos, pos in zip(event.pos, position))
                button_index = self.get_index_at_position(position_on_surface)

                # get index of button to call, if there is no valid index None should be given
                index_of_button_to_call = self.current_button_arrangement.pressed_index if (
                        (button_index is not None or self.process_not_longer_touched_buttons)
                        and self.current_button_arrangement.pressed_index is not None) \
                    else None

                self.current_button_arrangement.mouse_up(button_index,
                                                         set_selected=self.selected_mode,
                                                         always_set_selected=self.process_not_longer_touched_buttons)

                if index_of_button_to_call is not None:
                    # call command:
                    self.current_button_arrangement.buttons[index_of_button_to_call].call_commands()

                    # set new current_arrangement:
                    arrangement_pointer = (self.current_button_arrangement.
                                           get_arrangement_pointer_at_index(index_of_button_to_call))

                    if arrangement_pointer is None:
                        continue

                    self.set_current_arrangement(arrangement_pointer)
                    self.current_button_arrangement.set_hovered(self.get_index_at_position(position_on_surface))

                    self.reload_surface = True

        self.updated_buttons = self.current_button_arrangement.terminate_surface()

    def handel_events(self, events: list[pygame.Event] | tuple[pygame.Event, ...]):
        self._logic(events, self.position)

    def _blit_if_necessary(self, surface: pygame.Surface, position: tuple[int, int], force_blit: bool = False):
        """
        method to blit the ButtonBox at a given position and a given surface in an efficient way
        :param surface: surface to blit on
        :param position: position on the surface to blit at
        :param force_blit: if True the ButtonBox is blitted completely even if it is not necessary (to be used if the
                           ButtonBoxes position has changed)
        """
        if force_blit:
            self.reload_surface = True

        if self.reload_surface:
            self.updated_buttons = True

            if self.button_layout_size[0] != self.arrangement_shape[0]:
                pygame.draw.rect(surface,
                                 self.background_colour,
                                 (position[0] + self.current_button_arrangement.surface.get_width(),
                                  position[1],
                                  self.size[0] - self.current_button_arrangement.surface.get_width(),
                                  self.size[1])
                                 )

            if self.button_layout_size[1] != self.arrangement_shape[1]:
                pygame.draw.rect(surface,
                                 self.background_colour,
                                 (position[0],
                                  position[1] + self.current_button_arrangement.surface.get_height(),
                                  self.size[0],
                                  self.size[1] - self.current_button_arrangement.surface.get_height())
                                 )

        if self.updated_buttons:
            surface.blit(self.current_button_arrangement.surface, position)

        self.updated_buttons = False
        self.reload_surface = False

    def blit(self, surface: pygame.Surface, force_blit: bool = False) -> None:
        self._blit_if_necessary(surface,
                                self.position,
                                force_blit)


class EmbeddedButtonBox(ButtonBox):
    def __init__(self,
                 left_position: int | PositionHint | None,
                 right_position: int | PositionHint | None,
                 top_position: int | PositionHint | None,
                 bottom_position: int | PositionHint | None,
                 outline_width: int,
                 button_layout_size: tuple[int, int],
                 button_size: int,
                 button_padding_size: int = 10,
                 border_padding_size: int = None,
                 selected_mode: bool = False,
                 background_colour: tuple[int, ...] | np.ndarray = (255, 255, 255),
                 process_not_longer_touched_buttons: bool = False,
                 additional_padding_size: int = 0,
                 outline_colour: tuple[int, ...] | np.ndarray = (0, 0, 0),
                 outline_corner_radius: int = -1,
                 heading_text: str = None,
                 heading_font: pygame.Font | str = "",
                 font_size: int = 15,
                 bold_font: bool = False,
                 italic_font: bool = False,
                 horizontal_heading_position: int = None,
                 additional_horizontal_heading_offset: int = 0,
                 vertical_heading_offset: int = 0,
                 heading_padding: int = 3,
                 heading_colour: tuple[int, ...] = (0, 0, 0),
                 heading_background_colour: tuple[int, ...] = None,
                 heading_antialias: bool = True,
                 heading_surface: pygame.Surface = None,
                 additional_top_padding: int = -1,
                 additional_bottom_padding: int = -1,
                 additional_left_padding: int = -1,
                 additional_right_padding: int = -1,
                 top_offset: int = 0
                 ):
        """
        child class of ButtonBox, adding an outline and optional title to the blitted ButtonBox
        :param outline_width: width of the outline around the internal ButtonBox
        :param button_layout_size: number of buttons along x- and y-axis in this order (handed to parent ButtonBox)
        :param button_size: size of the buttons inside the ButtonBox as an integer (handed to parent ButtonBox)
        :param button_padding_size: empty space between buttons (handed to parent ButtonBox)
        :param border_padding_size: empty space between other buttons and the underlying ButtonBoxes edge
                                    (note that you can add additional padding between internal ButtonBox and outline by
                                    specifying the additional_padding_size parameter, this is might be necessary for a
                                    bigger outline_corner_radius) (handed to parent ButtonBox)
        :param selected_mode: boolean specifying if a pressed button should be displayed as selected
                              (handed to parent ButtonBox)
        :param background_colour: background colour inside the EmbeddedButtonBox (handed to parent ButtonBox as well)
        :param process_not_longer_touched_buttons: if argument is truthy pressed buttons that are no longer hovered
                                                   over will be processed anyway, meaning that their commands and
                                                   pointers will be called (handed to parent ButtonBox as well)
        :param additional_padding_size: space between internal ButtonBoxes surface and outline (see border padding size)
                                        (this parameter can be further specified and overwritten by the up, down, left
                                        and right_additional_padding size parameters)
        :param outline_colour: colour of the outline around the buttons
        :param outline_corner_radius: rect argument for the outline, specifying the radius of rounded corners, make sure
                                      to chose a big enough additional padding size, as a big corner radius can let the
                                      round corners reach inside the area of the internal ButtonBoxes surface, leading
                                      to graphical bugs (-1 means no round corners)
        :param heading_text: string to be displayed as a heading for the EmbeddedButtonBox (can be overwritten by
                             heading_surface)
        :param heading_font: pygame font or name of sys font to be used for rendering the heading text
        :param font_size: font size to be used for creating SysFont (ignored if pygame font is specified)
        :param bold_font: specifying if created SysFont should be bold (ignored if pygame font is specified)
        :param italic_font: specifying if created SysFont should be italic (ignored if pygame font is specified)
        :param horizontal_heading_position: horizontal position of the heading relative to EmbeddedButtonBoxes position
        :param additional_horizontal_heading_offset: horizontal shift of heading relative to its default location
                                                     (unused if horizontal_heading_offset is specified)
        :param vertical_heading_offset: vertical position of heading relative to its default position
        :param heading_padding: left and right space between outline and heading
        :param heading_colour: font colour of rendered heading text
        :param heading_background_colour: font background colour of rendered heading text
        :param heading_antialias: use antialiasing for rendering heading text
        :param heading_surface: surface to be used as heading instead of heading text
        :param additional_top_padding: specify additional_padding in top direction (see additional_padding)
        :param additional_bottom_padding: specify additional_padding in bottom direction (see additional_padding)
        :param additional_left_padding: specify additional_padding in left direction (see additional_padding)
        :param additional_right_padding: specify additional_padding in right direction (see additional_padding)
        :param top_offset: vertical offset to be added to the draw location specified in the blit_if_necessary method
        """
        super().__init__(
            left_position=left_position,
            right_position=right_position,
            top_position=top_position,
            bottom_position=bottom_position,
            button_layout_size=button_layout_size,
            button_size=button_size,
            button_padding_size=button_padding_size,
            border_padding_size=border_padding_size,
            selected_mode=selected_mode,
            background_colour=background_colour,
            process_not_longer_touched_buttons=process_not_longer_touched_buttons
        )

        # initialise outline parameters:
        self.outline_width = outline_width
        self.additional_padding_size = additional_padding_size
        self.outline_colour = outline_colour
        self.outline_corner_radius = outline_corner_radius
        self.internal_rect_corner_radius = self._get_internal_corner_radius()

        # initialise padding and offset:
        self.top_padding = additional_top_padding if additional_top_padding != -1 else additional_padding_size
        self.down_padding = additional_bottom_padding if additional_bottom_padding != -1 else additional_padding_size
        self.left_padding = additional_left_padding if additional_left_padding != -1 else additional_padding_size
        self.right_padding = additional_right_padding if additional_right_padding != -1 else additional_padding_size
        self.top_offset = top_offset

        # initialise heading if heading is specified:
        if heading_surface is not None or heading_text is not None:
            # store initial values:
            self._heading_text = heading_text
            self._additional_heading_offset = additional_horizontal_heading_offset
            self._heading_vertical_offset = vertical_heading_offset
            self._heading_padding = heading_padding
            self._heading_colour = heading_colour
            self._heading_background_colour = heading_background_colour
            self._heading_antialias = heading_antialias
            self._heading_surface = heading_surface
            self._heading_font_size = font_size
            self._heading_bold_font = bold_font
            self._heading_italic_font = italic_font

            # define needed values to later get heading position and surface:
            self.horizontal_heading_position = horizontal_heading_position if horizontal_heading_position is not None \
                else max((self.outline_corner_radius, self.outline_width)) + self._additional_heading_offset
            self.heading_font = heading_font if isinstance(heading_font, pygame.Font) \
                else pygame.font.SysFont(heading_font,
                                         self._heading_font_size,
                                         self._heading_bold_font,
                                         self._heading_italic_font)

            # define heading surface and position:
            self.heading = self._get_heading()
            self.heading_position = self._get_heading_position()

        else:
            self.heading = None

    def _get_heading(self) -> pygame.Surface:
        """
        get surface to be used as heading, requires initialised heading parameters
        :return: heading surface
        """
        if self._heading_surface is not None:
            text_render = self._heading_surface

        else:
            text_render = self.heading_font.render(self._heading_text,
                                                   self._heading_antialias,
                                                   self._heading_colour,
                                                   self._heading_background_colour)

        if (text_render.get_height() >= self.outline_width and
                self._heading_padding == 0 and
                self._heading_background_colour is not None):
            return text_render

        background_surface = pygame.Surface((text_render.get_width() + 2 * self._heading_padding,
                                             max((self.outline_width, text_render.get_height()))
                                             + self._heading_vertical_offset))
        background_surface.fill(self.background_colour)
        background_surface.blit(text_render,
                                (self._heading_padding,
                                 (background_surface.get_height() - text_render.get_height()) // 2
                                 + self._heading_vertical_offset))
        return background_surface

    def _get_heading_position(self) -> tuple[int, int]:
        """
        get position for the heading surface to be blitted at (relative to EmbeddedButtonBoxes position
        :return: position to be blitted at
        """
        return (self.horizontal_heading_position,
                self.top_offset + self.outline_width // 2 - self.heading.get_height() // 2)

    def _get_internal_corner_radius(self):
        """
        method to get the needed corner radius for additional padding rect, so that its corners aren't visible
        :return: additional padding rects corner radius
        """
        if self.outline_corner_radius == -1:
            return -1

        smallest_radius = math.sqrt(self.outline_corner_radius ** 2 / 2)

        if not self.outline_corner_radius - smallest_radius > self.outline_width:
            return -1

        corner_diagonal = math.sqrt(2 * (self.outline_corner_radius - self.outline_width) ** 2)
        needed_radius = math.ceil(math.sqrt(2) * math.sqrt((self.outline_corner_radius - corner_diagonal) ** 2) -
                                  self.outline_corner_radius + corner_diagonal)

        return needed_radius + 1

    def _get_position_with_outline_width(self, position: tuple[int, int]) -> tuple:
        """
        method to add outline and additional padding to a given position
        :param position: position of the EmbeddedButtonBox
        :return: combined position
        """
        return (position[0] + self.outline_width + self.left_padding,
                position[1] + self.outline_width + self.top_padding + self.top_offset)

    def get_size(self) -> tuple[int, int]:
        """
        method to get size of EmbeddedButtonBox
        :return: size of EmbeddedButtonBox
        """
        parent_size = super().get_size()
        return (parent_size[0] + self.outline_width * 2 + self.left_padding + self.right_padding,
                parent_size[1] + self.outline_width * 2 + self.top_padding + self.down_padding + self.top_offset)

    def _logic(self, events: list[pygame.Event, ...] | tuple[pygame.Event, ...], position: tuple):
        """
        calls parents logic method with modified position
        :param events: events to be handled
        :param position: position of the EmbeddedButtonBox
        """
        super()._logic(events, self._get_position_with_outline_width(position))

    def handel_events(self, events: list[pygame.Event] | tuple[pygame.Event, ...]):
        self._logic(events, self.position)

    def _draw_additional_padding(self, surface: pygame.Surface,
                                 position: tuple[int, int],
                                 parent_size: tuple[int, int]):
        """
        method to draw additional padding around ButtonBox
        :param surface: surface to blit on
        :param position: position the button box should be blitted at
        :param parent_size: size of the underlining ButtonBox
        """
        pygame.draw.rect(surface,
                         self.background_colour,
                         (
                             position[0] + self.outline_width,
                             position[1] + self.outline_width + self.top_offset,
                             self.left_padding + parent_size[0] + self.right_padding,
                             self.top_padding + parent_size[1] + self.down_padding
                         ),
                         self.additional_padding_size,
                         self.internal_rect_corner_radius)

    def _draw_outline(self, surface: pygame.Surface,
                      position: tuple[int, int],
                      parent_size: tuple[int, int]):
        """
        method to draw the specified outline around padding and ButtonBox
        :param surface: surface to blit on
        :param position: position the button box should be blitted at
        :param parent_size: size of the underlining ButtonBox
        """
        pygame.draw.rect(surface,
                         self.outline_colour,
                         (
                             position[0],
                             position[1] + self.top_offset,
                             self.outline_width * 2 + self.left_padding + parent_size[0] + self.right_padding,
                             self.outline_width * 2 + self.top_padding + parent_size[1] + self.down_padding
                         ),
                         self.outline_width,
                         self.outline_corner_radius)

    def _draw_heading(self, surface: pygame.Surface, position: tuple[int, int]):
        """
        method to draw the specified outline around padding and ButtonBox
        :param surface: surface to blit on
        :param position: position the button box should be blitted at
        """
        surface.blit(self.heading,
                     tuple(heading_pos + pos for heading_pos, pos in zip(self.heading_position, position))
                     )

    def _blit_if_necessary(self, surface: pygame.Surface, position: tuple[int, int], force_blit: bool = False):
        """
        overwrites ButtonBoxes method by adding additional padding, heading and outline
        :param surface: surface to blit on
        :param position: position on the surface to blit at
        :param force_blit: if True the ButtonBox is blitted completely even if it is not necessary (to be used if the
                           ButtonBoxes position has changed)
        """
        if force_blit:
            self.reload_surface = True

        reload_surface = self.reload_surface

        super()._blit_if_necessary(surface, self._get_position_with_outline_width(position))

        if reload_surface:
            parent_size = super().get_size()

            if self.additional_padding_size != 0:
                self._draw_additional_padding(surface, position, parent_size)

            self._draw_outline(surface, position, parent_size)

            if self.heading is not None:
                self._draw_heading(surface, position)

    def blit_on_surface(self, surface: pygame.Surface, position: tuple[int, int]):
        """
        blit ButtonBox on surface at given position (equivalent to blit_if_necessary(..., force_blit=True))
        :param surface: surface to blit on
        :param position: position on the surface to blit at
        """
        self._blit_if_necessary(surface, position, True)


if __name__ == "__main__":
    # test code:
    import matplotlib.pyplot as plt

    pygame.init()
    window = pygame.display.set_mode((1000, 1000))
    window.fill((255, 255, 255))

    p_appearance = ButtonAppearance(1.2, alpha=150)
    h_appearance = ButtonAppearance(size_percentage=1,
                                    background_appearance=(ButtonBackgroundAppearance(colour=(150, 150, 150),
                                                                                      line_width=3),
                                                           ButtonBackgroundAppearance(colour=(150, 150, 150),
                                                                                      size_percentage=0.9,
                                                                                      corner_radius_percentage=0.5))
                                    )
    s_appearance = ButtonAppearance(background_appearance=ButtonBackgroundAppearance(size_percentage=1.1,
                                                                                     colour=(150, 150, 150),
                                                                                     corner_radius_percentage=0.1))
    pas_appearance = ButtonAppearance(alpha=150, grayscale=True)
    test_button = Button(pygame.image.load(r"assets\logo.png").convert_alpha(),
                         commands=lambda a, b: print(f":^) {a + b}"),
                         args=("halihalo", " here am I"),
                         pressed_appearance=p_appearance,
                         hovered_appearance=h_appearance,
                         selected_appearance=s_appearance,
                         passive_appearance=pas_appearance)

    TUERKIS = (64, 214, 218)
    LIGHT_GRAY = (200, 200, 200)
    WHITE = (255, 255, 255)
    test_box = EmbeddedButtonBox(50,
                                 None,
                                 50,
                                 None,
                                 10,
                                 (4, 2),
                                 70,
                                 button_padding_size=20,
                                 additional_padding_size=7,
                                 background_colour=WHITE,
                                 outline_corner_radius=30,
                                 selected_mode=True,
                                 heading_text="Menü",
                                 font_size=25,
                                 heading_font="comic sans",
                                 vertical_heading_offset=-3,
                                 process_not_longer_touched_buttons=True)

    test_box.add_button_arrangement("first",
                                    (3, 2),
                                    (test_button,) * 5,
                                    arrangement_pointers=("2",) + (None,) * 4,
                                    passive_buttons=(False, False, True, False, True))
    test_box.add_button_arrangement("2",
                                    (4, 1),
                                    (test_button,) * 4,
                                    arrangement_pointers=("first",) + (None,) * 3)

    manager = GuiManager(window)
    manager.add_element(test_box)
    manager.arrange_elements()

    clock = pygame.time.Clock()

    timings = []
    while True:
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                plt.plot(range(len(timings)), timings)
                plt.title("frametime x60:")
                plt.show()
                exit()

        start = time.perf_counter()
        # test_box._logic(events, (100, 100))
        # test_box._blit_if_necessary(window, (100, 100))
        manager.handle_events(events)
        manager.blit_elements()
        end = time.perf_counter()
        timings.append((end - start) * 60)

        # window.blit(test_box.current_button_arrangement.surface, (300, 500))

        pygame.display.update()
        clock.tick(60)
