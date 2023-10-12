from __future__ import annotations

import math

import numpy as np
import pygame as pygame

from world import World


class ZoomLayer:
    def __init__(self,
                 field_size: int,
                 border_width: int,
                 max_surface_size: np.ndarray,
                 world_size: np.ndarray,
                 main_arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
                 kara_position: np.ndarray,
                 kara_rotation: np.ndarray,
                 textures: tuple[pygame.Surface, pygame.Surface, pygame.Surface, pygame.Surface, pygame.Surface],
                 display_size: np.ndarray,
                 left_up_corner: np.ndarray = np.array((0, 0)),
                 background_colour: tuple | np.ndarray = (255, 255, 255),
                 border_colour: tuple | np.ndarray = (0, 0, 0),
                 array_copy_border: int = 20_000
                 ):
        """
        class to represent and generate a layer of the WorldDisplay
        :param field_size: size of one field on surface
        :param border_width: width of the border inbetween fields
        :param max_surface_size: maximum size of surface (bigger surfaces require less rendering)
        :param world_size: size of the displayed world (shape of the underlying numpy matrix's)
        :param main_arrays: tuple of worlds colour, leaf and object layer in this order (underlying shared arrays)
        :param kara_position: array of kara position (underlying shared array)
        :param kara_rotation: array of shape (1) with values from 0 to 3 (underlying shared array)
        :param textures: alpha surface of kara, leaf, tree, light_mushroom and heavy mushroom texture
        :param display_size: displayed size on the display
        :param left_up_corner: initial position at the left up corner of the displayed display
        :param background_colour: default colour of the fields (r, g, b)
        :param border_colour: colour of the separating grid (r, g, b)
        :param array_copy_border: DISCONTINUED estimated number of updated fields at which different rendering is used
        """
        self.field_size = field_size
        self.border_width = border_width
        self.background_colour = background_colour
        self.border_colour = border_colour
        self.world_size = world_size
        self.combined_field_size = field_size + border_width
        self.border_width_vector = np.array((self.border_width,) * 2)
        self.field_size_vector = np.array((self.field_size,) * 2)
        self.display_field_size = self.get_display_field_size(display_size)
        self.copy_arrays_to_displayed_arrays = True if (self.display_field_size[0] * self.display_field_size[1]
                                                        > array_copy_border) else False
        self.updated = False

        self.colour_layer, self.leaf_layer, self.object_layer = main_arrays
        self.kara_position = kara_position
        self.kara_rotation = kara_rotation

        self.kara_texture, self.leaf_texture, self.tree_texture, self.heavy_mushroom_texture, \
            self.light_mushroom_texture = self._scale_textures(textures)
        self.rotated_kara_textures = self._get_rotated_kara_textures(self.kara_texture)
        self.object_textures = (None, self.tree_texture, self.light_mushroom_texture, self.heavy_mushroom_texture)
        self.surface_field_size = self.get_surface_field_size(max_surface_size)
        self.surface = pygame.Surface(self.surface_field_size * self.combined_field_size + self.border_width_vector)
        self.generate_empty_grid()
        self.combined_texture_dict = {}

        self.displayed_colour_layer, self.displayed_leaf_layer, self.displayed_object_layer = \
            self._get_empty_displayed_arrays(self.surface_field_size)
        self.displayed_kara_position = None
        self.displayed_kara_rotation = None
        self.many_updated_fields = False

        self.left_up_corner_field = self.left_up_corner_field = left_up_corner // self.combined_field_size
        self.surface_offset = np.array((0, 0))
        self.set_surface_offset_to_middle()

    def get_display_field_size(self, display_size: np.ndarray) -> np.ndarray:
        """
        method to get the maximum number of fields that can be seen on the display_size specified
        :param display_size: size of the displayed part of the surface
        :return: maximum number of fields on display as 2-dimensional numpy array
        """
        return (np.ceil((display_size - self.border_width_vector) / self.combined_field_size).astype(int) +
                np.array((1,) * 2))

    def get_surface_field_size(self, max_surface_size: np.ndarray) -> np.ndarray:
        """
        method to get the number of fields on the ZoomLayers surface
        :param max_surface_size: maximal number fields on sur
        :return: number of fields on surface as numpy array
        """
        return np.clip((max_surface_size - self.border_width_vector) // self.combined_field_size, 0, self.world_size)

    def _scale_textures(self, textures: tuple):
        """
        method to get a scaled to the ZoomLayers field_size copy of the handled surfaces
        :param textures: pygame surfaces to be scaled
        :return: scaled copy of the input surfaces
        """
        return tuple(pygame.transform.smoothscale(texture, self.field_size_vector) for texture in textures)

    @staticmethod
    def _get_rotated_kara_textures(kara_texture: pygame.Surface):
        """
        method to get the 4 rotation states of kara as rotated surfaces
        :param kara_texture: surface to be used to create the rotated copies
        :return: rotated surfaces in clockwise order, starting with the initial rotation
        """
        return tuple(pygame.transform.rotate(kara_texture, rotation) for rotation in range(0, -360, -90))

    def _get_empty_displayed_arrays(self, surface_field_size: np.ndarray):
        """
        method to get arrays representing the displayed state of every displayed field on surface in three layers
        :param surface_field_size: size of the surface in fields
        :return: displayed colour, leaf, object layer
        """
        size = tuple(surface_field_size)
        return (np.full(size + (3,), self.background_colour, dtype=np.uint8), np.zeros(surface_field_size),
                np.zeros(surface_field_size))

    def generate_empty_grid(self):
        """
        generating empty grid on ZoomLayers surface
        """
        self.surface.fill(self.background_colour)

        for column in range(self.surface_field_size[0] + 1):
            pygame.draw.rect(self.surface, self.border_colour,
                             (column * self.combined_field_size, 0, self.border_width, self.surface.get_height()))

        for row in range(self.surface_field_size[1] + 1):
            pygame.draw.rect(self.surface, self.border_colour,
                             (0, row * self.combined_field_size, self.surface.get_width(), self.border_width))

    def generate_fields(self):
        """
        method to render all the visible updated fields on surface
        """
        # figuring out which part of the surface is actually displayed:
        left_up_corner = self.left_up_corner_field
        right_down_corner = np.clip(left_up_corner + self.display_field_size, 0, self.world_size)
        left_up_corner = np.clip(left_up_corner, 0, self.world_size)
        displayed_left_up = left_up_corner - self.surface_offset
        displayed_right_down = right_down_corner - self.surface_offset

        # figuring out if kara moved since last render:
        kara_moved = (np.any(self.displayed_kara_position != self.kara_position) or
                      self.displayed_kara_rotation != self.kara_rotation[0])

        # decide between different render types:
        if (self.copy_arrays_to_displayed_arrays and self.many_updated_fields) or True:  # temporally manually locked
            # defining cutout of the different world layers on display:
            colour_layer_copy = self.colour_layer[left_up_corner[0]:right_down_corner[0],
                                                  left_up_corner[1]:right_down_corner[1]].copy()
            leaf_layer_copy = self.leaf_layer[left_up_corner[0]:right_down_corner[0],
                                              left_up_corner[1]:right_down_corner[1]].copy()
            object_layer_copy = self.object_layer[left_up_corner[0]:right_down_corner[0],
                                                  left_up_corner[1]:right_down_corner[1]].copy()

            displayed_colour_layer_view = self.displayed_colour_layer[displayed_left_up[0]:displayed_right_down[0],
                                                                      displayed_left_up[1]:displayed_right_down[1]]
            displayed_leaf_layer_view = self.displayed_leaf_layer[displayed_left_up[0]:displayed_right_down[0],
                                                                  displayed_left_up[1]:displayed_right_down[1]]
            displayed_object_layer_view = self.displayed_object_layer[displayed_left_up[0]:displayed_right_down[0],
                                                                      displayed_left_up[1]:displayed_right_down[1]]

            # finding all fields with changed contend:
            update_fields = np.unique(np.concatenate((
                np.column_stack(np.asarray(colour_layer_copy != displayed_colour_layer_view).nonzero()[:2]),
                np.column_stack(np.asarray(leaf_layer_copy != displayed_leaf_layer_view).nonzero()),
                np.column_stack(np.asarray(object_layer_copy != displayed_object_layer_view).nonzero()),
            )), axis=0)

            # updating updated fields on surface
            # for field in update_fields:
            #     normal_position = field + displayed_left_up
            #     self.surface.blit(
            #         self.get_combined_texture(
            #             colour_layer_copy[field[0], field[1]],
            #             leaf_layer_copy[field[0], field[1]],
            #             object_layer_copy[field[0], field[1]]),
            #         self.border_width_vector + normal_position * self.combined_field_size
            #     )

            self.surface.fblits(zip(tuple(self.get_combined_texture(
                        colour_layer_copy[field[0], field[1]],
                        leaf_layer_copy[field[0], field[1]],
                        object_layer_copy[field[0], field[1]])
                                          for field in update_fields),
                                    self.border_width_vector + (update_fields + displayed_left_up)
                                    * self.combined_field_size))

            # setting displayed world layers cutouts to the normal world layers cutouts
            displayed_colour_layer_view[:, :] = colour_layer_copy
            displayed_leaf_layer_view[:, :] = leaf_layer_copy
            displayed_object_layer_view[:, :] = object_layer_copy

        else:
            # finding all fields on display with changed contend:
            update_fields = np.unique(np.concatenate((
                np.column_stack(np.asarray(
                    self.colour_layer[left_up_corner[0]:right_down_corner[0], left_up_corner[1]:right_down_corner[1]] !=
                    self.displayed_colour_layer[displayed_left_up[0]:displayed_right_down[0],
                                                displayed_left_up[1]:displayed_right_down[1]]
                ).nonzero()[:2]),
                np.column_stack(np.asarray(
                    self.leaf_layer[left_up_corner[0]:right_down_corner[0], left_up_corner[1]:right_down_corner[1]] !=
                    self.displayed_leaf_layer[displayed_left_up[0]:displayed_right_down[0],
                                              displayed_left_up[1]:displayed_right_down[1]]
                ).nonzero()),
                np.column_stack(np.asarray(
                    self.object_layer[left_up_corner[0]:right_down_corner[0], left_up_corner[1]:right_down_corner[1]] !=
                    self.displayed_object_layer[displayed_left_up[0]:displayed_right_down[0],
                                                displayed_left_up[1]:displayed_right_down[1]]
                ).nonzero()),
            )), axis=0)

            # updating updated fields on surface and changing displayed layers content according to changes
            for field in update_fields:
                normal_position = field + displayed_left_up
                real_position = tuple(normal_position + self.surface_offset)
                self.surface.blit(
                    self.get_combined_texture(
                        self.colour_layer[real_position],
                        self.leaf_layer[real_position],
                        self.object_layer[real_position]),
                    self.border_width_vector + normal_position * self.combined_field_size
                )

                self.displayed_colour_layer[normal_position[0], normal_position[1]] = self.colour_layer[real_position]
                self.displayed_leaf_layer[normal_position[0], normal_position[1]] = self.leaf_layer[real_position]
                self.displayed_object_layer[normal_position[0], normal_position[1]] = self.object_layer[real_position]

        # update karas displayed position:
        if kara_moved:
            self.displayed_kara_position = self.kara_position.copy()
            self.displayed_kara_rotation = self.kara_rotation[0]

            self.updated = True

        # blitting kara on surface if necessary:
        displayed_array_kara_position = self.displayed_kara_position - self.surface_offset
        if np.all(self.displayed_kara_position >= left_up_corner) and np.all(
                self.displayed_kara_position < right_down_corner):

            self.surface.blit(self.rotated_kara_textures[self.kara_rotation[0]],
                              self.border_width_vector + (self.displayed_kara_position - self.surface_offset)
                              * self.combined_field_size)

            self.displayed_object_layer[displayed_array_kara_position[0],
                                        displayed_array_kara_position[1]] = 255

        # reset many update fields
        self.many_updated_fields = False

        if len(update_fields) > 0:
            self.updated = True

    def get_combined_texture(self, colour: np.ndarray, leaf: int, kara_object: int) -> pygame.Surface:
        """
        method to handle and deliver and ,if needed, create all (stored) combined textures
        :param colour: colour of the desired combined texture
        :param leaf: value indicating if leaf is on the desired combined texture or not (0 == False, 1 == True)
        :param kara_object: index of the object on the desired combined texture (0 if none object is present)
        :return: combined texture
        """
        key = (colour.data.tobytes(), leaf, kara_object)

        if key not in self.combined_texture_dict:
            self.combined_texture_dict[key] = self._generate_combined_texture(colour, leaf, kara_object)

        return self.combined_texture_dict[key]

    def _generate_combined_texture(self, colour: np.ndarray, leaf: int, kara_object: int) -> pygame.Surface:
        """
        method to get a combined texture of colour, leaf and object layer
        :param colour: colour of the field
        :param leaf: value indicating if leaf is on field or not (0 == False, 1 == True)
        :param kara_object: index of the object on the field (0 if none object is present)
        :return: surface with combined textures combined on it
        """
        surface = pygame.Surface(self.field_size_vector)
        surface.fill(colour)

        if leaf:
            surface.blit(self.leaf_texture, (0, 0))

        kara_object_texture = self.object_textures[kara_object]
        if kara_object_texture is not None:
            surface.blit(kara_object_texture, (0, 0))

        return surface

    def set_to_left_up(self, left_up_coordinate: np.ndarray):
        """
        method to set the displays left_up_corner coordinate
        :param left_up_coordinate: new left_up_coordinate
        """
        self.left_up_corner_field = left_up_coordinate // self.combined_field_size

        # checking if a new surface offset is necessary:
        right_down_corner = np.clip(self.left_up_corner_field + self.display_field_size,
                                    0, self.world_size - np.array((1, 1)))
        left_up_corner = np.clip(self.left_up_corner_field, 0, self.world_size)

        if np.any(left_up_corner < self.surface_offset) or \
                np.any(right_down_corner >= self.surface_offset + self.surface_field_size):
            print("middle")
            self.set_surface_offset_to_middle()

    def set_surface_offset_to_middle(self):
        """
        method to calculate and set the surface offset, so that the new surfaces middle matches the displays middle
        """
        middle_displayed_field = (self.left_up_corner_field + self.display_field_size // 2)
        targeted_surface_offset = middle_displayed_field - self.surface_field_size // 2

        self._set_surface_offset(np.clip(targeted_surface_offset, 0, self.world_size - self.surface_field_size))

    def _set_surface_offset(self, new_surface_offset: np.array):
        """
        method to set s new surface offset and reformatting the surface properly
        :param new_surface_offset: surface offset to be set to
        """
        print("surface_offset")
        # DISCONTINUED checking for huge render load:
        if self.surface_offset is None or\
                np.prod(self.display_field_size) - abs(np.prod(new_surface_offset - self.surface_offset)) > 10_000:
            self.many_updated_fields = True

        # reformatting surface:
        if self.surface_offset is not None:
            shift_fields = new_surface_offset - self.surface_offset
            shift_fields = shift_fields.astype(np.int32)
            common_size = self.surface_field_size - np.abs(shift_fields)

            if all(common_size > 0) and True:
                root_left_up = np.clip(shift_fields, 0, self.world_size)
                targeted_left_up = np.clip(-shift_fields, 0, self.world_size)

                self.surface.blit(self.surface, targeted_left_up * self.combined_field_size,
                                  np.concatenate((root_left_up * self.combined_field_size,
                                                  common_size * self.combined_field_size)))

                for display_array in (self.displayed_colour_layer,
                                      self.displayed_leaf_layer,
                                      self.displayed_object_layer):
                    display_array[targeted_left_up[0]:targeted_left_up[0] + common_size[0],
                                  targeted_left_up[1]:targeted_left_up[1] + common_size[1]] =\
                        display_array[root_left_up[0]:root_left_up[0] + common_size[0],
                                      root_left_up[1]:root_left_up[1] + common_size[1]]

        self.surface_offset = new_surface_offset


class WorldDisplay:
    def __init__(self,
                 world: World,
                 display_size: np.ndarray,
                 kara_texture: pygame.Surface,
                 leaf_texture: pygame.Surface,
                 tree_texture: pygame.Surface,
                 light_mushroom_texture: pygame.Surface,
                 heavy_mushroom_texture: pygame.Surface,
                 zoom_index: int = 12,
                 max_surface_size: np.ndarray = np.array((3000, 3000)),
                 border_width: int = 1,
                 background_colour: tuple | np.ndarray = (255, 255, 255),
                 border_colour: tuple | np.ndarray = (0, 0, 0),
                 left_up_corner: np.ndarray = np.array((0, 0)),
                 array_copy_border: int = 20_000,
                 zoom_factor: float = 1.2,
                 visible_map_percentage: float = 0.1,
                 field_size_for_small_rendering: int = 2
                 ):
        """
        class to display a world dynamically
        :param world: World class with constant size
        :param display_size: size of the display the world is drawn in
        :param kara_texture: pygame transformed alpha surface representing kara texture
        :param leaf_texture: pygame transformed alpha surface representing leaf texture
        :param tree_texture: pygame transformed alpha surface representing tree texture
        :param light_mushroom_texture: pygame transformed alpha surface representing light_mushroom texture
        :param heavy_mushroom_texture: pygame transformed alpha surface representing heavy_mushroom texture
        :param zoom_index: initial zoom index to start with (note that the scale is logarithmic
        :param max_surface_size: maximal size allowed for ZoomLayers
        :param border_width: with of the border between fields (grid)
        :param background_colour: default field colour to render with
        :param border_colour: colour of the border
        :param left_up_corner: initial position on world in left up corner of the display
        :param array_copy_border: DISCONTINUED minimum approximate updated fields to use differnent rendering procedure
        :param zoom_factor: factor between zoom steps
        :param visible_map_percentage: percentage of display size of world that can not leave the display
        :param field_size_for_small_rendering: PAUSED maximum field_size to use special rendering procedure
        """
        self.world = world
        self.display_size = display_size

        self.kara_texture = kara_texture
        self.leaf_texture = leaf_texture
        self.tree_texture = tree_texture
        self.light_mushroom_texture = light_mushroom_texture
        self.heavy_mushroom_texture = heavy_mushroom_texture

        self.max_surface_size = max_surface_size
        self.border_width = border_width
        self.border_width_vector = np.array((self.border_width, ) * 2)
        self.field_size_for_small_rendering = field_size_for_small_rendering
        self.background_colour = background_colour
        self.border_colour = border_colour
        self.left_up_corner = left_up_corner
        self.array_copy_border = array_copy_border
        self.visible_map_percentage = visible_map_percentage

        self.zoom_factor = zoom_factor
        self.zoom_steps = (tuple(dict.fromkeys(tuple(round(self.zoom_factor ** i) for i in
                                                     range(int(math.log(min(self.display_size) - self.border_width * 2,
                                                                        self.zoom_factor)))))) +
                           (min(self.display_size) - self.border_width * 2,))
        self.zoom_index = zoom_index

        self.zoom_layers = {}
        self.field_size = None
        self.current_zoom_layer = None
        self._set_field_size(self.zoom_steps[self.zoom_index])

        self.mouse_pressed = False

    @property
    def combined_field_size(self) -> np.ndarray:
        """
        READ ONLY property to get the current ZoomLayers combined field size
        :return: combined_field_size of current ZoomLayer
        """
        # return self.field_size + self.border_width
        return self.current_zoom_layer.combined_field_size

    def _generate_zoom_layer(self, field_size: int, left_up_corner: np.ndarray = None) -> ZoomLayer:
        """
        method to generate new ZoomLayers (with everything initialized)
        :param field_size: field size to initialize with
        :param left_up_corner: optional value to replace self.left_up_corner for setting the new current ZoomLayers
        left_up_corner (only used in special cases like zooming)
        :return: fully initialized ZoomLayer
        """
        return ZoomLayer(field_size,
                         self.border_width,
                         self.max_surface_size,
                         np.array(self.world.size),
                         self.world.get_np_arrays(include_placeholder=False),
                         self.world.kara_position,
                         self.world.kara_rotation,
                         self.get_textures(),
                         self.display_size,
                         left_up_corner if left_up_corner is not None else self.left_up_corner,
                         self.background_colour,
                         self.border_colour,
                         self.array_copy_border)

    def get_textures(self) -> tuple[pygame.Surface, pygame.Surface, pygame.Surface, pygame.Surface, pygame.Surface]:
        """
        method to get all bare object textures
        :return: kara, leaf, tree, heavy_mushroom and light_mushroom texture in this order
        """
        return (self.kara_texture, self.leaf_texture, self.tree_texture, self.heavy_mushroom_texture,
                self.light_mushroom_texture)

    def _set_field_size(self, value: int, left_up_corner: np.ndarray = None):
        """
        internal method to set a new field_size, corresponding to a new current ZoomLayer and handling it
        :param value: new field size to be applied
        :param left_up_corner: optional value to replace self.left_up_corner for setting the new current ZoomLayers
        left_up_corner (only used in special cases like zooming)
        """
        self.field_size = value

        if self.field_size not in self.zoom_layers:
            print("generating")
            self.zoom_layers[self.field_size] = self._generate_zoom_layer(self.field_size, left_up_corner)
            self.current_zoom_layer = self.zoom_layers[self.field_size]

        else:
            self.current_zoom_layer = self.zoom_layers[self.field_size]
            self.current_zoom_layer.set_to_left_up(left_up_corner if left_up_corner is not None
                                                   else self.left_up_corner)

        self.current_zoom_layer.many_updated_fields = True

    def zoom(self, steps: int, zoom_position: np.ndarray = None):
        """
        method to zoom in end out of the world logarithmically
        :param steps: steps to zoom out (positive steps mean zooming out, negative in)
        :param zoom_position: position to zoom the world around.
        If None is given, the left up corner of the display is used
        """
        self.zoom_index = sorted((0, self.zoom_index + steps, len(self.zoom_steps) - 1))[1]
        new_field_size = self.zoom_steps[self.zoom_index]

        zoom_factor = ((new_field_size + self.border_width) /
                       (self.field_size + self.border_width))

        new_left_up_corner = self._get_new_left_up((self.left_up_corner + zoom_position) * zoom_factor - zoom_position
                                                   if zoom_position is not None else self.left_up_corner * zoom_factor,
                                                   forced_field_size=new_field_size)

        new_left_up_corner = new_left_up_corner.astype(np.int32)

        self._set_field_size(new_field_size, new_left_up_corner)
        self.left_up_corner = new_left_up_corner
        self.field_size = new_field_size

    def move_surface(self, direction_vector: np.ndarray):
        """
        method to move the world (ignores zero movement) (no more handling method required)
        (calls current ZoomLayers set_to_left_up method)
        :param direction_vector: vector of movement of the world in respect of old position
        """
        new_left_up = self._get_new_left_up(self.left_up_corner + direction_vector)

        if np.any(new_left_up != self.left_up_corner):
            self.left_up_corner = new_left_up
            self.current_zoom_layer.set_to_left_up(self.left_up_corner)

    def _get_new_left_up(self, new_position: np.ndarray, forced_field_size: int = None):
        """
        internal method to restrict any new left_up coordinates to not let the map get out of display area
        :param new_position: new_left_up to be restricted
        :param forced_field_size: get new left up for forced field size
        :return: restricted coordinate of new_left_up position
        """
        used_combined_field_size = forced_field_size + self.border_width \
            if forced_field_size is not None else self.combined_field_size
        remainder = np.maximum((self.display_size * self.visible_map_percentage).astype(np.int32),
                               2 * self.combined_field_size)

        new_left_up = np.clip(new_position, -self.display_size + remainder,
                              self.world.size * used_combined_field_size - remainder)

        return new_left_up

    def update_surface(self):
        """
        method to render internal surface
        """
        self.current_zoom_layer.generate_fields()

    def _fill_display_area(self, destination: pygame.Surface, position: tuple | np.ndarray, colour: tuple | np.ndarray):
        """
        internal method to fill display with background colour
        (note that calling the update_surface method before is has to be done manually if wanted)
        :param destination: surface to be drawn on
        :param position: position of display on specified surface
        :param colour: colour of backgrund to be filled with
        """
        pygame.draw.rect(destination, colour, tuple(position) + tuple(self.display_size))

    def draw(self, destination: pygame.Surface,
             position: np.array,
             background_colour: tuple | np.array = (255, 255, 255)):
        """
        method to draw the rendered surface onto desired destination surface
        :param destination: surface to blit on
        :param position: position to blit on destination surface
        :param background_colour: colour of the surface to blit on, if param is None this isn't done
        """
        if self.current_zoom_layer.updated:
            if background_colour is not None:
                self._fill_display_area(destination, position, background_colour)

            surface_left_overhead = np.maximum(self.left_up_corner, 0)
            surface_position = -self.left_up_corner + position + surface_left_overhead
            destination.blit(self.current_zoom_layer.surface,
                             surface_position,
                             np.concatenate((
                                 np.maximum(self.left_up_corner, 0) - self.current_zoom_layer.surface_offset *
                                 self.combined_field_size,
                                 self.display_size + self.left_up_corner - surface_left_overhead
                             )))

            pygame.draw.rect(destination, (255, 0, 0), tuple(position - self.left_up_corner) + (3, 3))

    def get_field_coordinate_at_position(self, position_on_display: np.ndarray) -> np.ndarray:
        """
        :param position_on_display: position relative to left up corner of the display to get the field coordinate of
        :return: absolute coordinate of targeted field
        """
        return (self.left_up_corner + position_on_display) // self.combined_field_size

    def input_logic(self, events: tuple[pygame.Event] | list[pygame.Event], display_pos: np.ndarray):
        """
        handling mouse input for moving and zooming the world
        :param events: all pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION and
        pygame.MOUSEBUTTONDOWN, pygame.MOUSEWEEL on display events
        :param display_pos: left_up corner of the display
        """
        total_motion = (0, 0)
        total_zoom = 0
        for event in events:
            if event.type == pygame.MOUSEWHEEL:
                total_zoom += event.y

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self.mouse_pressed = True

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.mouse_pressed = False

            elif event.type == pygame.MOUSEMOTION:
                if self.mouse_pressed:
                    total_motion = tuple(a + b for a, b in zip(total_motion, event.rel))

        if total_zoom != 0:
            self.zoom(total_zoom, np.array(pygame.mouse.get_pos()) - display_pos)

        if total_motion != (0, 0):
            pass
            self.move_surface(np.array(total_motion) * -1)
