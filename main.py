import time

import numpy as np
import pygame

import world
import sub_process
import world_display
import user_interface as ui
import world_manipulator

if __name__ == "__main__":
    pygame.init()
    window = pygame.display.set_mode((1700, 1100), flags=pygame.DOUBLEBUF, vsync=True)
    window.fill((200, 200, 200))

    test_world = world.World()
    test_world.create_arrays(np.array((500, 500)))
    # print(test_world.object_layer.shape, test_world.object_layer)
    # test_world.create_world_folder(r"C:\Users\Pi07\Desktop\kara_08.23", name="test_world")
    # test_world.save_to_world_folder(r"C:\Users\Pi07\Desktop\kara_08.23\test_world")
    # print(test_world.check_for_wrong_world_format(r"C:\Users\Pi07\Desktop\kara_08.23\test_world"))
    # test_world.load_from_world_folder(r"C:\Users\Pi07\Desktop\kara_08.23\test_world")
    # test_world.save_to_world_folder(r"C:\Users\Pi07\Desktop\kara_08.23\test_world")

    # test_world.colour_layer[10][10] = np.array((255, 0, 0), dtype=np.uint8)
    test_world.colour_layer[5, 5] = (255, 0, 0)

    display_pos = (160, 120)
    display = world_display.WorldDisplay(test_world,
                                         np.array((1300, 820)),
                                         pygame.image.load(r"assets\kara.png").convert_alpha(),
                                         pygame.image.load(r"assets\leaf.png").convert_alpha(),
                                         pygame.image.load(r"assets\tree.png").convert_alpha(),
                                         pygame.image.load(r"assets\mushroom.png").convert_alpha(),
                                         pygame.image.load(r"assets\mushroom.png").convert_alpha(),
                                         max_surface_size=np.array((3000, 3000))
                                         )

    print(display.zoom_steps)
    complex_code = """
count = -1
colours = tuple(np.random.randint(0, 255, 3) for _ in range(71))
while True:
    for _ in range(500):
        count += 1
        if count % 100_000 == 0 and True:
            print(count // 1000)

        if kara.on_leaf():
            kara.remove_leaf()
        else:
            kara.put_leaf()
        kara.move()
        kara.set_colour(colours[count % 71])
        
    kara.right_turn()
    kara.move()
    kara.left_turn()
    """

    simpe_code = """
count = -1
while True:
    count += 1
    for _ in range(300):
        count += 1
        kara.set_colour(np.array((0, 0, (count) % 255)))
        if kara.on_leaf():
            kara.remove_leaf()
        else:
            kara.put_leaf()
        kara.move()
        
        
    kara.right_turn()
    kara.move()
    kara.left_turn()
    if count >= 1000 and False:
        while True:
            pass

    """

    sub_process_control = sub_process.SubProcessControl(test_world, 0.00)
    sub_process_control.start_subprocess(complex_code)

    execution_control = ui.EmbeddedButtonBox(5, (2, 1), 70, 10)

    hovered_appearance = ui.ButtonAppearance(background_appearance=ui.ButtonBackgroundAppearance(line_width=3,
                                                                                                 size_percentage=1.05,
                                                                                                 colour=(200, 200, 200))
                                             )
    pressed_appearance = ui.ButtonAppearance(background_appearance=ui.ButtonBackgroundAppearance(colour=(150, 150, 150),
                                                                                                 size_percentage=1.05)
                                             )
    passive_appearance = ui.ButtonAppearance(alpha=50, size_percentage=1)

    play_button = ui.Button(pygame.image.load(r"assets\play_button.png").convert_alpha(),
                            commands=sub_process_control.start_subprocess,
                            args=(complex_code,),
                            hovered_appearance=hovered_appearance,
                            pressed_appearance=pressed_appearance)

    resume_button = ui.Button(pygame.image.load(r"assets\play_button.png").convert_alpha(),
                              commands=sub_process_control.resume,
                              hovered_appearance=hovered_appearance,
                              pressed_appearance=pressed_appearance)

    pause_button = ui.Button(pygame.image.load(r"assets\pause_button.png").convert_alpha(),
                             commands=sub_process_control.pause,
                             hovered_appearance=hovered_appearance,
                             pressed_appearance=pressed_appearance)

    stop_button = ui.Button(pygame.image.load(r"assets\stop_button.png").convert_alpha(),
                            commands=sub_process_control.stop_subprocess,
                            hovered_appearance=hovered_appearance,
                            pressed_appearance=pressed_appearance,
                            passive_appearance=passive_appearance)

    execution_control.add_button_arrangement("not executing",
                                             (2, 1),
                                             (play_button, stop_button),
                                             ("executing", None),
                                             (False, True))

    execution_control.add_button_arrangement("executing",
                                             (2, 1),
                                             (pause_button, stop_button),
                                             ("paused", "not executing"))

    execution_control.add_button_arrangement("paused",
                                             (2, 1),
                                             (resume_button, stop_button),
                                             ("executing", "not executing"))

    execution_control.set_current_arrangement("executing")
    execution_control_position = (1270, 970)

    print(display.display_size, display.current_zoom_layer.display_field_size, display.current_zoom_layer.field_size)

    manipulator = world_manipulator.WorldManipulator(test_world, display, display_pos)

    # pygame.draw.rect(window, (255, 0, 0), (97, 97) + tuple(map(lambda x: x + 6, display.display_size)), 3)

    clock = pygame.time.Clock()
    tick = 0
    while True:
        tick += 1
        # print(test_world.kara_position, sub_process_control.process.is_alive())

        events = pygame.event.get()

        for event in events:
            # print(event)
            if event.type == pygame.QUIT:
                sub_process_control.stop_subprocess()
                exit()
            elif event.type == pygame.MOUSEWHEEL and False:
                print(event)
                display.zoom(event.y, np.array(pygame.mouse.get_pos()) - display_pos)
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE:
                    if sub_process_control.paused:
                        sub_process_control.resume()
                    else:
                        sub_process_control.pause()
                elif event.key == pygame.K_UP:
                    sub_process_control.set_waiting_time(sub_process_control.waiting_time + 1 / 2 ** 16)
                    print(sub_process_control.waiting_time)
                elif event.key == pygame.K_DOWN and sub_process_control.waiting_time > 0:
                    sub_process_control.set_waiting_time(sub_process_control.waiting_time - 1 / 2 ** 16)
                    print(sub_process_control.waiting_time)

        display.input_logic(events, np.array(display_pos))

        if tick == 10 ** 3 and False:
            window.blit(pygame.surfarray.make_surface(test_world.colour_layer), (0, 0))
            pygame.display.update()
            time.sleep(5)

        display.update_surface()
        # window.blit(display.current_zoom_layer.surface, (950, 300))
        display.draw(window, display_pos)

        # manipulator.logic(events)
        execution_control._logic(events, execution_control_position)
        execution_control._blit_if_necessary(window, execution_control_position)

        pygame.display.update()
        # print(display.get_field_coordinate_at_position(np.array(pygame.mouse.get_pos()) - display_pos))
        if clock.get_fps() < 55 or tick % 1000 == 0:
            print(clock.get_fps())
        # print(tick)
        clock.tick(60)
