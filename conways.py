import os
from pathlib import Path

from max.driver import CPU, Accelerator, Tensor, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph, TensorType, ops
import numpy as np
import pygame
from sys import argv


output_type = DType.uint8

class GOL:
    def __init__(self, state, update_func, display_size):
        self.state = state
        self.update = update_func
        pygame.init()
        self.display = pygame.display.set_mode(display_size, flags=pygame.SCALED, depth=8)
        pygame.display.set_caption("Conway Game of Life")

    def start(self):
        clock = pygame.time.Clock()
        running = True
        while running:

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            surf = pygame.surfarray.make_surface(self.state)
            self.display.blit(surf, (0, 0))
            pygame.display.update()
            self.state = self.update(self.state)
            
            clock.tick(20)
        
        pygame.quit()

if __name__ == '__main__':
    if directory := os.getenv("BUILD_WORKSPACE_DIRECTORY"):
        os.chdir(directory)
    path = Path(__file__).parent / "kernels.mojopkg"

    filename: str
    if len(argv) == 2:
        filename = argv[1]
    else:
        filename = './starting_points/gun.txt'

    data: str
    with open(filename) as f:
        data = f.read().splitlines()

    HEIGHT = len(data)
    WIDTH = len(data[0])

    xv = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    for y in range(HEIGHT):
        for x in range(WIDTH):
            if data[y][x] != '.':
                xv[y, x] = 255

    graph = Graph(
        "life",
        forward=lambda x: ops.custom(
            name="conway",
            values=[x],
            out_types=[TensorType(dtype=x.dtype, shape=x.tensor.shape)],
        )[0].tensor,
        input_types=[TensorType(output_type, shape=[HEIGHT, WIDTH])]
    )
    device = CPU() if accelerator_count() == 0 else Accelerator()
    session = InferenceSession(
        devices=[device],
        custom_extensions=path,
    )
    model = session.load(graph)

    def update(state):
        x = Tensor.from_numpy(state).to(device)
        res = model.execute(x)[0]
        arr = res.to(CPU()).to_numpy()
        return arr

    game = GOL(xv, update, (HEIGHT, WIDTH))

    game.start()
