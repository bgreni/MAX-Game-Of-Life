import os
from pathlib import Path

from max.driver import CPU, Accelerator, Tensor, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph, TensorType, ops
import numpy as np
import pygame
from argparse import *


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


    parser = ArgumentParser(
        prog="ConwayGameOfLife",
        description="Conways Game of Life implemented with MAX"
    )

    parser.add_argument('-f', '--filename', default='./starting_points/gun.txt')
    parser.add_argument('-w', '--wrap', action='store_true')

    args = parser.parse_args()

    wrap = True

    data: str
    with open(args.filename) as f:
        data = f.read().splitlines()

    PADDING = 10

    from pprint import pprint
    pprint(data)

    for i in range(len(data)):
        data[i] = ('.' * PADDING) + data[i] + ('.' * PADDING)

    pad = ["." * len(data[0]) for _ in range(PADDING)]

    data = pad + data + pad

    HEIGHT = len(data)
    WIDTH = len(data[0])

    xv = np.zeros((WIDTH, HEIGHT), dtype=np.uint8)
    for y in range(HEIGHT):
        for x in range(WIDTH):
            if data[y][x] != '.':
                xv[x, y] = 255

    graph = Graph(
        "life",
        forward=lambda x: ops.custom(
            name="conway",
            values=[x],
            out_types=[TensorType(dtype=x.dtype, shape=x.tensor.shape)],
            parameters={"wrap": args.wrap},
        )[0].tensor,
        input_types=[TensorType(output_type, shape=[WIDTH, HEIGHT])]
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

    game = GOL(xv, update, (WIDTH, HEIGHT))

    game.start()
