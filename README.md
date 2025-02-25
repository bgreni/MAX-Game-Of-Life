# Conways Game of Life

Conways Game of Life implemented within a MAX kernal, rendered using pygame.

## Running the program

There is a `play` magic task that will handle packaging the kernel, and running
the python script to setup the game. I accepts two arguments, `--filename`, which
points it to a text file containing the representation of the initial board state,
and `--wrap` which defines whether the board coordinates wrap around. Which can
be useful for spaceship patterns.

```
magic run play --filename ./starting_points/spaceship.txt --wrap
```