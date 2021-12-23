from __future__ import annotations

import multiprocessing
from copy import deepcopy
from dataclasses import dataclass
from functools import total_ordering
from random import randint
from typing import List, Tuple, Generator

import numpy as np
from botcity.web import WebBot
from botcity.web.cv2find import Box

labels = ['closed', 'empty', 'flag'] + [str(n) for n in range(1, 9)]


@dataclass
@total_ordering
class Cell:
    content: str
    element: Box

    def __same_height(self, other: Cell):
        return abs(self.element.top - other.element.top) < self.element.height / 4

    def __same_width(self, other: Cell):
        return abs(self.element.left - other.element.left) < self.element.width / 4

    def __lt__(self, other: Cell):
        above = other.element.top - self.element.top >= self.element.height / 4
        left = other.element.left - self.element.left >= self.element.width / 4
        return above or (self.__same_height(other) and left)

    def __eq__(self, other: Cell):
        return self.__same_height(other) and self.__same_width(other)


class Bot(WebBot):
    def __init__(self):
        # Driver Options
        super().__init__()
        self.headless = False
        self.driver_path = "./chromedriver.exe"

        # Game info
        self.updates = 0
        self.kickstart = True
        self.restarts = 0
        self.height = 16
        self.width = 30
        self.grid = np.empty((self.height, self.width), Cell)

    def update_cell(self, cell: Cell, content: str):
        cell.content = content
        self.state.element = cell.element
        if content == 'flag':
            self.right_click()
        else:
            self.click()

    def grab_surrounding_cells(self, row: int, col: int, content: str, grid: np.ndarray = None) -> List[Cell]:
        # init
        grid = self.grid if grid is None else grid
        cells = []

        # For each cell in a 3x3 area around a cell...
        for i in range(row - 1, row + 2):
            for j in range(col - 1, col + 2):
                # Skips the central cell
                if i == row and j == col:
                    continue

                # Skips out of bound cells
                if i < 0 or j < 0 or i >= grid.shape[0] or j >= grid.shape[1]:
                    continue

                # Counts the number of surrounding cells that matches what we're looking for
                if grid[i][j].content == content:
                    cells.append(grid[i][j])

        return cells

    def flags_around(self, row: int, col: int, grid: np.ndarray = None) -> int:
        return len(self.grab_surrounding_cells(row, col, 'flag', grid))

    def closed_cells_around(self, row: int, col: int, grid: np.ndarray = None) -> List[Cell]:
        return self.grab_surrounding_cells(row, col, 'closed', grid)

    def random_cell(self) -> Cell:
        return self.grid[randint(0, self.height - 1)][randint(0, self.width - 1)]

    def restart(self):
        print("Restarting the game. Attempt #", self.restarts + 2)
        self.kickstart = True
        self.restarts += 1
        self.updates = 0
        self.click()

    def simulation(self, starting_cell: Tuple[int, int], guess: str) -> bool:
        # Init
        played = True
        grid = deepcopy(self.grid)
        grid[starting_cell].content = guess

        while played:
            played = False
            # For each field of the game grid...
            for (row, col), cell in np.ndenumerate(grid):
                # Looks for a numeric cell
                if cell.content in ['closed', 'empty', 'flag', 'open']:
                    continue

                # Grabs information about the cell's surroundings
                remaining_bombs = int(cell.content) - self.flags_around(row, col, grid)
                closed_cells = self.closed_cells_around(row, col, grid)

                # Integrity verification
                if remaining_bombs < 0 or remaining_bombs > len(closed_cells):
                    return False

                # Safe land exploration
                if remaining_bombs == 0:
                    for safe_land in closed_cells:
                        safe_land.content = 'open'
                        played = True

                # Bomb discovery
                elif remaining_bombs == len(closed_cells):
                    for hidden_bomb in closed_cells:
                        hidden_bomb.content = 'flag'
                        played = True

        return True

    def find_all_helper(self, label) -> List[Cell]:
        findings = self.find_all(label, matching=0.98, waiting_time=100)
        return [Cell(content=label, element=element) for element in findings]

    def update_grid(self, game_over_restart: bool = True):
        # Init
        self.grid = []
        expected_size = self.height * self.width

        # Finds all the cells in the grid
        for label in labels:
            cells = self.find_all_helper(label)
            self.grid.extend(cells)
            if len(self.grid) >= self.height * self.width:
                break

        # Checks for a Game over
        if len(self.grid) < expected_size:
            # If it's not game over, then it's an unexpected error
            if not self.find('game_over'):
                raise ValueError("An unexpected error has occurred: some fields were not found by the update_grid().")

            # Log
            if not self.kickstart:
                print("Game Over!")

            # Restart if allowed
            if game_over_restart:
                self.restart()
                return self.update_grid(False)

            # Raises if unable to restart
            raise ValueError("Unable to restart the game!")

        # Integrity verification
        assert len(self.grid) == expected_size

        # Sorts the Cells and reshapes them into a grid (numpy array)
        self.grid.sort()
        self.grid = np.array(self.grid, Cell)
        self.grid = self.grid.reshape((self.height, self.width))

    def play_turn(self):
        # Init
        played_once = False
        played = True

        while played:
            # Plays until it needs to update the GUI
            played = False

            # For each field of the game grid...
            for (row, col), cell in np.ndenumerate(self.grid):
                # Looks for a numeric cell
                if cell.content in ['closed', 'empty', 'flag', 'open']:
                    continue

                # Grabs information about the cell's surroundings
                remaining_bombs = int(cell.content) - self.flags_around(row, col)
                closed_cells = self.closed_cells_around(row, col)

                # Integrity verification
                assert 0 <= remaining_bombs <= len(closed_cells)

                # Safe land exploration
                if remaining_bombs == 0:
                    for safe_land in closed_cells:
                        self.update_cell(safe_land, 'open')
                        played = True
                        played_once = True

                # Bomb discovery
                elif remaining_bombs == len(closed_cells):
                    for hidden_bomb in closed_cells:
                        self.update_cell(hidden_bomb, 'flag')
                        self.kickstart = False
                        played = True
                        played_once = True

        # Advanced mechanics
        if not played_once and not self.kickstart:
            # Finds a start point for a simulation
            for (row, col), cell in np.ndenumerate(self.grid):
                # Looks for a numeric cell
                if cell.content in ['closed', 'empty', 'flag', 'open']:
                    continue

                # For each cell in a 3x3 area around the cell...
                for i in range(row - 1, row + 2):
                    for j in range(col - 1, col + 2):
                        # Skips the central cell
                        if i == row and j == col:
                            continue

                        # Skips out of bound cells
                        if i < 0 or j < 0 or i >= self.height or j >= self.width:
                            continue

                        # Look for closed cells only
                        if self.grid[i][j].content != 'closed':
                            continue

                        # Simulates the cell being a safe land. If the simulation fails, then the cell has a bomb.
                        if not self.simulation((i, j), 'open'):
                            print(f"Performing advanced play at cell ({i}, {j})")
                            self.update_cell(self.grid[i][j], 'flag')
                            played_once = True

                        # Simulates the cell containing a bomb. If the simulation fails, the cell is a safe land.
                        elif not self.simulation((i, j), 'flag'):
                            print(f"Performing advanced play at cell ({i}, {j})")
                            self.update_cell(self.grid[i][j], 'open')
                            played_once = True

        # Out of options
        if not played_once:
            # Grabs a random closed cell
            random_cell = self.random_cell()
            while random_cell.content != 'closed':
                random_cell = self.random_cell()

            # Log
            if not self.kickstart:
                print("Clicking at a random cell")

            # Clicks on it
            self.state.element = random_cell.element
            self.click()

    def action(self, execution=None):
        # Opens the Minesweeper website.
        self.browse("https://minesweeper.online/start/3")
        self.maximize_window()
        self.wait(1000)

        # Plays the Game
        while True:
            # Updates the grid
            self.update_grid()
            self.play_turn()

            # Good Ending
            if not self.kickstart and self.find("game_finished", matching=0.98, waiting_time=500):
                print("Game Finished successfully!")
                break

            # Unstuck
            if self.updates > len(self.grid):
                raise ValueError("Error: The bot was stuck for unknown reasons")

        # Wait for 10 seconds before closing
        self.wait(10000)

        # Stop the browser and clean up
        self.stop_browser()


if __name__ == '__main__':
    Bot.main()
