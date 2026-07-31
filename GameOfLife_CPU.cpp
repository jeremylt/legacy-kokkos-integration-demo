/**
 @file GameOfLife_CPU.cpp

 @brief Reference implementation of the Game of Life simulation with execution only on CPU.
**/

#include <format>
#include <iostream>
#include <string>
#include <vector>

/**
 @brief Reads user input for a given parameter with a default value.

 @param prompt        The input prompt for the user.
 @param default_value The default value to use if no input is provided.
 @param min           The minimum value to accept (inclusive).
 @param max           The maximum value to accept (inclusive).

 @return The user-provided value or the default value.
**/
static int readUserInput(const std::string &prompt, const int default_value, const int min, const int max) {
  std::cout << std::format("{} (default: {}): ", prompt, default_value);
  int user_value;
  std::cin >> user_value;

  // Only accept user value if it is an integer in range [min, max]
  if (std::cin.fail() || user_value < min || user_value > max) {
    std::cout << std::format("Invalid input. Using default value of {}.\n\n", default_value);
    std::cin.clear();
    std::cin.ignore();
    user_value = default_value;
  } else {
    std::cout << std::format("Accepting user input value of {}.\n\n", user_value);
  }
  return user_value;
}

/**
 @brief Displays the current state of the board.

 @param board       A vector of boolean values (encoded as integers) representing the board state.
 @param num_rows    The number of rows in the board.
 @param num_columns The number of columns in the board.

 @return none
**/
static void viewBoard(const std::vector<int> &board, const int num_rows, const int num_columns) {
  // Assumes row contents are contiguous
  std::cout << "|";
  for (int column = 0; column < num_columns; column++) {
    std::cout << "-";
  }
  std::cout << "|\n";
  for (int row = 0; row < num_rows; row++) {
    std::cout << "|";
    for (int column = 0; column < num_columns; column++) {
      std::cout << (board[row * num_columns + column] ? "X" : ".");
    }
    std::cout << "|\n";
  }
  std::cout << "|";
  for (int column = 0; column < num_columns; column++) {
    std::cout << "-";
  }
  std::cout << "|\n\n";
}

/**
 @brief Counts the number of live neighbors for a given cell.

 @param board       The current state of the board.
 @param row         The row index of the cell.
 @param column      The column index of the cell.
 @param num_rows    The number of rows in the board.
 @param num_columns The number of columns in the board.

 @return The number of live neighbors.
**/
static int countLiveNeighbors(const std::vector<int> &board, const int row, const int column, const int num_rows,
                              const int num_columns) {
  auto neighbor_count = 0;

  for (int r = row - 1; r <= row + 1; r++) {
    auto neighbor_row = (r + num_rows) % num_rows;

    for (int c = column - 1; c <= column + 1; c++) {
      auto neighbor_column = (c + num_columns) % num_columns;

      // Count all neighbor cells
      if (neighbor_row == row && neighbor_column == column) {
        continue;
      }
      neighbor_count += board[neighbor_row * num_columns + neighbor_column];
    }
  }
  return neighbor_count;
}

/**
 @brief Advances the Game of Life simulation by one generation.

 @param current_generation The current state of the board.
 @param next_generation    The next state of the board.
 @param num_rows           The number of rows in the board.
 @param num_columns        The number of columns in the board.
 @param min_birth          The min number of neighboring cells to be active for dead cell to activate.
 @param max_birth          The max number of neighboring cells to be active for dead cell to activate.
 @param min_remain         The min number of neighboring cells to be active for live cell to remain.
 @param max_remain         The max number of neighboring cells to be active for live cell to remain.

 @return none
**/
static void stepGeneration(const std::vector<int> &current_generation, std::vector<int> &next_generation,
                           const int num_rows, const int num_columns, const int min_birth, const int max_birth,
                           const int min_remain, const int max_remain) {
  // Loop over each cell
  for (int row = 0; row < num_rows; row++) {
    for (int column = 0; column < num_columns; column++) {
      auto neighbor_count = countLiveNeighbors(current_generation, row, column, num_rows, num_columns);

      // Grow/live if 2-3 neighbors, otherwise die
      auto is_alive_now = current_generation[row * num_columns + column];
      auto is_alive_next = (is_alive_now && neighbor_count >= min_birth && neighbor_count <= max_birth) ||
                           (neighbor_count >= min_remain && neighbor_count <= max_remain);

      next_generation[row * num_columns + column] = is_alive_next;
    }
  }
}

/**
 @brief The main function that runs the Game of Life simulation.

 @return 0 on successful execution.
**/
int main(int argc, char **argv) {
  // Input sizes
  const int num_rows = readUserInput("Enter the number of rows", 1080, 1, std::numeric_limits<int>::max());
  const int num_columns = readUserInput("Enter the number of columns", 1920, 1, std::numeric_limits<int>::max());

  std::cout << std::format("Total board size is {}*{} = {}.\n\n", num_rows, num_columns, num_rows * num_columns);

  // Birth and remain rules
  const int min_birth = readUserInput("Enter the minimum number of live neighbors to trigger birth", 2, 0, 8);
  const int max_birth = readUserInput("Enter the maximum number of live neighbors to trigger birth", 3, 0, 8);
  const int min_remain = readUserInput("Enter the minimum number of live neighbors for cell retention", 3, 0, 8);
  const int max_remain = readUserInput("Enter the maximum number of live neighbors for cell retention", 3, 0, 8);

  // Initialize
  // -- Note, using ints here instead of bools because Kokkos does not have views (vecs) of bools
  std::vector<int> current_generation(num_rows * num_columns, 0);
  std::vector<int> next_generation(num_rows * num_columns, 0);

  // -- Reference checkerboard implementation
  const bool use_checkerboard =
      readUserInput("Use checkerboard pattern? (0 for no, otherwise yes)", 1, 0, std::numeric_limits<int>::max());

  if (use_checkerboard) {
    for (int row = 0; row < num_rows; row++) {
      for (int column = 0; column < num_columns; column++) {
        current_generation[row * num_columns + column] = (row + column) % 2;
      }
    }
  } else {
    for (auto &cell : current_generation) {
      cell = rand() % 2;
    }
  }
  std::cout << std::format("Initial Generation:\n");
  viewBoard(current_generation, num_rows, num_columns);

  // Step simulation through generations
  const int num_steps = readUserInput("steps", 10, 1, std::numeric_limits<int>::max());

  for (auto generation = 0; generation < num_steps; generation++) {
    // -- Step the simulation
    stepGeneration(current_generation, next_generation, num_rows, num_columns, min_birth, max_birth, min_remain,
                   max_remain);
    std::swap(current_generation, next_generation);
    // -- And view the new board
    viewBoard(current_generation, num_rows, num_columns);
  }
  return 0;
}