/**
 @file GameOfLife_CPU.cpp

 @brief Reference implementation of the Game of Life simulation with execution only on CPU.
**/

#include <chrono>
#include <format>
#include <iostream>
#include <numeric>
#include <ranges>
#include <string>
#include <vector>

/**
 @brief Wrap function call in timing data collection macro.

 @param call  Function call to execute.
 @param times The array of timing values.

 @return None.
**/
#define TimedCall(call_, times_)                                                                                       \
  {                                                                                                                    \
    const auto start_ = std::chrono::steady_clock::now();                                                              \
    call_;                                                                                                             \
    const auto stop_ = std::chrono::steady_clock::now();                                                               \
    times_.push_back(timeBetween(start_, stop_));                                                                      \
  }

/**
 @brief Count nanoseconds between two times.

 @param start The start time.
 @param stop  The stop time.

 @return The number of nanoseconds between the two times.
**/
static long int timeBetween(const std::chrono::steady_clock::time_point &start,
                            const std::chrono::steady_clock::time_point &stop) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
}

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

 @param board       A vector of boolean values representing the board state.
 @param num_rows    The number of rows in the board.
 @param num_columns The number of columns in the board.

 @return none
**/
static void viewBoard(const std::vector<bool> &board, const int num_rows, const int num_columns) {
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
 @brief Display timing statistics.

 @param times Vector of timing data, in miliseconds.
 @param name  Name of the timing data.

 @return none
**/
static void viewTimingStatistics(const std::vector<long int> &times, const std::string &name) {
  const long int min = *std::min_element(times.begin(), times.end());
  const long int max = *std::max_element(times.begin(), times.end());
  double average = 0;

  for (auto time : times) {
    average += (1.0 * time) / times.size();
  }

  std::cout << std::format("Timing information, {}:\n", name);
  std::cout << std::format("  min: \t\t{} ns\n", min);
  std::cout << std::format("  max: \t\t{} ns\n", max);
  std::cout << std::format("  average: \t{} ns\n", average);
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
static int countLiveNeighbors(const std::vector<bool> &board, const int row, const int column, const int num_rows,
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
static void stepGeneration(const std::vector<bool> &current_generation, std::vector<bool> &next_generation,
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
  // Timing data
  std::vector<long int> times;

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
  std::vector<bool> current_generation(num_rows * num_columns, false);
  std::vector<bool> next_generation(num_rows * num_columns, false);

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
    for (auto cell = 0; cell < num_rows * num_columns; ++cell) {
      current_generation[cell] = rand() % 2;
    }
  }
  std::cout << std::format("Initial Generation:\n");
  viewBoard(current_generation, num_rows, num_columns);

  // Step simulation through generations
  const int num_steps = readUserInput("Enter the number of generations", 10, 1, std::numeric_limits<int>::max());

  const auto start_time = std::chrono::steady_clock::now();

  for (auto generation = 0; generation < num_steps; generation++) {
    // -- Step the simulation
    TimedCall(stepGeneration(current_generation, next_generation, num_rows, num_columns, min_birth, max_birth,
                             min_remain, max_remain),
              times);
    std::swap(current_generation, next_generation);
    // -- And view the new board
    viewBoard(current_generation, num_rows, num_columns);
  }
  const auto stop_time = std::chrono::steady_clock::now();

  // Timing info
  std::cout << std::format("Total time: \t{}\n\n", timeBetween(start_time, stop_time));
  viewTimingStatistics(times, "step Generation");

  return 0;
}
