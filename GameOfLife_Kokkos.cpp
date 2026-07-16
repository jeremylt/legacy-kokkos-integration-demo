/**
 @file GameOfLife_Kokkos.cpp

 @brief Implementation of the Game of Life simulation with execution on GPU using Kokkos.
**/

#include <Kokkos_Core.hpp>
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
KOKKOS_INLINE_FUNCTION int countLiveNeighbors(const Kokkos::View<int *> &board, const int row, const int column,
                                              const int num_rows, const int num_columns) {
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

 @return none
**/
static void stepGeneration(const Kokkos::View<int *> &current_generation, Kokkos::View<int *> &next_generation,
                           const int num_rows, const int num_columns) {
  KOKKOS_IF_ON_HOST(
      (std::cout << std::format("!-- Note: This portion of the core function is on the host ------------\n");))
  // Loop over each cell
  Kokkos::parallel_for(
      "step Generation", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {num_rows, num_columns}),
      KOKKOS_LAMBDA(const int row, const int column) {
        KOKKOS_IF_ON_HOST((std::cout << std::format("!-- Note: This portion of the kernel should be on the device if "
                                                    "you followed the instructions in the README and "
                                                    "configured Kokkos to execute on the device. This message should "
                                                    "be compiled out if Kokkos was built correctly! --\n");))
        auto neighbor_count = countLiveNeighbors(current_generation, row, column, num_rows, num_columns);

        // Grow/live if 2-3 neighbors, otherwise die
        auto is_alive_now = current_generation[row * num_columns + column];
        auto is_alive_next = (neighbor_count == 2 && is_alive_now) || neighbor_count == 3;

        next_generation[row * num_columns + column] = is_alive_next;
      });
}

/**
 @brief The main function that runs the Game of Life simulation.

 @return 0 on successful execution.
**/
int main(int argc, char **argv) {
  // Initialize Kokkos runtime
  Kokkos::initialize(argc, argv);

  // Input sizes
  const int num_rows = readUserInput("Enter the number of rows", 1080, 1, std::numeric_limits<int>::max());
  const int num_columns = readUserInput("Enter the number of columns", 1920, 1, std::numeric_limits<int>::max());

  std::cout << std::format("Total board size is {}*{} = {}.\n\n", num_rows, num_columns, num_rows * num_columns);

  // Initalize
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

  // We are making minimal modifications here to demonstrate incrementally adding Kokkos to existing code.
  // I'll call out a few items of note as the code progresses.

  // Create Kokkos views
  {
    // -- Unmanaged views are used here to show re-use of existing memory space.
    // -- We don't want to create a new host allocation and double up on the host side memory usage.
    Kokkos::View<int *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> current_generation_view_h(
        current_generation.data(), current_generation.size());
    Kokkos::View<int *> current_generation_view_d("current generation", current_generation.size());
    // -- Push from legacy CPU memory to Kokkos managed (device) memory
    std::cout << "!-- Copying current generation from Legacy (host) to Kokkos (device) --\n\n";
    Kokkos::deep_copy(current_generation_view_d, current_generation_view_h);
    // -- Here, I don't yet need a host view, because the computation will all be on device
    Kokkos::View<int *> next_generation_view_d("next generation", next_generation.size());

    // Step simulation through generations
    const int num_steps = readUserInput("steps", 10, 1, std::numeric_limits<int>::max());

    for (auto generation = 0; generation < num_steps; generation++) {
      // -- Step the simulation
      std::cout << "!-- Executing core function in Kokkos (device) memory -----------------\n";
      stepGeneration(current_generation_view_d, next_generation_view_d, num_rows, num_columns);
      std::swap(current_generation_view_d, next_generation_view_d);
      // -- And finally view the new board
      // ---- I am pulling down to host for this to use the existing legacy CPU side helper function.
      // ---- But note that each of these copies incurs a performance cost.
      std::cout << "!-- Copying current generation from Kokkos (device) to Legacy (host) --\n\n";
      Kokkos::deep_copy(current_generation_view_h, current_generation_view_d);
      std::cout << std::format("Generation {}:\n", generation + 1);
      viewBoard(current_generation, num_rows, num_columns);
    }
    // -- Here I am ensuring that the final memory state of both legacy vecs matches what the CPU only version would
    // have.
    // ---- So I create an unmanaged view of the next_generation CPU vector and then copy the device view into it.
    // ---- Note that this holds the *previous* generation at this point due to the swaps (in both the legacy code and
    //        this version).
    // ---- We would not need to do this if we did not care about exactly replicating the legacy CPU memory state.
    Kokkos::View<int *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> next_generation_view_h(
        next_generation.data(), next_generation.size());
    std::cout << "!-- Copying previous generation from Kokkos (device) to Legacy (host) --\n";
    Kokkos::deep_copy(next_generation_view_h, next_generation_view_d);
  }

  // Cleanup
  Kokkos::finalize();
  return 0;
}