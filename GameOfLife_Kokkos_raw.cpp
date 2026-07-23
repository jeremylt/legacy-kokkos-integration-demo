/**
 @file GameOfLife_Kokkos.cpp

 @brief Implementation of the Game of Life simulation with execution on GPU using Kokkos.
        In this file, the memory management logic is exposed more readily to the reader.
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
// I am not sure the best way to compile away this function annotation if we don't want to pull in Kokkos as a
//   dependency, though that's not the case for the current code (4C) that I am considering.
// The only change the the function inputs is to use a view instead of a vector.
KOKKOS_INLINE_FUNCTION int countLiveNeighbors(const Kokkos::View<int *> &board, const int row, const int column,
                                              const int num_rows, const int num_columns) {
  KOKKOS_IF_ON_HOST((std::cout << std::format("!-- Note: This portion of the kernel should be on the device if "
                                              "you followed the instructions in the README and "
                                              "configured Kokkos to execute on the device. This message should "
                                              "be compiled out if Kokkos was built correctly! --\n");))
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
// The only change the the function signature is to use views instead of vectors.
static void stepGeneration(const Kokkos::View<int *> &current_generation, Kokkos::View<int *> &next_generation,
                           const int num_rows, const int num_columns) {
  KOKKOS_IF_ON_HOST(
      (std::cout << std::format("!-- Note: This portion of the core function is on the host ------------\n");))
  // Loop over each cell
  // -- This whole loop has been moved over onto the device. This was previously a pair of nested loops, but
  //      here Kokkos gets to decide how to distribute the loop bodies based upon the resources it has.
  // -- Note that we really do want as much independence, particularly in terms of write access, as possible so that
  //      Kokkos can do a good job parallelizing the work here and not having to synchronize any threads.
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
  @brief Helper to sync from Legacy (host) vector to Kokkos (device) view.

  @param legacy The legacy vector to copy from.
  @param view   The Kokkos view to copy into.

  @return none
**/
static void syncFromLegacyHost(std::vector<int> legacy, Kokkos::View<int *> view) {
  //  Here I am creating a temporary Kokkos view using the memory space of the Legacy (host) vector
  //    and immediately the contents to the Kokkos execution space (device).
  Kokkos::deep_copy(view, Kokkos::View<int *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
                              legacy.data(), legacy.size()));
}

/**
  @brief Helper to sync from Kokkos (device) view to Legacy (host) vector.

  @param view   The Kokkos view to copy from.
  @param legacy The legacy vector to copy into.

  @return none
**/
static void syncToLegacyHost(const Kokkos::View<int *> view, std::vector<int> &legacy) {
  // And this function is the opposite - I am creating a temporary Kokkos view using the memory space of the
  //   Legacy (host) vector, but copying from the Kokkos execution space (device).
  Kokkos::deep_copy(
      Kokkos::View<int *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(legacy.data(), legacy.size()),
      view);
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

  // We are making minimal modifications here to demonstrate incrementally adding Kokkos to existing code.
  // I'll call out a few items of note as the code progresses.

  // Create Kokkos views
  // -- Note that we are creating a scope here to ensure that the views are destroyed before the underlying host memory
  //      vectors are freed. This sort of thing is one of those classic 'gotchas' in managing multiple memory spaces.
  {
    // -- Performance note: allocating memory has a cost. So if we are going to enter this function multiple times, then
    //      it would be a good idea to cache these views somewhere. That's why these views are created *outside*
    //      of the loop over the number of generations given below! In general, we want to avoid
    //        1) excessive/duplicate allocations
    //        2) any unneeded copies back and forth between host/device
    //      which does encourage us to do as much computation on the device once we have 'paid' the cost of shipping
    //      the data all the way up to the device. So it is better to port contiguous regions of computation.
    Kokkos::View<int *> current_generation_view("current generation", current_generation.size());
    // -- Push from legacy CPU memory to Kokkos managed (device) memory
    std::cout << "!-- Copying current generation from Legacy (host) to Kokkos (device) --\n\n";
    syncFromLegacyHost(current_generation, current_generation_view);
    // -- Here, I am just creating a Kokkos view, as we don't need any host side memory (yet!)
    Kokkos::View<int *> next_generation_view("next generation", next_generation.size());

    // Step simulation through generations
    const int num_steps = readUserInput("steps", 10, 1, std::numeric_limits<int>::max());

    for (auto generation = 0; generation < num_steps; generation++) {
      // -- Step the simulation
      std::cout << "!-- Executing core function in Kokkos (device) memory -----------------\n";
      stepGeneration(current_generation_view, next_generation_view, num_rows, num_columns);
      std::swap(current_generation_view, next_generation_view);
      // -- And finally view the new board
      // ---- I am pulling down to host for this to use the existing legacy CPU side helper function.
      //      Note that each of these copies incurs a performance cost.
      std::cout << "!-- Copying current generation from Kokkos (device) to Legacy (host) --\n\n";
      syncToLegacyHost(current_generation_view, current_generation);
      std::cout << std::format("Generation {}:\n", generation + 1);
      viewBoard(current_generation, num_rows, num_columns);
    }
    // -- Here I am ensuring that the final memory state of both legacy vecs matches what the CPU only version has.
    //      Note that this holds the *previous* generation at this point due to the swaps (in both the legacy code and
    //      this version), so really the variable name is potentially misleading. I didn't have a quick fix :shrug:.
    //      We would not need to do this if we did not care about exactly replicating the legacy CPU memory state.
    std::cout << "!-- Copying previous generation from Kokkos (device) to Legacy (host) --\n";
    syncToLegacyHost(next_generation_view, next_generation);
  }

  // Cleanup
  Kokkos::finalize();
  return 0;
}