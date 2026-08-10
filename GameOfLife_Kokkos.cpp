/**
 @file GameOfLife_Kokkos.cpp

 @brief Implementation of the Game of Life simulation with execution on GPU using Kokkos.
**/

#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>
#include <format>
#include <iostream>
#include <string>
#include <vector>

enum MemorySpace {
  DefaultSpace, // Device data access
  HostSpace,    // Host data access
};

class DataContainer final {
private:
  // Kokkos View on host only
  Kokkos::View<bool *, Kokkos::HostSpace> view_host_;
  // Kokkos DualView
  mutable Kokkos::DualView<bool *> view_dual_;
  // Flag for valid DualView
  mutable bool is_dual_valid_ = false;

  /**
    @brief Create a device memory space when requested.

    Note - this is marked as const because it changes the representation
             of the underlying data, not its logical state

  @return none
  **/
  inline void init_dual_view() const {
    if (is_dual_valid_)
      return;
    // Need to create mirror view in default space and copy the data over
    // Using a mirror view prevents a double allocation if the default space is on the host
    Kokkos::View<bool *> view_device =
        Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace::memory_space(), view_host_);
    view_dual_ = Kokkos::DualView<bool *>(view_device, view_host_);
    is_dual_valid_ = true;
  }

public:
  /**
   @brief Initialize container with name and memory.

   @param name            Name to set for the underlying Kokkos view.
   @param container_size  Size to set for the underlying Kokkos view.

 @return none
  **/
  void setup(const std::string &name, int container_size) {
    view_host_ = Kokkos::View<bool *, Kokkos::HostSpace>(name, container_size);
    is_dual_valid_ = false;
  }

  /**
    @brief Report if host memory space is up to date.

  @return bool, if host memory space is up to date.
  **/
  bool is_sync_host() const {
    if (!is_dual_valid_)
      return true;
    return view_dual_.need_sync_device() || !view_dual_.need_sync_host();
  }

  /**
    @brief Report if device memory space is up to date.

  @return bool, if device memory space is up to date.
  **/
  bool is_device_host() const {
    if (!is_dual_valid_)
      return false;
    return view_dual_.need_sync_host() || !view_dual_.need_sync_device();
  }

  /**
    @brief Get read-only pointer to underlying data in target memory space.

    @param space Memory space to get pointer in.

  @return Read only pointer to underlying data in target memory space.
  **/
  inline const bool *get_data(MemorySpace space = DefaultSpace) const {
    if (space == DefaultSpace) {
      if (!is_dual_valid_) {
        init_dual_view();
      }
      view_dual_.sync_device();
      return view_dual_.view_device().data();
    } else {
      if (!is_dual_valid_) {
        return view_host_.data();
      }
      view_dual_.sync_host();
      return view_dual_.view_host().data();
    }
  }

  /**
    @brief Get writable pointer to underlying data in target memory space.

    @param space Memory space to get pointer in.

  @return Writable pointer to underlying data in target memory space.
  **/
  inline bool *get_data_writable(MemorySpace space = DefaultSpace) {
    bool *data = const_cast<bool *>(const_cast<const DataContainer &>(*this).get_data(space));

    if (is_dual_valid_) {
      if (space == DefaultSpace) {
        view_dual_.modify_device();
      } else {
        view_dual_.modify_host();
      }
    }
    return data;
  }
};

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
static void viewBoard(const bool *board, const int num_rows, const int num_columns) {
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
KOKKOS_INLINE_FUNCTION int countLiveNeighbors(const bool *board, const int row, const int column, const int num_rows,
                                              const int num_columns) {
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
 @param min_birth          The min number of neighboring cells to be active for dead cell to activate.
 @param max_birth          The max number of neighboring cells to be active for dead cell to activate.
 @param min_remain         The min number of neighboring cells to be active for live cell to remain.
 @param max_remain         The max number of neighboring cells to be active for live cell to remain.

 @return none
**/
// The only change the the function signature is to use views instead of vectors.
static void stepGeneration(const bool *current_generation, bool *next_generation, const int num_rows,
                           const int num_columns, const int min_birth, const int max_birth, const int min_remain,
                           const int max_remain) {
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
        // -- Note that the variables min_* and max_* are captured automatically, we only need to manage arrays
        auto is_alive_now = current_generation[row * num_columns + column];
        auto is_alive_next = (is_alive_now && neighbor_count >= min_birth && neighbor_count <= max_birth) ||
                             (neighbor_count >= min_remain && neighbor_count <= max_remain);

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

  // Birth and remain rules
  const int min_birth = readUserInput("Enter the minimum number of live neighbors to trigger birth", 2, 0, 8);
  const int max_birth = readUserInput("Enter the maximum number of live neighbors to trigger birth", 3, 0, 8);
  const int min_remain = readUserInput("Enter the minimum number of live neighbors for cell retention", 3, 0, 8);
  const int max_remain = readUserInput("Enter the maximum number of live neighbors for cell retention", 3, 0, 8);

  // Initialize
  // -- Note, using ints here instead of bools because Kokkos does not have views (vecs) of bools
  {
    // In this version, I am using a small wrapper around the Kokkos DualView to make the changes to the
    //   underlying code less intrusive. In this case, if the code had been built to use pointers to
    //   arrays from the start, then only the lines requesting array access would need to be changed.
    DataContainer current_generation, next_generation;
    current_generation.setup("current generation", num_rows * num_columns);
    next_generation.setup("next generation", num_rows * num_columns);

    // -- Reference checkerboard implementation
    const bool use_checkerboard =
        readUserInput("Use checkerboard pattern? (0 for no, otherwise yes)", 1, 0, std::numeric_limits<int>::max());

    if (use_checkerboard) {
      auto current_generation_data = current_generation.get_data_writable(HostSpace);

      for (int row = 0; row < num_rows; row++) {
        for (int column = 0; column < num_columns; column++) {
          current_generation_data[row * num_columns + column] = (row + column) % 2;
        }
      }
    } else {
      auto current_generation_data = current_generation.get_data_writable(HostSpace);

      for (auto cell = 0; cell < num_rows * num_columns; ++cell) {
        current_generation_data[cell] = rand() % 2;
      }
    }
    std::cout << std::format("Initial Generation:\n");
    viewBoard(current_generation.get_data(HostSpace), num_rows, num_columns);
    std::cout << std::format("!-- Host memory {} up to date {}-------------------------------------\n",
                             current_generation.is_sync_host() ? "is" : "is not",
                             current_generation.is_sync_host() ? "----" : "");
    std::cout << std::format("!-- Device memory {} up to date {}-----------------------------------\n\n",
                             current_generation.is_device_host() ? "is" : "is not",
                             current_generation.is_device_host() ? "----" : "");

    // Step simulation through generations
    const int num_steps = readUserInput("steps", 10, 1, std::numeric_limits<int>::max());

    for (auto generation = 0; generation < num_steps; generation++) {
      // -- Step the simulation
      std::cout << "!-- Executing core function in Kokkos (device) memory -----------------\n";
      stepGeneration(current_generation.get_data(), next_generation.get_data_writable(), num_rows, num_columns,
                     min_birth, max_birth, min_remain, max_remain);
      std::swap(current_generation, next_generation);
      std::cout << std::format("!-- Host memory {} up to date {}-------------------------------------\n",
                               current_generation.is_sync_host() ? "is" : "is not",
                               current_generation.is_sync_host() ? "----" : "");
      std::cout << std::format("!-- Device memory {} up to date {}-----------------------------------\n\n",
                               current_generation.is_device_host() ? "is" : "is not",
                               current_generation.is_device_host() ? "----" : "");
      // -- And finally view the new board
      std::cout << std::format("Generation {}:\n", generation + 1);
      viewBoard(current_generation.get_data(HostSpace), num_rows, num_columns);
      std::cout << std::format("!-- Host memory {} up to date {}-------------------------------------\n",
                               current_generation.is_sync_host() ? "is" : "is not",
                               current_generation.is_sync_host() ? "----" : "");
      std::cout << std::format("!-- Device memory {} up to date {}-----------------------------------\n\n",
                               current_generation.is_device_host() ? "is" : "is not",
                               current_generation.is_device_host() ? "----" : "");
    }
  }

  // Cleanup
  Kokkos::finalize();
  return 0;
}
