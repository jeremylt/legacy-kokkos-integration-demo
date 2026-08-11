/**
 @file GameOfLife_Kokkos.cpp

 @brief Implementation of the Game of Life simulation with execution on GPU using Kokkos.
        In this file, we show using a more advanced View with a user defined struct.
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

// This struct is here to demonstrate that Kokkos views can use custom data types!
struct CellData final {
  // State
  bool is_alive;
  // Old state
  bool was_alive;
  // Did the state flip
  bool is_changed;
  // Previous neighbors
  int previous_neighbors;
  // Current state
  int current_neighbors;
};

class DataContainer final {
private:
  // Kokkos View on host only
  Kokkos::View<CellData *, Kokkos::HostSpace> view_host_;
  // Kokkos DualView
  mutable Kokkos::DualView<CellData *> view_dual_;
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
    Kokkos::View<CellData *> view_device =
        Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace::memory_space(), view_host_);
    view_dual_ = Kokkos::DualView<CellData *>(view_device, view_host_);
    is_dual_valid_ = true;
  }

public:
  /**
   @brief Initialize container with name and memory.

   @param name           Name to set for the underlying Kokkos view.
   @param container_size Size to set for the underlying Kokkos view.

 @return none
  **/
  void setup(const std::string &name, int container_size) {
    view_host_ = Kokkos::View<CellData *, Kokkos::HostSpace>(name, container_size);
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
  bool is_sync_device() const {
    if (!is_dual_valid_)
      return false;
    return view_dual_.need_sync_host() || !view_dual_.need_sync_device();
  }

  /**
    @brief Get read-only pointer to underlying data in target memory space.

    @param space Memory space to get pointer in.

  @return Read only pointer to underlying data in target memory space.
  **/
  inline const CellData *get_data(MemorySpace space = DefaultSpace) const {
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
  inline CellData *get_data_writable(MemorySpace space = DefaultSpace) {
    CellData *data = const_cast<CellData *>(const_cast<const DataContainer &>(*this).get_data(space));

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

 @param board       The board state in an array of CellData.
 @param num_rows    The number of rows in the board.
 @param num_columns The number of columns in the board.

 @return none
**/
static void viewBoard(const CellData *board, int num_rows, int num_columns) {
  // Assumes row contents are contiguous
  std::cout << "|";
  for (int column = 0; column < num_columns; column++) {
    std::cout << "-";
  }
  std::cout << "|\n";
  for (int row = 0; row < num_rows; row++) {
    std::cout << "|";
    for (int column = 0; column < num_columns; column++) {
      std::cout << (board[row * num_columns + column].is_alive ? "X" : ".");
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
KOKKOS_INLINE_FUNCTION int countLiveNeighbors(const CellData *board, const int row, const int column,
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
      neighbor_count += board[neighbor_row * num_columns + neighbor_column].was_alive;
    }
  }
  return neighbor_count;
}

/**
 @brief Advances the Game of Life simulation by one generation.

 @param cell_data   The current board.
 @param num_rows    The number of rows in the board.
 @param num_columns The number of columns in the board.
 @param min_birth   The min number of neighboring cells to be active for dead cell to activate.
 @param max_birth   The max number of neighboring cells to be active for dead cell to activate.
 @param min_remain  The min number of neighboring cells to be active for live cell to remain.
 @param max_remain  The max number of neighboring cells to be active for live cell to remain.

 @return none
**/
// The only change the the function signature is to use views instead of vectors.
static void stepGeneration(CellData *board, const int num_rows, const int num_columns, const int min_birth,
                           const int max_birth, const int min_remain, const int max_remain) {
  KOKKOS_IF_ON_HOST(
      (std::cout << std::format("!-- Note: This portion of the core function is on the host ------------\n");))
  // Loop over each cell
  // -- Need to copy old data first
  Kokkos::parallel_for(
      "save history", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {num_rows, num_columns}),
      KOKKOS_LAMBDA(const int row, const int column) {
        KOKKOS_IF_ON_HOST((std::cout << std::format("!-- Note: This portion of the kernel should be on the device if "
                                                    "you followed the instructions in the README and "
                                                    "configured Kokkos to execute on the device. This message should "
                                                    "be compiled out if Kokkos was built correctly! --\n");))
        board[row * num_columns + column].previous_neighbors = board[row * num_columns + column].current_neighbors;
        board[row * num_columns + column].is_changed =
            board[row * num_columns + column].is_alive == board[row * num_columns + column].was_alive;
        board[row * num_columns + column].was_alive = board[row * num_columns + column].is_alive;
      });
  // Loop over each cell
  // -- Then update the new current values
  Kokkos::parallel_for(
      "step Generation", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {num_rows, num_columns}),
      KOKKOS_LAMBDA(const int row, const int column) {
        auto neighbor_count = countLiveNeighbors(board, row, column, num_rows, num_columns);
        board[row * num_columns + column].current_neighbors = neighbor_count;

        // Grow/live if 2-3 neighbors, otherwise die
        // -- Note that the variables min_* and max_* are captured automatically, we only need to manage arrays
        auto was_alive = board[row * num_columns + column].was_alive;
        auto is_alive = (was_alive && neighbor_count >= min_birth && neighbor_count <= max_birth) ||
                        (neighbor_count >= min_remain && neighbor_count <= max_remain);
        board[row * num_columns + column].is_alive = is_alive;
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
  {
    DataContainer board;
    board.setup("current generation", num_rows * num_columns);

    // -- Reference checkerboard implementation
    const bool use_checkerboard =
        readUserInput("Use checkerboard pattern? (0 for no, otherwise yes)", 1, 0, std::numeric_limits<int>::max());

    if (use_checkerboard) {
      auto board_data = board.get_data_writable(HostSpace);

      for (int row = 0; row < num_rows; row++) {
        for (int column = 0; column < num_columns; column++) {
          board_data[row * num_columns + column].is_alive = (row + column) % 2;
        }
      }
    } else {
      auto board_data = board.get_data_writable(HostSpace);

      for (auto cell = 0; cell < num_rows * num_columns; ++cell) {
        board_data[cell].is_alive = rand() % 2;
      }
    }
    // And set the history data
    {
      auto board_data = board.get_data_writable(HostSpace);

      for (int row = 0; row < num_rows; row++) {
        for (int column = 0; column < num_columns; column++) {
          board_data[row * num_columns + column].current_neighbors =
              countLiveNeighbors(board_data, row, column, num_rows, num_columns);
        }
      }
    }
    std::cout << std::format("Initial Generation:\n");
    viewBoard(board.get_data(HostSpace), num_rows, num_columns);
    std::cout << std::format("!-- Host memory {} up to date {}-------------------------------------\n",
                             board.is_sync_host() ? "is" : "is not", board.is_sync_host() ? "----" : "");
    std::cout << std::format("!-- Device memory {} up to date {}-----------------------------------\n\n",
                             board.is_sync_device() ? "is" : "is not", board.is_sync_device() ? "----" : "");

    // Step simulation through generations
    const int num_steps = readUserInput("steps", 10, 1, std::numeric_limits<int>::max());

    for (auto generation = 0; generation < num_steps; generation++) {
      // -- Step the simulation
      std::cout << "!-- Executing core function in Kokkos (device) memory -----------------\n";
      stepGeneration(board.get_data_writable(), num_rows, num_columns, min_birth, max_birth, min_remain, max_remain);
      std::cout << std::format("!-- Host memory {} up to date {}-------------------------------------\n",
                               board.is_sync_host() ? "is" : "is not", board.is_sync_host() ? "----" : "");
      std::cout << std::format("!-- Device memory {} up to date {}-----------------------------------\n\n",
                               board.is_sync_device() ? "is" : "is not", board.is_sync_device() ? "----" : "");
      // -- And finally view the new board
      std::cout << std::format("Generation {}:\n", generation + 1);
      viewBoard(board.get_data(HostSpace), num_rows, num_columns);
      std::cout << std::format("!-- Host memory {} up to date {}-------------------------------------\n",
                               board.is_sync_host() ? "is" : "is not", board.is_sync_host() ? "----" : "");
      std::cout << std::format("!-- Device memory {} up to date {}-----------------------------------\n\n",
                               board.is_sync_device() ? "is" : "is not", board.is_sync_device() ? "----" : "");
    }
  }

  // Cleanup
  Kokkos::finalize();
  return 0;
}
