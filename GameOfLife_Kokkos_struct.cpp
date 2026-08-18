/**
 @file GameOfLife_Kokkos.cpp

 @brief Implementation of the Game of Life simulation with execution on GPU using Kokkos.
        In this file, we show using a more advanced View with a user defined struct as well as playing with other
          features. This is a testbed file.
**/

#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>
#include <chrono>
#include <format>
#include <iostream>
#include <map>
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

enum class MemorySpace {
  Default, // Device data access
  Host,    // Host data access
};

// Checking that Kokkos supports both enums and enum classes
enum class CellState {
  Alive, // Cell is alive
  Dead,  // And dead, as it says
};

std::map<CellState, std::string> cellStateToString = {{CellState::Alive, "X"}, {CellState::Dead, "."}};

// This struct is here to demonstrate that Kokkos views can use custom data types!
struct CellData final {
  // State
  CellState current_state;
  // Old state
  CellState previous_state;
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
  mutable bool is_valid_dual_ = false;

  /**
    @brief Create a device memory space when requested.

    Note - this is marked as const because it changes the representation
             of the underlying data, not its logical state

  @return none
  **/
  inline void init_view_dual() const {
    if (is_valid_dual_)
      return;
    // Need to create mirror view in default space and copy the data over
    // Using a mirror view prevents a double allocation if the default space is on the host
    Kokkos::View<CellData *> view_device =
        Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace::memory_space(), view_host_);
    view_dual_ = Kokkos::DualView<CellData *>(view_device, view_host_);
    is_valid_dual_ = true;
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
    is_valid_dual_ = false;
  }

  /**
    @brief Report if host memory space is up to date.

  @return bool, if host memory space is up to date.
  **/
  bool is_sync_host() const {
    if (!is_valid_dual_)
      return true;
    return view_dual_.need_sync<Kokkos::DefaultExecutionSpace>() || !view_dual_.need_sync<Kokkos::HostSpace>();
  }

  /**
    @brief Report if device memory space is up to date.

  @return bool, if device memory space is up to date.
  **/
  bool is_sync_device() const {
    if (!is_valid_dual_)
      return false;
    return view_dual_.need_sync<Kokkos::HostSpace>() || !view_dual_.need_sync<Kokkos::DefaultExecutionSpace>();
  }

  /**
    @brief Get read-only pointer to underlying data in target memory space.

    @param space Memory space to get pointer in.

  @return Read only pointer to underlying data in target memory space.
  **/
  inline const CellData *get_data(MemorySpace space = MemorySpace::Default) const {
    std::cout << std::format("\n!-- {} memory requested {}-------------------------------------------\n",
                             space == MemorySpace::Default ? "Device" : "Host",
                             space == MemorySpace::Default ? "" : "--");
    std::cout << std::format("!-- Dual view {} initialized {}--------------------------------------\n",
                             is_valid_dual_ ? "is" : "is not", is_valid_dual_ ? "----" : "");
    std::cout << std::format("!-- Host memory {} up to date {}-------------------------------------\n",
                             is_sync_host() ? "is" : "is not", is_sync_host() ? "----" : "");
    std::cout << std::format("!-- Device memory {} up to date {}-----------------------------------\n\n",
                             is_sync_device() ? "is" : "is not", is_sync_device() ? "----" : "");
    if (space == MemorySpace::Default) {
      if (!is_valid_dual_) {
        std::cout << std::format("!-- Initializing dual view --------------------------------------------\n");
        std::cout << std::format("!-- Copying from host to device ---------------------------------------\n\n");
        init_view_dual();
      }
      if (!is_sync_device())
        std::cout << std::format("!-- Copying from host to device ---------------------------------------\n\n");
      view_dual_.sync<Kokkos::DefaultExecutionSpace>();
      return view_dual_.view<Kokkos::DefaultExecutionSpace>().data();
    } else {
      if (!is_sync_host())
        std::cout << std::format("!-- Copying from device to host ---------------------------------------\n\n");
      if (!is_valid_dual_)
        return view_host_.data();
      view_dual_.sync<Kokkos::HostSpace>();
      return view_dual_.view<Kokkos::HostSpace>().data();
    }
  }

  /**
    @brief Get writable pointer to underlying data in target memory space.

    @param space Memory space to get pointer in.

  @return Writable pointer to underlying data in target memory space.
  **/
  inline CellData *get_data_writable(MemorySpace space = MemorySpace::Default) {
    CellData *data = const_cast<CellData *>(const_cast<const DataContainer &>(*this).get_data(space));

    if (is_valid_dual_) {
      if (space == MemorySpace::Default)
        view_dual_.modify<Kokkos::DefaultExecutionSpace>();
      else
        view_dual_.modify<Kokkos::HostSpace>();
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
      std::cout << cellStateToString[board[row * num_columns + column].current_state];
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
 @brief Display the current number of live cells on the board.

 @param board     The board state in an array of CellData.
 @param num_cells The number of cells on the board.

 @return none
**/
static void viewLiveCount(const CellData *board, int num_cells) {
  int count = 0;

  Kokkos::parallel_reduce(
      "live count", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_cells),
      KOKKOS_LAMBDA(const int i, int &local_count) {
        if (board[i].current_state == CellState::Alive)
          local_count += 1;
      },
      Kokkos::Sum<int>(count));

  std::cout << std::format("Total live cells: {}\n\n", count);
}

/**
 @brief Display timing statistics.

 @param times Vector of timing data, in miliseconds.
 @param name  Name of the timing data.

 @return none
**/
static void viewTimingStatistics(const std::vector<long int> &times, const std::string &name) {
  const long int min = *std::ranges::min_element(times);
  const long int max = *std::ranges::max_element(times);
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
      neighbor_count += board[neighbor_row * num_columns + neighbor_column].previous_state == CellState::Alive;
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
            board[row * num_columns + column].current_state == board[row * num_columns + column].previous_state;
        board[row * num_columns + column].previous_state = board[row * num_columns + column].current_state;
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
        auto previous_state = board[row * num_columns + column].previous_state;
        auto current_state =
            (previous_state == CellState::Alive && neighbor_count >= min_birth && neighbor_count <= max_birth) ||
                    (neighbor_count >= min_remain && neighbor_count <= max_remain)
                ? CellState::Alive
                : CellState::Dead;
        board[row * num_columns + column].current_state = current_state;
      });
}

/**
 @brief The main function that runs the Game of Life simulation.

 @return 0 on successful execution.
**/
int main(int argc, char **argv) {
  // Initialize Kokkos runtime
  Kokkos::initialize(argc, argv);

  // Timing data
  std::vector<long int> times;
  std::vector<long int> times_device_to_host;

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
      auto board_data = board.get_data_writable(MemorySpace::Host);

      for (int row = 0; row < num_rows; row++) {
        for (int column = 0; column < num_columns; column++) {
          board_data[row * num_columns + column].current_state =
              (row + column) % 2 ? CellState::Alive : CellState::Dead;
        }
      }
    } else {
      auto board_data = board.get_data_writable(MemorySpace::Host);

      for (auto cell = 0; cell < num_rows * num_columns; ++cell) {
        board_data[cell].current_state = rand() % 2 ? CellState::Alive : CellState::Dead;
      }
    }
    // And set the history data
    {
      auto board_data = board.get_data_writable(MemorySpace::Host);

      for (int row = 0; row < num_rows; row++) {
        for (int column = 0; column < num_columns; column++) {
          board_data[row * num_columns + column].current_neighbors =
              countLiveNeighbors(board_data, row, column, num_rows, num_columns);
        }
      }
    }
    std::cout << std::format("Initial Generation:\n");
    viewBoard(board.get_data(MemorySpace::Host), num_rows, num_columns);
    viewLiveCount(board.get_data(MemorySpace::Default), num_rows * num_columns);

    // Step simulation through generations
    const int num_steps = readUserInput("Enter the number of generations", 10, 1, std::numeric_limits<int>::max());

    const auto start_time = std::chrono::steady_clock::now();

    for (auto generation = 0; generation < num_steps; generation++) {
      // -- Step the simulation
      std::cout << "!-- Executing core function in Kokkos (device) memory -----------------\n";
      TimedCall(stepGeneration(board.get_data_writable(), num_rows, num_columns, min_birth, max_birth, min_remain,
                               max_remain),
                times);
      // -- And finally view the new board
      std::cout << std::format("\n\nGeneration {}:\n", generation + 1);
      TimedCall(board.get_data(MemorySpace::Host), times_device_to_host);
      viewBoard(board.get_data(MemorySpace::Host), num_rows, num_columns);
      viewLiveCount(board.get_data(MemorySpace::Default), num_rows * num_columns);
    }
    const auto stop_time = std::chrono::steady_clock::now();

    // Timing info
    std::cout << std::format("Total time: \t{} ns\n\n", timeBetween(start_time, stop_time));
  }

  // Timing info
  viewTimingStatistics(times, "step Generation");
  viewTimingStatistics(times_device_to_host, "transfer Device to Host");

  // Cleanup
  Kokkos::finalize();
  return 0;
}
