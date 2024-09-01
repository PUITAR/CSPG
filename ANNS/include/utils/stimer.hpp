#ifndef INCLUDE_STIMER_HPP
#define INCLUDE_STIMER_HPP

#include <chrono>

namespace utils
{

  class STimer
  {
  private:
    using clock = std::chrono::high_resolution_clock;
    using time_point = std::chrono::time_point<clock>;
    time_point t1_;
    double total_;

  public:
    /// @brief A tiny timer to test runtime in "second" unit.
    STimer() : total_(0) {}

    void Reset() { total_ = 0; }

    void Start() { t1_ = clock::now(); }

    void Stop()
    {
      total_ += (std::chrono::duration<double, std::milli>(clock::now() - t1_).count() / 1000);
    }

    double GetTime() { return total_; }
  };

};

#endif