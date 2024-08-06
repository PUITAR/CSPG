#ifndef INCLUDE_STIMER_HPP
#define INCLUDE_STIMER_HPP

#include <chrono>

namespace utils {

class STimer
{
private:
    using clock = std::chrono::high_resolution_clock;
    using time_point = std::chrono::time_point<clock>;
    time_point t1_;
    double total_;

public:
    STimer();
    void Reset();
    void Start();
    void Stop();
    double GetTime();

};

};

#endif