#include <utils/stimer.hpp>


namespace utils {


/// @brief A tiny timer to test runtime in "second" unit.
STimer::STimer() : total_(0) {}

void STimer::Reset() { total_ = 0; }

void STimer::Start() { t1_ = clock::now(); }

void STimer::Stop()
{
    total_ += (
        std::chrono::duration<double, std::milli> (
            clock::now() - t1_
        ).count() / 1000
    );
}

double STimer::GetTime() { return total_; }

};