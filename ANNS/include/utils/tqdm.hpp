#ifndef INCLUDE_TQDM_HPP
#define INCLUDE_TQDM_HPP

#include <string>
#include <iostream>

namespace utils
{

class TQDM {
private:
    size_t N_{0};
    size_t curr_{0};
    std::string name_;
    std::string bar_;
    size_t TQDM_WIDTH{50};

public:
    explicit TQDM(size_t N, const std::string & name = "") : 
        N_(N), name_(name)
    {
        if (N < TQDM_WIDTH) TQDM_WIDTH = N;
        BarRebuild();
        curr_ = 0;  
        std::cout << name_ << bar_;
    }

    void Reset() {
        BarRebuild();
        curr_ = 0;
        UpdateBar();
    }

    void Next() {
        curr_ ++;
        if (curr_ > N_) return ;
        size_t idx = curr_ * TQDM_WIDTH / N_;
        if (idx) bar_[idx] = '=';
        UpdateBar();
    }

    ~TQDM() {
        std::cout << std::endl;
    }

private:
    inline void BarRebuild() {
        bar_ = std::string(TQDM_WIDTH + 2, ' ');
        bar_[0] = '[';
        bar_[TQDM_WIDTH + 1] = ']';
    }

    inline void UpdateBar() const {
        for (size_t i = 0; i < TQDM_WIDTH + 2; i++) {
            std::cout << '\b';
        }
        std::cout << bar_;
        flush(std::cout);
    }

};

} // namespace utils


#endif