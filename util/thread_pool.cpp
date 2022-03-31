/*
This is free and unencumbered software released into the public domain.

Anyone is free to copy, modify, publish, use, compile, sell, or
distribute this software, either in source code form or as a compiled
binary, for any purpose, commercial or non-commercial, and by any
means.

In jurisdictions that recognize copyright laws, the author or authors
of this software dedicate any and all copyright interest in the
software to the public domain. We make this dedication for the benefit
of the public at large and to the detriment of our heirs and
successors. We intend this dedication to be an overt act of
relinquishment in perpetuity of all present and future rights to this
software under copyright law.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

For more information, please refer to <https://unlicense.org>
*/

#include "thread_pool.h"

ThreadPool::ThreadPool(int num_threads) {
  for (int i = 0; i < num_threads; ++i) {
    threads_.emplace_back([this]() { this->InfiniteLoop(); });
  }
}

ThreadPool::~ThreadPool() {
  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    stop_ = true;
  }
  condition_.notify_all();
  for (auto& thread : threads_) {
    thread.join();
  }
}

void ThreadPool::InfiniteLoop() {
  std::function<void()> current_job;
  while (true) {
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);

      if (queue_.empty() && !stop_) {
        // No more work currently needs to be done.
        // Release the mutex, and wait for more work or destructor.
        condition_.wait(lock, [this]() { return !queue_.empty() || stop_; });
      }

      // mutex (re)acquired
      if (stop_) {
        return;
      } else if (!queue_.empty()) {
        // there is a job ready
        current_job = std::move(queue_.front());
        queue_.pop();
      } else {
        // spurious wakeup
        continue;
      }
      // release mutex
    }

    current_job();
  }
}

void ThreadPool::Push(std::function<void()> job) {
  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    queue_.push(std::move(job));
  }
  condition_.notify_one();
}

size_t ThreadPool::NumJobsQueued() {
  std::lock_guard<std::mutex> lock(queue_mutex_);
  return queue_.size();
}
