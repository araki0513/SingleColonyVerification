#include "ThreadPool.hpp"

ThreadPool::ThreadPool(int n_threads) {
    this->tasks = std::make_unique<std::queue<std::function<void()>>>();
    this->workers.reserve(n_threads);
    for (int i = 0; i < n_threads; ++i)
        workers.emplace_back(&ThreadPool::worker, this);
}

void ThreadPool::worker() {
    while (true) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(this->queue_mutex);    //给任务队列加锁，以保证线程安全，使用{}作用域自动释放锁
            this->condition.wait(lock, [this]() { return this->stop || !this->tasks->empty(); });   //等待条件变量，直到线程池停止或任务队列不为空
            if (this->stop && this->tasks->empty()) return;     //如果线程池停止且任务队列为空，退出线程
            task = std::move(this->tasks->front());     //获取任务队列的第一个任务
            this->tasks->pop();     //从任务队列中移除已获取的任务
        }
        task(); //执行任务
    }
}

ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(this->queue_mutex);
        this->stop = true;
    }
    this->condition.notify_all();
    for (auto& t : this->workers)
        if (t.joinable()) t.join();
}