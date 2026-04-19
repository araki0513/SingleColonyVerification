#include <thread>
#include <vector>
#include <queue>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <future>

class ThreadPool {
public:
    ThreadPool(int n_threads);
    ~ThreadPool();
    template<class Fn, class... Args>
    auto enqueue(Fn&& fn, Args&&... args) 
    -> std::future<typename std::invoke_result<Fn,Args...>::type>
    {
        using return_type = typename std::invoke_result<Fn, Args...>::type;
        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<Fn>(fn), std::forward<Args>(args)...)
        );
        {
            std::unique_lock<std::mutex> lock(this->queue_mutex);
            if (this->stop) {
                throw std::runtime_error("enqueue on stopped ThreadPool");
            }
            //this->tasks->emplace(std::bind(std::forward<Fn>(fn), std::forward<Args>(args)...));
            tasks->emplace([task](){ (*task)(); });
        }
        this->condition.notify_one();
        return task->get_future();
    }
private:
    void worker();
    std::vector<std::thread> workers;
    std::unique_ptr<std::queue<std::function<void()>>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop = false;
};
