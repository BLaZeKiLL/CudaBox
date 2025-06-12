#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

inline std::shared_ptr<spdlog::logger> GetLogger() {
  static std::shared_ptr<spdlog::logger> logger = [] {
    auto fmt = "[%Y-%m-%d %H:%M:%S.%f] [%n] [%^%l%$] %v";
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    console_sink->set_pattern(fmt);

    // Optional: control sink log level here
    console_sink->set_level(spdlog::level::trace); // or debug/info/etc.

    auto new_logger = std::make_shared<spdlog::logger>("cudabox", console_sink);
    new_logger->set_level(spdlog::level::trace); // Set logger's level
    spdlog::set_default_logger(new_logger);
    return new_logger;
  }();
  return logger;
}

#define CUDABOX_LOG_TRACE(...) GetLogger()->trace(__VA_ARGS__)
#define CUDABOX_LOG_DEBUG(...) GetLogger()->debug(__VA_ARGS__)
#define CUDABOX_LOG_INFO(...) GetLogger()->info(__VA_ARGS__)
#define CUDABOX_LOG_WARN(...) GetLogger()->warn(__VA_ARGS__)
#define CUDABOX_LOG_ERROR(...) GetLogger()->error(__VA_ARGS__)
#define CUDABOX_LOG_CRITICAL(...) GetLogger()->critical(__VA_ARGS__)
