#pragma once

#ifndef NDEBUG // debug
#define SPDLOG_TRACE_ON
#define __SNOW_LOG_LEVEL__ spdlog::level::trace
#else // release
#define __SNOW_LOG_LEVEL__ spdlog::level::info
#endif
#include <spdlog/spdlog.h>
#include <spdlog/fmt/ostr.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <vector>
#include <string>


/* get loggers of spdlog and set level */
std::vector<std::shared_ptr<spdlog::logger>> &GetLoggers();

// auto init
void _AutoInitialize();

// add logger
inline void AddLogger(const std::string &name)                          { GetLoggers().push_back(spdlog::stdout_color_mt(name)); }
inline void AddLogger(const std::string &name, const std::string &path) { GetLoggers().push_back(spdlog::rotating_logger_mt(name, path, 1048576 * 5, 3)); }
inline void _Check()                                                    { auto &logs = GetLoggers(); if (logs.size() == 0) { _AutoInitialize(); } }

/* easy access */
template<typename ...Args> void info(const char *fmt, const Args &... args)                 { _Check(); for (auto logger : GetLoggers()) logger->info(fmt, args...); }
template<typename ...Args> void warn(const char *fmt, const Args &... args)                 { _Check(); for (auto logger : GetLoggers()) logger->warn(fmt, args...); }
template<typename ...Args> void error(const char *fmt, const Args &... args)                { _Check(); for (auto logger : GetLoggers()) logger->error(fmt, args...); }
template<typename ...Args> void fatal(const char *fmt, const Args &... args)                { _Check(); for (auto logger : GetLoggers()) logger->critical(fmt, args...); exit(1); }
template<typename... Args> void assertion(bool flag)                                        { if (!flag) { _Check(); for (auto logger : GetLoggers()) { logger->critical("assertion failed"); } exit(1); } }
template<typename ...Args> void assertion(bool flag, const char *fmt, const Args &... args) { if (!flag) { _Check(); for (auto logger : GetLoggers()) { logger->critical(fmt, args...); } exit(1); } }
#ifndef NDEBUG
template<typename... Args> void debug(const char *fmt, const Args &... args)                { _Check(); for (auto logger : GetLoggers()) logger->debug(fmt, args...); }
#else
template<typename... Args> void debug(const char *fmt, const Args &... args) {}
#endif
