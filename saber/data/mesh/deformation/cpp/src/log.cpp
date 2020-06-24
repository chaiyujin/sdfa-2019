#include "log.h"

static bool gLevelSet = false;
static std::vector<std::shared_ptr<spdlog::logger>> gLoggers;

std::vector<std::shared_ptr<spdlog::logger>> &GetLoggers()
{
    if (!gLevelSet)
    {
        spdlog::set_level(__SNOW_LOG_LEVEL__);
        gLevelSet = true;
    }
    return gLoggers;
}

void _AutoInitialize()
{
    AddLogger("dg");
}
