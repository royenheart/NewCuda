#include "easylogging++.h"

INITIALIZE_EASYLOGGINGPP

int main(int argc, const char** argv) {
    // Load configuration from file
    el::Configurations conf("./log.conf");
    // Reconfigure single logger
    el::Loggers::reconfigureLogger("default", conf);
    // Actually reconfigure all loggers instead
    el::Loggers::reconfigureAllLoggers(conf);
    // Now all the loggers will use configuration from file
    LOG(INFO) << "NIHAO";
    return 0;
}