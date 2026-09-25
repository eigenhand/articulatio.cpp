#pragma once

#include <string>

namespace breeze {

struct ServerOptions {
    std::string model;
    std::string host = "127.0.0.1";
    int port = 8080;
    bool use_gpu = true;
    bool webui = false;
    bool verbose = false;
    std::string voices_dir = "voices";
    int ws_port = 0; // 0 puts it on port + 1, negative turns it off
    int chunk_first = 4;
    int chunk_max = 25;
    int split_chars = 600; // 0 keeps long text in a single pass
    // post-processing defaults for this server. any request may override them, whatever a request
    // leaves out falls back to these
    float speed = 1.0f;
    int trim_head_ms = 0;
    int trim_tail_ms = 0;
    int pause_ms = 0;
};

int run_server(const ServerOptions & opts);

}
