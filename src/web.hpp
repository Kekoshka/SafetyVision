#pragma once
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "camera.hpp"

// Асинхронный POST JSON (вебхук: мессенджер, СКУД, EHS-система). Поддерживается http://.
void post_json_async(const std::string& url, const std::string& body);

class WebServer {
public:
    WebServer(Services& svc, std::vector<std::unique_ptr<Camera>>& cams, std::function<void()> save_config);
    ~WebServer();
    bool run(const std::string& host, int port, const std::string& web_dir);  // блокирует
    void stop();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
