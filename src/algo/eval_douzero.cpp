#include "algo/eval_douzero.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
typedef SOCKET socket_t;
#define INVALID_SOCKET_VAL INVALID_SOCKET
#else
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>
typedef int socket_t;
#define INVALID_SOCKET_VAL (-1)
#endif

#include "algo/rollout.h"
#include "ddz/game.h"
#include "ddz/oracle.h"
#include "nn/gemm.h"
#include "nn/net.h"

namespace algo {

namespace {

struct SocketSubsystem {
    bool ok = false;
    SocketSubsystem() {
#if defined(_WIN32)
        WSADATA wsa;
        ok = (WSAStartup(MAKEWORD(2, 2), &wsa) == 0);
#else
        ok = true;
#endif
    }
    ~SocketSubsystem() {
#if defined(_WIN32)
        if (ok) WSACleanup();
#endif
    }
};

void closeSocket(socket_t s) {
#if defined(_WIN32)
    if (s != INVALID_SOCKET_VAL) closesocket(s);
#else
    if (s >= 0) close(s);
#endif
}

bool sendAll(socket_t s, const std::string& msg) {
    const char* p = msg.data();
    int left = static_cast<int>(msg.size());
    while (left > 0) {
        int n = send(s, p, left, 0);
        if (n <= 0) return false;
        p += n;
        left -= n;
    }
    return true;
}

bool recvLine(socket_t s, std::string& line) {
    line.clear();
    char c;
    while (true) {
        int n = recv(s, &c, 1, 0);
        if (n <= 0) return false;
        if (c == '\n') return true;
        if (c != '\r') line.push_back(c);
    }
}

std::string cardsTok(const ddz::CardSet& cs) {
    std::string s;
    for (int r = 0; r < ddz::kRanks; ++r) {
        for (int k = 0; k < cs.c[r]; ++k) {
            if (!s.empty()) s.push_back(',');
            s += std::to_string(ddz::rankValue(r));
        }
    }
    return s.empty() ? "-" : s;
}

ddz::CardSet parseCards(const std::string& tok) {
    ddz::CardSet cs;
    if (tok.empty() || tok == "-") return cs;
    std::stringstream ss(tok);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) continue;
        cs.add(ddz::valueRank(std::stoi(item)));
    }
    return cs;
}

const char* posName(int seat) {
    return seat == 0 ? "landlord" : seat == 1 ? "landlord_down" : "landlord_up";
}

ddz::CardSet douzeroLast(const ddz::Game& g) {
    if (g.seq.empty()) return {};
    if (!g.seq.back().empty()) return g.seq.back();
    if (g.seq.size() >= 2) return g.seq[g.seq.size() - 2];
    return {};
}

std::string joinActions(const std::vector<ddz::CardSet>& ms) {
    std::string s;
    for (size_t i = 0; i < ms.size(); ++i) {
        if (i) s.push_back('/');
        s += cardsTok(ms[i]);
    }
    return s;
}

bool askDouZero(socket_t sock, const ddz::Game& g, const std::vector<ddz::CardSet>& legal,
                ddz::CardSet& chosen) {
    ddz::CardSet other;
    for (int s = 0; s < 3; ++s)
        if (s != g.turn)
            for (int r = 0; r < ddz::kRanks; ++r) other.add(r, g.hand[s].c[r]);
    std::ostringstream o;
    o << posName(g.turn) << '\t'
      << cardsTok(g.hand[g.turn]) << '\t'
      << cardsTok(other) << '\t'
      << cardsTok(douzeroLast(g)) << '\t'
      << g.bombCount << '\t'
      << g.hand[0].total() << ' ' << g.hand[1].total() << ' ' << g.hand[2].total()
      << '\t'
      << cardsTok(g.played[0]) << ';' << cardsTok(g.played[1]) << ';'
      << cardsTok(g.played[2]) << '\t'
      << cardsTok(g.lastMove[0]) << ';' << cardsTok(g.lastMove[1]) << ';'
      << cardsTok(g.lastMove[2]) << '\t'
      << joinActions(g.seq) << '\t'
      << joinActions(legal);
    if (!sendAll(sock, o.str() + "\n")) return false;
    std::string reply;
    if (!recvLine(sock, reply)) return false;
    ddz::CardSet want = parseCards(reply);
    for (const ddz::CardSet& m : legal)
        if (m == want) {
            chosen = m;
            return true;
        }
    std::fprintf(stderr, "DouZero action not legal: %s\n", reply.c_str());
    chosen = legal.front();
    return true;
}

ddz::CardSet ourMove(const nn::Actor& actor, nn::ActorInfer& w, const ddz::Game& g,
                     const std::vector<ddz::LegalOption>& options) {
    algo::Transition t;
    ddz::EncodedState e = ddz::encodeState(g);
    t.imp = e.imp;
    t.scalar = e.scalar;
    t.extra = e.extra;
    t.extraScalar = e.extraScalar;
    for (const ddz::LegalOption& o : options) {
        t.actions.emplace_back(o.abstractId, o.feature);
    }
    std::vector<algo::Transition*> ptrs{&t};
    nn::Mat xImp, seq, actionFeat, actionSample, actionId, actionOffset, extra;
    algo::buildBatch(ptrs, xImp, seq, actionFeat, actionSample, actionId,
                     actionOffset, extra);
    const nn::Mat& logits =
        nn::actorInferForward(actor, w, xImp, seq, actionFeat, actionSample,
                              actionId, actionOffset);
    int best = options.front().abstractId;
    float bestL = logits.row(0)[best];
    for (const ddz::LegalOption& o : options) {
        float v = logits.row(0)[o.abstractId];
        if (v > bestL) {
            bestL = v;
            best = o.abstractId;
        }
    }
    for (const ddz::LegalOption& o : options)
        if (o.abstractId == best) return o.concrete;
    return options.front().concrete;
}

struct Tally {
    int games = 0;
    int wins = 0;
    double score = 0;
};

bool playOne(ddz::Game g, bool ourLandlord, const std::array<nn::Actor, 3>& actors,
             std::array<nn::ActorInfer, 3>& ws, socket_t sock, Tally& tally) {
    int guard = 0;
    while (!g.over && guard++ < 400) {
        auto options = ddz::legalOptions(g);
        if (options.empty()) {
            return false;
        }
        bool ours = ourLandlord ? g.turn == 0 : g.turn != 0;
        ddz::CardSet mv;
        if (ours) {
            mv = ourMove(actors[g.turn], ws[g.turn], g, options);
        } else {
            std::vector<ddz::CardSet> legal;
            legal.reserve(options.size());
            for (const ddz::LegalOption& o : options) legal.push_back(o.concrete);
            if (!askDouZero(sock, g, legal, mv)) return false;
        }
        g.step(mv);
    }
    if (!g.over) return false;
    double mult = std::pow(2.0, g.bombCount);
    double landlordScore = (g.winner == 0 ? 2.0 : -2.0) * mult;
    double ours = ourLandlord ? landlordScore : -landlordScore;
    bool win = ourLandlord ? g.winner == 0 : g.winner != 0;
    tally.games += 1;
    tally.wins += win ? 1 : 0;
    tally.score += ours;
    return true;
}

}  // namespace

std::shared_ptr<ActorGroup> cloneActors(const std::array<nn::Actor, 3>& src) {
    auto dst = std::make_shared<ActorGroup>();
    for (int s = 0; s < 3; ++s) {
        (*dst)[s].init(src[s].cfg, 1);
        auto da = (*dst)[s].params();
        auto sa = const_cast<nn::Actor&>(src[s]).params();
        for (size_t i = 0; i < sa.size(); ++i) {
            da[i]->w = sa[i]->w;
        }
        (*dst)[s].prepareInference();
    }
    return dst;
}

DouZeroEvalResult evaluateAgainstDouZero(std::shared_ptr<const ActorGroup> actors,
                                        const DouZeroEvalConfig& cfg) {
    if (!actors) {
        DouZeroEvalResult r;
        r.error = "null actors";
        return r;
    }
    return evaluateAgainstDouZero(*actors, cfg);
}

DouZeroEvalResult evaluateAgainstDouZero(const std::array<nn::Actor, 3>& actors,
                                        const DouZeroEvalConfig& cfg) {
    DouZeroEvalResult res;
    // Evaluation runs purely on CPU; ensure this thread does not attempt GPU GEMM.
    nn::gemmSetThreadGpu(0);

    // Save cloned model weights and metadata to disk for traceability
    if (!cfg.saveDir.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(cfg.saveDir, ec);
        for (int s = 0; s < 3; ++s) {
            std::string path = cfg.saveDir + "/actor" + std::to_string(s) + ".bin";
            actors[s].save(path.c_str());
        }
        std::string metaPath = cfg.saveDir + "/meta.txt";
        FILE* mf = std::fopen(metaPath.c_str(), "w");
        if (mf) {
            std::fprintf(mf, "update %d\nminutes %.2f\nlabel %s\n",
                         cfg.update, cfg.elapsedMinutes,
                         (cfg.label.empty() ? ("u" + std::to_string(cfg.update)) : cfg.label).c_str());
            std::fclose(mf);
        }

        // Protect disk space: prune oldest snapshots if maxSnapshots limit is set
        if (cfg.maxSnapshots > 0) {
            try {
                std::filesystem::path parent = std::filesystem::path(cfg.saveDir).parent_path();
                if (std::filesystem::exists(parent) && std::filesystem::is_directory(parent)) {
                    std::vector<std::pair<std::filesystem::file_time_type, std::filesystem::path>> dirs;
                    for (const auto& entry : std::filesystem::directory_iterator(parent)) {
                        if (entry.is_directory()) {
                            dirs.emplace_back(entry.last_write_time(), entry.path());
                        }
                    }
                    if (static_cast<int>(dirs.size()) > cfg.maxSnapshots) {
                        std::sort(dirs.begin(), dirs.end(), [](const auto& a, const auto& b) {
                            return a.first < b.first;
                        });
                        int toRemove = static_cast<int>(dirs.size()) - cfg.maxSnapshots;
                        for (int i = 0; i < toRemove; ++i) {
                            std::filesystem::remove_all(dirs[i].second, ec);
                        }
                    }
                }
            } catch (...) {
            }
        }
    }

    SocketSubsystem sockSub;
    if (!sockSub.ok) {
        res.error = "network subsystem init failed";
        return res;
    }

    socket_t sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (sock == INVALID_SOCKET_VAL) {
        res.error = "failed to create socket";
        return res;
    }

#if defined(_WIN32)
    DWORD timeoutMs = 30000;
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, (const char*)&timeoutMs, sizeof(timeoutMs));
    setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, (const char*)&timeoutMs, sizeof(timeoutMs));
#else
    struct timeval tv;
    tv.tv_sec = 30;
    tv.tv_usec = 0;
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv, sizeof(tv));
    setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, (const char*)&tv, sizeof(tv));
#endif

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(cfg.port));
    if (inet_pton(AF_INET, cfg.host.c_str(), &addr.sin_addr) <= 0) {
        closeSocket(sock);
        res.error = "invalid DouZero host: " + cfg.host;
        return res;
    }

    if (connect(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        closeSocket(sock);
        res.error = "cannot connect to DouZero server at " + cfg.host + ":" + std::to_string(cfg.port);
        return res;
    }

    std::array<nn::ActorInfer, 3> ws;
    uint64_t seed = cfg.seed;
    if (seed == 0) {
        seed = static_cast<uint64_t>(std::chrono::system_clock::now().time_since_epoch().count());
    }
    ddz::Rng rng(seed);

    Tally asLandlord, asPeasant;
    int decksPlayed = 0;
    for (int i = 0; i < cfg.decks; ++i) {
        ddz::Game g;
        g.deal(rng);
        auto hands = g.hand;
        auto bottom = g.bottom;
        if (!playOne(g, true, actors, ws, sock, asLandlord)) {
            res.error = "communication failed while playing as landlord";
            break;
        }
        ddz::Game g2;
        g2.hand = hands;
        g2.bottom = bottom;
        g2.turn = 0;
        g2.lastPlayer = 0;
        if (!playOne(g2, false, actors, ws, sock, asPeasant)) {
            res.error = "communication failed while playing as peasant";
            break;
        }
        decksPlayed++;
        if (cfg.verbose && (i + 1) % 20 == 0) {
            int curGames = asLandlord.games + asPeasant.games;
            std::printf("deck %d/%d  WP %.3f ADP %.3f\n", i + 1, cfg.decks,
                        curGames ? double(asLandlord.wins + asPeasant.wins) / curGames : 0.0,
                        curGames ? (asLandlord.score + asPeasant.score) / curGames : 0.0);
            std::fflush(stdout);
        }
    }

    sendAll(sock, "QUIT\n");
    closeSocket(sock);

    res.decks = decksPlayed;
    res.games = asLandlord.games + asPeasant.games;
    if (res.games > 0) {
        res.ok = (decksPlayed == cfg.decks);
        res.wp = double(asLandlord.wins + asPeasant.wins) / res.games;
        res.adp = (asLandlord.score + asPeasant.score) / res.games;
        res.wpLandlord = asLandlord.games ? double(asLandlord.wins) / asLandlord.games : 0.0;
        res.adpLandlord = asLandlord.games ? asLandlord.score / asLandlord.games : 0.0;
        res.wpPeasant = asPeasant.games ? double(asPeasant.wins) / asPeasant.games : 0.0;
        res.adpPeasant = asPeasant.games ? asPeasant.score / asPeasant.games : 0.0;
    }

    if (res.ok && !cfg.csvPath.empty()) {
        std::filesystem::path p(cfg.csvPath);
        if (p.has_parent_path()) {
            std::error_code ec;
            std::filesystem::create_directories(p.parent_path(), ec);
        }
        bool needHeader = true;
        if (FILE* probe = std::fopen(cfg.csvPath.c_str(), "rb")) {
            std::fseek(probe, 0, SEEK_END);
            needHeader = (std::ftell(probe) <= 0);
            std::fclose(probe);
        }
        FILE* f = std::fopen(cfg.csvPath.c_str(), "ab");
        if (f) {
            if (needHeader) {
                std::fprintf(f, "update,minutes,timestamp,label,decks,games,wp,adp,wp_landlord,adp_landlord,wp_peasant,adp_peasant\n");
            }
            auto now = std::chrono::system_clock::now();
            std::time_t t = std::chrono::system_clock::to_time_t(now);
            char timeBuf[32];
            std::tm tmNow{};
#if defined(_WIN32)
            localtime_s(&tmNow, &t);
#else
            localtime_r(&t, &tmNow);
#endif
            std::strftime(timeBuf, sizeof(timeBuf), "%Y-%m-%d %H:%M:%S", &tmNow);
            std::string label = cfg.label.empty() ? ("u" + std::to_string(cfg.update)) : cfg.label;
            std::fprintf(f, "%d,%.2f,%s,%s,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                         cfg.update, cfg.elapsedMinutes, timeBuf, label.c_str(),
                         res.decks, res.games, res.wp, res.adp,
                         res.wpLandlord, res.adpLandlord, res.wpPeasant, res.adpPeasant);
            std::fclose(f);
        }
    }

    if (!cfg.saveDir.empty() && res.ok) {
        std::string metaPath = cfg.saveDir + "/meta.txt";
        FILE* mf = std::fopen(metaPath.c_str(), "a");
        if (mf) {
            std::fprintf(mf, "games %d\nwp %.4f\nadp %.4f\nwp_landlord %.4f\nadp_landlord %.4f\nwp_peasant %.4f\nadp_peasant %.4f\n",
                         res.games, res.wp, res.adp, res.wpLandlord, res.adpLandlord, res.wpPeasant, res.adpPeasant);
            std::fclose(mf);
        }
    }

    return res;
}

}  // namespace algo
