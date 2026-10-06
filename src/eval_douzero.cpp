// Play saved PerfectDou actors against a DouZero server.
// The training process is not involved. Inference stays on the CPU.
//
// perfectdou_eval --models DIR --decks N --port 18765 --label u1100
//                 --csv build-win/curve/curve.csv
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>

#include "algo/rollout.h"
#include "ddz/game.h"
#include "ddz/oracle.h"
#include "nn/gemm.h"
#include "nn/net.h"

namespace {

std::string argValue(int argc, char** argv, const char* key, const char* def) {
    for (int i = 1; i < argc - 1; ++i)
        if (std::strcmp(argv[i], key) == 0) return argv[i + 1];
    return def;
}

bool sendAll(SOCKET s, const std::string& msg) {
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

bool recvLine(SOCKET s, std::string& line) {
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

bool askDouZero(SOCKET sock, const ddz::Game& g, const std::vector<ddz::CardSet>& legal,
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

ddz::CardSet ourMove(nn::Actor& actor, nn::ActorInfer& w, const ddz::Game& g,
                     const std::vector<ddz::LegalOption>& options) {
    algo::Transition t;
    ddz::EncodedState e = ddz::encodeState(g);
    t.imp = e.imp;
    t.scalar = e.scalar;
    t.extra = e.extra;
    t.extraScalar = e.extraScalar;
    for (const ddz::LegalOption& o : options) {
        t.mask[o.abstractId >> 6] |= uint64_t(1) << (o.abstractId & 63);
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

void playOne(ddz::Game g, bool ourLandlord, std::array<nn::Actor, 3>& actor,
             std::array<nn::ActorInfer, 3>& ws, SOCKET sock, Tally& tally) {
    int guard = 0;
    while (!g.over && guard++ < 400) {
        auto options = ddz::legalOptions(g);
        if (options.empty()) {
            std::fprintf(stderr, "no legal move\n");
            return;
        }
        bool ours = ourLandlord ? g.turn == 0 : g.turn != 0;
        ddz::CardSet mv;
        if (ours) {
            mv = ourMove(actor[g.turn], ws[g.turn], g, options);
        } else {
            std::vector<ddz::CardSet> legal;
            legal.reserve(options.size());
            for (const ddz::LegalOption& o : options) legal.push_back(o.concrete);
            if (!askDouZero(sock, g, legal, mv)) return;
        }
        g.step(mv);
    }
    if (!g.over) return;
    double mult = std::pow(2.0, g.bombCount);
    double landlordScore = (g.winner == 0 ? 2.0 : -2.0) * mult;
    double ours = ourLandlord ? landlordScore : -landlordScore;
    bool win = ourLandlord ? g.winner == 0 : g.winner != 0;
    tally.games += 1;
    tally.wins += win ? 1 : 0;
    tally.score += ours;
}

}  // namespace

int main(int argc, char** argv) {
    std::string models = argValue(argc, argv, "--models", "");
    int decks = std::atoi(argValue(argc, argv, "--decks", "200").c_str());
    int port = std::atoi(argValue(argc, argv, "--port", "18765").c_str());
    std::string label = argValue(argc, argv, "--label", "snap");
    std::string csv = argValue(argc, argv, "--csv", "build-win/curve/curve.csv");
    uint64_t seed = std::strtoull(argValue(argc, argv, "--seed", "1").c_str(), nullptr, 10);
    if (models.empty() || decks <= 0) {
        std::fprintf(stderr, "usage: perfectdou_eval --models DIR --decks N --port P\n");
        return 2;
    }

    nn::gemmInit();
    nn::gemmSetGpu(false);
    nn::gemmSetThreadGpu(0);

    std::array<nn::Actor, 3> actor;
    for (int s = 0; s < 3; ++s) {
        actor[s].load((models + "/actor" + std::to_string(s) + ".bin").c_str());
        actor[s].prepareInference();
    }
    std::array<nn::ActorInfer, 3> ws;

    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) return 1;
    SOCKET sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<u_short>(port));
    inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
    if (connect(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        std::fprintf(stderr, "cannot connect to DouZero server on port %d\n", port);
        return 1;
    }

    ddz::Rng rng(seed);
    Tally asLandlord, asPeasant;
    for (int i = 0; i < decks; ++i) {
        ddz::Game g;
        g.deal(rng);
        auto hands = g.hand;
        auto bottom = g.bottom;
        playOne(g, true, actor, ws, sock, asLandlord);
        ddz::Game g2;
        g2.hand = hands;
        g2.bottom = bottom;
        g2.turn = 0;
        g2.lastPlayer = 0;
        playOne(g2, false, actor, ws, sock, asPeasant);
        if ((i + 1) % 20 == 0) {
            int games = asLandlord.games + asPeasant.games;
            std::printf("deck %d/%d  WP %.3f ADP %.3f\n", i + 1, decks,
                        games ? double(asLandlord.wins + asPeasant.wins) / games : 0.0,
                        games ? (asLandlord.score + asPeasant.score) / games : 0.0);
            std::fflush(stdout);
        }
    }
    sendAll(sock, "QUIT\n");
    closesocket(sock);
    WSACleanup();

    int games = asLandlord.games + asPeasant.games;
    double wp = games ? double(asLandlord.wins + asPeasant.wins) / games : 0;
    double adp = games ? (asLandlord.score + asPeasant.score) / games : 0;
    double wpL = asLandlord.games ? double(asLandlord.wins) / asLandlord.games : 0;
    double adpL = asLandlord.games ? asLandlord.score / asLandlord.games : 0;
    double wpP = asPeasant.games ? double(asPeasant.wins) / asPeasant.games : 0;
    double adpP = asPeasant.games ? asPeasant.score / asPeasant.games : 0;
    std::printf("done games %d WP %.3f ADP %.3f | landlord WP %.3f ADP %.3f | "
                "peasant WP %.3f ADP %.3f\n",
                games, wp, adp, wpL, adpL, wpP, adpP);

    bool needHeader = true;
    if (FILE* probe = std::fopen(csv.c_str(), "rb")) {
        int ch = std::fgetc(probe);
        needHeader = ch == EOF;
        std::fclose(probe);
    }
    FILE* f = std::fopen(csv.c_str(), "ab");
    if (!f) {
        std::fprintf(stderr, "cannot append %s\n", csv.c_str());
        return 1;
    }
    if (needHeader)
        std::fprintf(f, "label,decks,games,wp,adp,wp_landlord,adp_landlord,wp_peasant,adp_peasant\n");
    std::fprintf(f, "%s,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                 label.c_str(), decks, games, wp, adp, wpL, adpL, wpP, adpP);
    std::fclose(f);
    return 0;
}
