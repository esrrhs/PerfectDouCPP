// Unit tests for the game engine, action space, oracle and features.
#include <cassert>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <set>
#include <vector>

#include "ddz/action_space.h"
#include "ddz/features.h"
#include "ddz/game.h"
#include "ddz/moves.h"
#include "ddz/oracle.h"
#include "algo/ppo.h"

using namespace ddz;

#define CHECK(cond)                                                       \
    do {                                                                  \
        if (!(cond)) {                                                    \
            std::printf("CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, \
                        #cond);                                           \
            std::abort();                                                 \
        }                                                                 \
    } while (0)

static void testDetect() {
    auto t = [](const std::string& s, int type) {
        MoveInfo i = detectMove(CardSet::parse(s));
        CHECK(i.type == type);
    };
    t("3", MT_SINGLE);
    t("R", MT_SINGLE);
    t("33", MT_PAIR);
    t("22", MT_PAIR);
    t("555", MT_TRIPLE);
    t("TTTT", MT_BOMB);
    t("BR", MT_ROCKET);
    t("3334", MT_TRIPLE_ONE);
    t("3444", MT_TRIPLE_ONE);
    t("33344", MT_TRIPLE_TWO);
    t("34567", MT_STRAIGHT);
    t("3456789TJQKA", MT_STRAIGHT);  // longest straight, ends at A
    CHECK(detectMove(CardSet::parse("3456789TJQKA2")).type == MT_WRONG);
    t("334455", MT_PAIR_STRAIGHT);
    t("33445566778899TTJJQQ", MT_PAIR_STRAIGHT);  // len 10
    t("333444", MT_PLANE);
    t("33344456", MT_PLANE_SOLO);    // plane of 2 trios + 2 solo kickers
    t("33344455", MT_PLANE_SOLO);    // two equal kickers count as two solos
    t("3334445566", MT_PLANE_PAIR);  // plane of 2 trios + 2 pair kickers
    t("333345", MT_FOUR_TWO);
    t("333344", MT_FOUR_TWO);  // two equal kicker cards accepted
    t("33334455", MT_FOUR_TWO_PAIR);
    // edge case from DouZero detector: 12 cards with four consecutive trios
    MoveInfo e = detectMove(CardSet::parse("333444555666"));
    CHECK(e.type == MT_PLANE || e.type == MT_PLANE_SOLO);
    CHECK(detectMove(CardSet::parse("3344")).type == MT_WRONG);
    CHECK(detectMove(CardSet::parse("")).type == MT_PASS);
    // planes never include rank 2
    CHECK(detectMove(CardSet::parse("TTTJJJQQQKKKAAA222")).type == MT_WRONG);
}

static void testActionSpace() {
    const auto& tab = abstractTable();
    CHECK(tab.size() == 621);
    CHECK(tab[0].name == "3");
    CHECK(tab[14].name == "R");
    CHECK(tab[41].name == "3334");
    CHECK(tab[222].name == "222R");
    CHECK(tab[379].name == "34567");
    CHECK(tab[606].name == "3333");
    CHECK(tab[619].name == "BR");
    CHECK(tab[620].name == "pass");

    auto roundTrip = [&](const std::string& s) {
        CardSet m = CardSet::parse(s);
        MoveInfo mi = detectMove(m);
        int id = concreteToAbstract(m, mi);
        CHECK(id >= 0 && id < 621);
        CHECK(tab[id].main == m || tab[id].hasKicker);
    };
    roundTrip("3");
    roundTrip("3334");
    roundTrip("222R");
    roundTrip("34567");
    roundTrip("3334445566");
    roundTrip("33334455");
    roundTrip("333345");
    roundTrip("3333");
    roundTrip("BR");

    // The official specific_map maps ambiguous long planes to every matching
    // abstract template, not only to the detector's preferred interpretation.
    CardSet ambiguous = CardSet::parse("333444555666");
    int matches = 0;
    for (const AbstractAction& a : tab)
        if (abstractMatches(a, ambiguous)) ++matches;
    CHECK(matches > 1);
}

static void testBottomCards() {
    Rng rng(17);
    Game game;
    game.deal(rng);
    int rank = -1;
    for (int r = 0; r < kRanks; ++r)
        if (game.bottom.c[r] > 0) {
            rank = r;
            break;
        }
    CHECK(rank >= 0);
    int before = game.bottom.total();
    int rankBefore = game.bottom.c[rank];
    CardSet move;
    move.add(rank);
    game.step(move);
    CHECK(game.bottom.total() == before - 1);
    CHECK(game.bottom.c[rank] == rankBefore - 1);
}

static void testGAE() {
    std::vector<algo::Transition> tr(10);
    for (int i = 0; i < 10; ++i) {
        tr[i].gameId = 3;
        tr[i].reward = 1.0f;
        tr[i].value = 0.0f;
    }
    tr.back().terminal = true;
    algo::computeGAE(tr, 1.0f, 1.0f, 3);
    const float expected[] = {3, 2, 1, 3, 2, 1, 3, 2, 1, 1};
    for (int i = 0; i < 10; ++i) {
        CHECK(std::abs(tr[i].adv - expected[i]) < 1e-6f);
        CHECK(std::abs(tr[i].ret - expected[i]) < 1e-6f);
    }
}

static void testGenerator() {
    // full free-lead generation for a rich hand
    CardSet hand = CardSet::parse("333444556677BR");
    std::vector<CardSet> ms = genAllMoves(hand);
    std::set<uint64_t> keys;
    for (auto& m : ms) {
        CHECK(hand.contains(m));
        CHECK(detectMove(m).type != MT_WRONG);
        keys.insert(m.key());
    }
    CHECK(keys.size() == ms.size());  // no duplicates
    // rocket and bombs present
    CardSet rocket = CardSet::parse("BR");
    bool hasRocket = false;
    for (auto& m : ms)
        if (m == rocket) hasRocket = true;
    CHECK(hasRocket);

    // only higher singles / bombs beat a 3, plus pass
    std::vector<CardSet> beat = legalMoves(hand, CardSet::parse("3"));
    bool hasPass = false, hasSolo4 = false, hasSolo3 = false;
    for (auto& m : beat) {
        if (m.empty()) { hasPass = true; continue; }
        MoveInfo i = detectMove(m);
        bool ok = i.type == MT_SINGLE || i.type == MT_BOMB ||
                  i.type == MT_ROCKET;
        CHECK(ok);
        if (m == CardSet::parse("4")) hasSolo4 = true;
        if (m == CardSet::parse("3")) hasSolo3 = true;
    }
    CHECK(hasPass && hasSolo4 && !hasSolo3);

    // rocket cannot be beaten (pass only)
    std::vector<CardSet> vsRocket = legalMoves(hand, rocket);
    CHECK(vsRocket.size() == 1);
    CHECK(vsRocket[0].empty());

    // free lead has no pass
    std::vector<CardSet> lead = legalMoves(hand, CardSet{});
    for (auto& m : lead) CHECK(!m.empty());

    // planes must have the same length to beat
    CardSet h2 = CardSet::parse("44455566678");
    std::vector<CardSet> bp = legalMoves(h2, CardSet::parse("333444"));
    bool hasLen2 = false, hasLen3 = false;
    for (auto& m : bp) {
        MoveInfo i = detectMove(m);
        if (m.empty()) continue;
        if (i.type == MT_BOMB || i.type == MT_ROCKET) continue;
        CHECK((i.type == MT_PLANE || i.type == MT_PLANE_SOLO ||
               i.type == MT_PLANE_PAIR) &&
              i.len == 2);
        if (i.type == MT_PLANE && m == CardSet::parse("444555")) hasLen2 = true;
        if (i.len == 3) hasLen3 = true;
    }
    CHECK(hasLen2 && !hasLen3);
}

static void testOracle() {
    auto steps = [](const std::string& s) {
        return minSteps(CardSet::parse(s));
    };
    CHECK(steps("34567") == 1);                 // straight
    CHECK(steps("334455") == 1);                // pair straight
    CHECK(steps("333444555") == 1);             // plane
    CHECK(steps("33344456") == 1);              // plane + two solos
    CHECK(steps("3334445566") == 1);            // plane + two pairs
    CHECK(steps("33334455") == 1);              // four + two pairs
    CHECK(steps("333345") == 1);                // four + two solos
    CHECK(steps("BR") == 1);                    // rocket
    CHECK(steps("333555") == 2);                // non-consecutive trios
    CHECK(steps("34") == 2);                    // two loose singles
    CHECK(steps("333444555666") <= 2);          // bare plane chains
    CHECK(steps("333444666777") == 2);          // two separated len-2 planes
    CHECK(steps("333555777") == 2);             // 777+55 then 333+5
    // random sanity: never more than card count, at least 1
    std::mt19937 rng(7);
    for (int t = 0; t < 200; ++t) {
        int n = 4 + rng() % 17;
        std::vector<int> deck;
        for (int r = 0; r < kNormalRanks; ++r)
            for (int k = 0; k < 4; ++k) deck.push_back(r);
        deck.push_back(13);
        deck.push_back(14);
        std::shuffle(deck.begin(), deck.end(), rng);
        CardSet h;
        for (int i = 0; i < n; ++i) h.add(deck[i]);
        int s = minSteps(h);
        CHECK(s >= 1 && s <= h.total());
    }
}

static void testRandomGames() {
    Rng rng(123);
    long long totalMoves = 0;
    int games = 300;
    int landlordWins = 0;
    for (int g = 0; g < games; ++g) {
        Game game;
        game.deal(rng);
        CHECK(game.hand[0].total() == 20);
        CHECK(game.hand[1].total() == 17);
        CHECK(game.hand[2].total() == 17);
        int guard = 0;
        while (!game.over) {
            std::vector<CardSet> legal = game.legal();
            CHECK(!legal.empty());
            CardSet m = legal[rng.below(int(legal.size()))];
            CHECK(game.hand[game.turn].contains(m));
            // every non-pass must beat the current rival play
            if (!m.empty() && !game.toBeat().empty()) {
                MoveInfo a = detectMove(m), b = detectMove(game.toBeat());
                bool bombBeats = (a.type == MT_BOMB && b.type != MT_BOMB &&
                                  b.type != MT_ROCKET) ||
                                 a.type == MT_ROCKET;
                bool same = (a.type == b.type && a.len == b.len &&
                             a.rank > b.rank);
                bool planeLike = (a.type == b.type) &&
                                 (a.type == MT_PLANE_SOLO ||
                                  a.type == MT_PLANE_PAIR) &&
                                 a.len == b.len;
                CHECK(bombBeats || same || planeLike);
            }
            game.step(m);
            ++totalMoves;
            ++guard;
            CHECK(guard < 10000);
        }
        if (game.winner == 0) ++landlordWins;
        // winner is whoever emptied their hand
        if (game.winner == 0)
            CHECK(game.hand[0].total() == 0);
        else
            CHECK(game.hand[1].total() == 0 || game.hand[2].total() == 0);
        double p0 = game.payoff(0), p1 = game.payoff(1);
        CHECK(std::abs(p0) == 2.0 * std::pow(2.0, game.bombCount));
        CHECK(std::abs(p1) == std::pow(2.0, game.bombCount));
    }
    std::printf("  random games: %d, landlord wins %d, avg moves %.1f\n", games,
                landlordWins, double(totalMoves) / games);
}

static void testFeatures() {
    Rng rng(9);
    for (int g = 0; g < 20; ++g) {
        Game game;
        game.deal(rng);
        int guard = 0;
        while (!game.over) {
            EncodedState e = encodeState(game);
            std::vector<LegalOption> opts = legalOptions(game);
            CHECK(!opts.empty());
            for (const LegalOption& o : opts) {
                CHECK(o.abstractId >= 0 && o.abstractId < 621);
                CHECK(game.hand[game.turn].contains(o.concrete));
                if (!o.concrete.empty())
                    CHECK(detectMove(o.concrete).type != MT_WRONG);
                std::array<uint8_t, kCardMat> mm;
                actionCardMatrix(o.abstractId, mm);
            }
            std::vector<CardSet> legal = game.legal();
            // every valid concrete legal move maps to one of the abstract options
            for (const CardSet& m : legal) {
                MoveInfo mi = detectMove(m);
                if (mi.type == MT_WRONG) continue;  // filtered degenerate combos
                int id = concreteToAbstract(m, mi);
                CHECK(id >= 0);
                bool found = false;
                for (const LegalOption& o : opts)
                    if (o.abstractId == id) found = true;
                CHECK(found);
            }
            // play decoded concrete of first option
            CardSet chosen = opts.front().concrete;
            bool inLegal = false;
            for (const CardSet& m : legal)
                if (m == chosen) inLegal = true;
            CHECK(inLegal);
            game.step(chosen);
            if (++guard > 10000) break;
        }
    }
}

int main() {
    std::printf("testDetect...\n"); testDetect();
    std::printf("testActionSpace...\n"); testActionSpace();
    std::printf("testGenerator...\n"); testGenerator();
    std::printf("testOracle...\n"); testOracle();
    std::printf("testBottomCards...\n"); testBottomCards();
    std::printf("testGAE...\n"); testGAE();
    std::printf("testRandomGames...\n"); testRandomGames();
    std::printf("testFeatures...\n"); testFeatures();
    std::printf("ALL TESTS PASSED\n");
    return 0;
}
