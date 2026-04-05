#include "draw.hpp"
#include "geometry.hpp"

constexpr int dim = 2;
constexpr int playercount = 1;
constexpr int rays = 20;

constexpr double dt = 0.01;

using envmap = vector<shape>;
using data = pair<envmap, array<pair<point, point>, playercount>>;

ftype clip(ftype val, ftype lim) {
    if (abs(val) < lim)
        return val;
    else if (val <= -lim)
        return -lim;
    else
        return lim;
}

const ftype acceldelta = 1e-4, steerdelta = 1e-2;

class agent {
    friend class simulationinstance;

    ftype aclip = 0.5, steerclip = PI / 6, vclip = 0.01;
    shape egogeometry = {rotate(point(0.02, 0), point(0, 0), 0),
                         rotate(point(0.02, 0), point(0, 0), 2 * PI / 3),
                         rotate(point(0.02, 0), point(0, 0), 4 * PI / 3)};
    ftype steer = 0, theta = 0, v = 0, acceleration = 0;
    point r = {0, 0};

    void update() {
        v += clip(acceleration, aclip), theta += clip(steer, steerclip),
            v = clip(v, vclip),
            egogeometry =
                rotate(egogeometry, point(0, 0), clip(steer, steerclip));
        r = r + v * point(cos(theta), sin(theta));
    }

    void draw() { drawpolygonboundary(r + egogeometry); }

   public:
    function<void(const envmap&, const array<pair<point, point>, playercount>&,
                  const array<point, rays>&, const agent&, ftype&, ftype&)>
        calculate_1 =
            [](const envmap& curmap,
               const array<pair<point, point>, playercount>& playerdata,
               const array<point, rays>& raycasts, const agent& curplayer,
               ftype& a, ftype& steer) {};

    function<void(const array<point, rays>&, const agent&, ftype&, ftype&)>
        calculate_2 = [](const array<point, rays>& raycasts,
                         const agent& curplayer, ftype& a, ftype& steer) {};
};

class simulationinstance {
    string debuglog;
    vector<vector<bool>> pcollidemat, obscollidemat;

    array<agent, playercount> players;
    array<pair<point, point>, playercount> pdata;

    bool collision = false;
    double endtime, currenttime;

    array<point, rays> raycastagent(const agent& a,
                                    const ftype& noisedeviation = 0) {
        default_random_engine gen;
        normal_distribution<ftype> nd(0, noisedeviation);

        array<point, rays> res;
        ftype theta;

        for (int i = 0; i < rays; i++) {
            theta = ur(rng);
            res[i] = raycast(a.r, mp, 2 * PI * i / rays + a.theta) +
                     nd(gen) * point(cos(2 * PI * theta), sin(2 * PI * theta));
        }

        return res;
    }

    void update() {
        for (int i = 0; i < mp.size() - 1; i++)
            movementspecifier[i](mp[i], currenttime);

        for (int i = 0; i < playercount; i++) {
            if (!humanmode)
                players[i].calculate_1(mp, pdata, raycastagent(players[i],0.02f),
                                       players[i], players[i].acceleration,
                                       players[i].steer);
            pdata[i].first = players[i].r,
            pdata[i].second = point(players[i].v, players[i].theta);
        }

        for (int i = 0; i < playercount; i++) players[i].update();
        currenttime += dt;
    }

    void collidecheck() {
        for (int i = 0; i < playercount; i++)
            for (int j = i + 1; j < playercount; j++)
                if (intersect(players[i].r + players[i].egogeometry,
                              players[j].r + players[j].egogeometry))
                    collision = true, pcollidemat[i][j] = true;

        for (int i = 0; i < playercount; i++)
            for (int j = 0; j < mp.size(); j++)
                if (intersect(players[i].r + players[i].egogeometry, mp[j]))
                    collision = true, obscollidemat[i][j] = true;
    }

    void draw(int rcagent = -1) {
        for (int i = 0; i < mp.size(); i++) drawpolygonboundary(mp[i]);

        if (rcagent >= 0 && rcagent < playercount) {
            for (int i = 0; i < playercount; i++)
                if (i == currentactive) {
                    array<point, rays> currentagentcasts =
                        raycastagent(players[i]);

                    for (int j = 0; j < rays; j++)
                        drawline(players[i].r, currentagentcasts[j]),
                            drawpoint(currentagentcasts[j], GREEN);
                }
        }

        for (int i = 0; i < playercount; i++) players[i].draw();
    }

    void keyboardinput() {
        for (int i = KEY_ONE; i <= KEY_NINE; i++)
            if (IsKeyPressed(i)) currentactive = i - KEY_ONE;
        if (IsKeyDown(KEY_UP)) players[currentactive].acceleration = acceldelta;
        if (IsKeyDown(KEY_DOWN))
            players[currentactive].acceleration = -acceldelta;
        if (IsKeyDown(KEY_LEFT)) players[currentactive].steer = steerdelta;
        if (IsKeyDown(KEY_RIGHT)) players[currentactive].steer = -steerdelta;

        if (IsKeyReleased(KEY_UP)) players[currentactive].acceleration = 0;
        if (IsKeyReleased(KEY_DOWN)) players[currentactive].acceleration = 0;
        if (IsKeyReleased(KEY_LEFT)) players[currentactive].steer = 0;
        if (IsKeyReleased(KEY_RIGHT)) players[currentactive].steer = 0;
    }

    void log() {
        for (int i = 0; i < playercount; i++)
            for (int j = 0; j < playercount; j++)
                if (pcollidemat[i][j])
                    debuglog += "Collision between player " + to_string(i + 1) +
                                " and player " + to_string(j + 1) + '\n';

        for (int i = 0; i < playercount; i++)
            for (int j = 0; j < mp.size() - 1; j++)
                if (obscollidemat[i][j])
                    debuglog += "Collision between player " + to_string(i + 1) +
                                " and obstacle " + to_string(j + 1) + '\n';

        cout << debuglog;
    }

   public:
    envmap mp;
    vector<function<void(shape&, const ftype&)>> movementspecifier;

    bool humanmode, visualmode, blind;
    int currentactive;

    simulationinstance(const array<agent, playercount>& playerlist,
                       const ftype& en) {
        mp = genobs(10, 0.1, 10);

        pcollidemat.assign(playercount, vector<bool>(playercount, false));
        obscollidemat.assign(playercount, vector<bool>(mp.size(), false));

        auto nomovement = [](vector<point>& obstacle, const ftype& curtime) {};
        movementspecifier.assign(mp.size(), nomovement);

        players = playerlist;
        endtime = en, currenttime = 0, collision = false, humanmode = true,
        visualmode = true, blind = false, currentactive = 0;
    }

    void run() {
        if (visualmode) {
            InitWindow(screenWidth, screenHeight, "AGV Simulator");
            SetTargetFPS(60);
            while (!WindowShouldClose()) {
                BeginDrawing();
                ClearBackground(BLACK);

                if (humanmode) {
                    keyboardinput();
                    draw(currentactive);

                    update(), collidecheck();

                } else {
                    draw();
                    update(), collidecheck();
                }

                DrawFPS(10, 10);
                EndDrawing();
            }
            CloseWindow();
        } else
            while (currenttime < endtime) update(), collidecheck();

        log();
    }

    ~simulationinstance() {}
};
