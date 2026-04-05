#include "draw.hpp"
#include "geometry.hpp"
#include "simulation.hpp"
#include <vector>
#include <iostream>
#include <cmath>

using namespace std;

struct Triangle {
    int a, b, c;
    bool isBad = false;
    Triangle(int a, int b, int c) : a(a), b(b), c(c) {}
};

struct Edge {
    int a, b;
};

bool inCircumcircle(point pt, point v1, point v2, point v3) {
    ftype ax = v1.x - pt.x;
    ftype ay = v1.y - pt.y;
    ftype bx = v2.x - pt.x;
    ftype by = v2.y - pt.y;
    ftype cx = v3.x - pt.x;
    ftype cy = v3.y - pt.y;
    ftype det =
        (ax*ax + ay*ay) * (bx * cy - cx * by) -
        (bx*bx + by*by) * (ax * cy - cx * ay) +
        (cx*cx + cy*cy) * (ax * by - bx * ay);
    ftype orientation = (v2.x - v1.x) * (v3.y - v1.y) - (v2.y - v1.y) * (v3.x - v1.x);
    if (orientation > 0) return det > eps;
    return det < -eps;
}

vector<point> saved_waypoints;
bool initialized = false;
int current_wp_idx = 0;

void my_planner(const envmap& curmap,
                const array<pair<point, point>, playercount>& playerdata,
                const array<point, rays>& raycasts, const agent& curplayer,
                ftype& a, ftype& steer) {
    if (!initialized) {
        vector<point> pts;
        for (size_t i = 0; i < curmap.size(); i++) {
            for (auto& p : curmap[i]) {
                pts.push_back(p);
            }
        }
        
        ftype minX = 1e9, minY = 1e9, maxX = -1e9, maxY = -1e9;
        for (auto& p : pts) {
            minX = min(minX, p.x); minY = min(minY, p.y);
            maxX = max(maxX, p.x); maxY = max(maxY, p.y);
        }
        ftype dx = maxX - minX;
        ftype dy = maxY - minY;
        ftype deltaMax = max(dx, dy);
        ftype midX = (minX + maxX) / 2;
        ftype midY = (minY + maxY) / 2;

        pts.push_back(point(midX - 20 * deltaMax, midY - deltaMax));
        pts.push_back(point(midX, midY + 20 * deltaMax));
        pts.push_back(point(midX + 20 * deltaMax, midY - deltaMax));

        vector<Triangle> triangles;
        triangles.push_back(Triangle((int)pts.size()-3, (int)pts.size()-2, (int)pts.size()-1));

        for (int i = 0; i < pts.size() - 3; i++) {
            for (auto& t : triangles) {
                if (inCircumcircle(pts[i], pts[t.a], pts[t.b], pts[t.c])) t.isBad = true;
            }
            vector<Edge> polygon;
            for (size_t t1 = 0; t1 < triangles.size(); ++t1) {
                if (!triangles[t1].isBad) continue;
                Edge edges[3] = {{triangles[t1].a, triangles[t1].b}, {triangles[t1].b, triangles[t1].c}, {triangles[t1].c, triangles[t1].a}};
                for (int j = 0; j < 3; ++j) {
                    bool shared = false;
                    for (size_t t2 = 0; t2 < triangles.size(); ++t2) {
                        if (t1 != t2 && triangles[t2].isBad) {
                            Edge o[3] = {{triangles[t2].a, triangles[t2].b}, {triangles[t2].b, triangles[t2].c}, {triangles[t2].c, triangles[t2].a}};
                            for(int k=0; k<3; ++k) {
                                if ((edges[j].a == o[k].a && edges[j].b == o[k].b) || (edges[j].a == o[k].b && edges[j].b == o[k].a)) {
                                    shared = true; break;
                                }
                            }
                        }
                        if (shared) break;
                    }
                    if (!shared) polygon.push_back(edges[j]);
                }
            }
            vector<Triangle> nextTriangles;
            for (auto& t : triangles) if (!t.isBad) nextTriangles.push_back(t);
            for (auto& e : polygon) nextTriangles.push_back(Triangle(e.a, e.b, i));
            triangles = nextTriangles;
        }

        for (auto& t : triangles) {
            if (t.a >= pts.size() - 3 || t.b >= pts.size() - 3 || t.c >= pts.size() - 3) continue;
            point c((pts[t.a].x + pts[t.b].x + pts[t.c].x)/3.0f, (pts[t.a].y + pts[t.b].y + pts[t.c].y)/3.0f);
            
            bool valid = true;
            if (curmap.size() > 0) {
                for (size_t i = 0; i < curmap.size() - 1; i++) {
                    if (contains(curmap[i], c)) { valid = false; break; }
                }
                if (!contains(curmap.back(), c)) valid = false;
            }
            if (valid) saved_waypoints.push_back(c);
        }
        cout << "Generated " << saved_waypoints.size() << " waypoints.\n";
        initialized = true;
    }

    if (saved_waypoints.empty()) {
        a = 0; steer = 0; return;
    }

    point pos = playerdata[0].first;
    ftype theta = playerdata[0].second.y;

    if (current_wp_idx >= saved_waypoints.size()) current_wp_idx = 0;
    
    point target = saved_waypoints[current_wp_idx];
    if (dist(pos, target) < 0.2f) {
        current_wp_idx++;
        if (current_wp_idx >= saved_waypoints.size()) current_wp_idx = 0;
        target = saved_waypoints[current_wp_idx];
    }

    ftype desired_heading = arg(target - pos);
    ftype heading_error = desired_heading - theta;
    while (heading_error > PI) heading_error -= 2 * PI;
    while (heading_error < -PI) heading_error += 2 * PI;

    steer = heading_error * 0.5f; 
    a = 1.0f;
}

int main() {
    agent myagent;
    myagent.calculate_1 = my_planner;

    array<agent, playercount> myagents;
    for (int i = 0; i < playercount; i++) myagents[i] = myagent;

    ftype simultime = 30;

    simulationinstance s(myagents, simultime);
    s.visualmode = false;  // Headless test
    s.humanmode = false;
    
    s.run();

    return 0;
}
