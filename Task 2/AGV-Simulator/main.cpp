#include "draw.hpp"
#include "geometry.hpp"
#include "simulation.hpp"
#include <vector>
#include <iostream>
#include <algorithm>

using namespace std;

// Helper: Distance from point p to line segment defined by points a and b
ftype dist_to_segment(point p, point a, point b) {
    point ab = b - a;
    point ap = p - a;
    ftype dot_ap_ab = ap.x * ab.x + ap.y * ab.y;
    ftype dot_ab_ab = ab.x * ab.x + ab.y * ab.y;
    ftype t = dot_ap_ab / dot_ab_ab;
    t = max(0.0f, min(1.0f, t)); // Clamp t to the segment [0, 1]
    point closest = a + t * ab;
    return dist(p, closest);
}

struct Triangle {
    int a, b, c;
    bool isBad = false;
    Triangle(int a, int b, int c) : a(a), b(b), c(c) {}
};

struct Edge {
    int a, b;
};

bool inCircumcircle(point pt, point v1, point v2, point v3) {
    ftype ax = v1.x - pt.x, ay = v1.y - pt.y;
    ftype bx = v2.x - pt.x, by = v2.y - pt.y;
    ftype cx = v3.x - pt.x, cy = v3.y - pt.y;
    ftype det = (ax*ax + ay*ay) * (bx * cy - cx * by) -
                (bx*bx + by*by) * (ax * cy - cx * ay) +
                (cx*cx + cy*cy) * (ax * by - bx * ay);
    ftype orientation = (v2.x - v1.x) * (v3.y - v1.y) - (v2.y - v1.y) * (v3.x - v1.x);
    return (orientation > 0) ? (det > eps) : (det < -eps);
}

vector<point> saved_waypoints;
bool initialized = false;
int current_wp_idx = 0;

void my_planner(const envmap& curmap,
                const array<pair<point, point>, playercount>& playerdata,
                const array<point, rays>& raycasts, const agent& curplayer,
                ftype& a, ftype& steer) {
    
    // --- STEP 1: GLOBAL PLANNING ---
    if (!initialized) {
        vector<point> pts;
        for (const auto& poly : curmap) {
            for (const auto& p : poly) pts.push_back(p);
        }
        
        ftype minX = 1e9, minY = 1e9, maxX = -1e9, maxY = -1e9;
        for (auto& p : pts) {
            minX = min(minX, p.x); minY = min(minY, p.y);
            maxX = max(maxX, p.x); maxY = max(maxY, p.y);
        }
        ftype dx = maxX - minX, dy = maxY - minY;
        ftype deltaMax = max(dx, dy), midX = (minX + maxX) / 2, midY = (minY + maxY) / 2;
        pts.push_back(point(midX - 20 * deltaMax, midY - deltaMax));
        pts.push_back(point(midX, midY + 20 * deltaMax));
        pts.push_back(point(midX + 20 * deltaMax, midY - deltaMax));

        vector<Triangle> triangles;
        triangles.push_back(Triangle((int)pts.size()-3, (int)pts.size()-2, (int)pts.size()-1));

        for (int i = 0; i < (int)pts.size() - 3; i++) {
            for (auto& t : triangles) if (inCircumcircle(pts[i], pts[t.a], pts[t.b], pts[t.c])) t.isBad = true;
            vector<Edge> polygon;
            for (size_t t1 = 0; t1 < triangles.size(); ++t1) {
                if (!triangles[t1].isBad) continue;
                Edge es[3] = {{triangles[t1].a, triangles[t1].b}, {triangles[t1].b, triangles[t1].c}, {triangles[t1].c, triangles[t1].a}};
                for (int j = 0; j < 3; ++j) {
                    bool shared = false;
                    for (size_t t2 = 0; t2 < triangles.size(); ++t2) {
                        if (t1 != t2 && triangles[t2].isBad) {
                            Edge o[3] = {{triangles[t2].a, triangles[t2].b}, {triangles[t2].b, triangles[t2].c}, {triangles[t2].c, triangles[t2].a}};
                            for(int k=0; k<3; ++k) if ((es[j].a == o[k].a && es[j].b == o[k].b) || (es[j].a == o[k].b && es[j].b == o[k].a)) { shared = true; break; }
                        }
                        if (shared) break;
                    }
                    if (!shared) polygon.push_back(es[j]);
                }
            }
            vector<Triangle> nextT;
            for (auto& t : triangles) if (!t.isBad) nextT.push_back(t);
            for (auto& e : polygon) nextT.push_back(Triangle(e.a, e.b, i));
            triangles = nextT;
        }

        ftype safety_margin = 0.08f; 
        for (auto& t : triangles) {
            if (t.a >= (int)pts.size() - 3 || t.b >= (int)pts.size() - 3 || t.c >= (int)pts.size() - 3) continue;
            point c((pts[t.a].x + pts[t.b].x + pts[t.c].x)/3.0f, (pts[t.a].y + pts[t.b].y + pts[t.c].y)/3.0f);
            
            bool valid = true;
            for (size_t i = 0; i < curmap.size() - 1; i++) {
                if (contains(curmap[i], c)) { valid = false; break; }
                for (size_t j = 0; j < curmap[i].size(); j++) {
                    if (dist_to_segment(c, curmap[i][j], curmap[i][(j+1)%curmap[i].size()]) < safety_margin) { valid = false; break; }
                }
                if (!valid) break;
            }
            if (valid && contains(curmap.back(), c)) saved_waypoints.push_back(c);
        }
        initialized = true;
    }

    if (saved_waypoints.empty()) { a = 0; steer = 0; return; }

    // --- STEP 2: LOCAL CONTROLLER ---
    // --- STEP 2: LOCAL CONTROLLER (STABILIZED) ---
    point pos = playerdata[0].first;
    ftype theta = playerdata[0].second.y; 

    point target = saved_waypoints[current_wp_idx];
    
    if (dist(pos, target) < 0.20f) {
        current_wp_idx = (current_wp_idx + 1) % saved_waypoints.size();
        target = saved_waypoints[current_wp_idx];
    }

    point f_att = target - pos;
    ftype d_att = dist(point(0,0), f_att);
    if (d_att > 1e-3f) f_att = (3.0f / d_att) * f_att; 

    point f_rep(0, 0);
    ftype min_obs_dist = 1e9;
    ftype d_safe = 0.40f; 
    
    // We will use the 20 raycasts provided by the agent instead of perfect map knowledge.
    for (int i = 0; i < rays; i++) {
        // Retrieve the raycast hit point (contains simulated noise)
        point hit = raycasts[i];
        ftype d = dist(pos, hit);
        
        // Raycasts hitting the far bounding box or mdist should be ignored
        if (d > 3.0f) continue; 
        
        min_obs_dist = min(min_obs_dist, d);
        
        if (d < d_safe && d > 0.001f) {    
            ftype d_eff = max(d, 0.05f); // Limit extreme force spikes from noisy points right on the agent
            point r_vec = (1.0f / d) * (pos - hit);
            
            // Repulsive force magnitude (tuned for point-cloud style repulsion)
            ftype force_mag = 0.015f * (1.0f / d_eff - 1.0f / d_safe) / (d_eff * d_eff);
            
            // --- SYMMETRY BREAKER (SWIRL) ---
            point tangent(-r_vec.y, r_vec.x);
            ftype alignment = tangent.x * f_att.x + tangent.y * f_att.y;
            if (alignment < 0.01f) { 
                tangent = point(r_vec.y, -r_vec.x); 
            }
            
            // Combine strict repulsion and tangential slide (Noise naturally jittering makes finding local minimum harder)
            f_rep = f_rep + force_mag * (r_vec + 1.2f * tangent);
        }
    }

    // Explicit Visible Frame Boundary Guard
    if (curmap.size() > 0 && !contains(curmap.back(), pos)) {
        f_rep = f_rep + point(-pos.x * 50.0f, -pos.y * 50.0f); // Massive pull towards origin if outside
    }

    point total_force = f_att + f_rep;

    // --- DEADLOCK ESCAPE ---
    if (dist(point(0, 0), total_force) < 0.1f) {
        total_force = f_att + point(-f_att.y * 0.5f, f_att.x * 0.5f);
    }

    ftype desired_heading = arg(total_force);
    ftype heading_error = desired_heading - theta;
    while (heading_error > PI) heading_error -= 2 * PI;
    while (heading_error < -PI) heading_error += 2 * PI;

    steer = heading_error * 0.5f; 
    
    if (min_obs_dist < 0.05f) {
        a = -0.0002f; 
    } else if (abs(heading_error) > 0.6f || min_obs_dist < 0.15f) {
        a = 0.0001f; 
    } else {
        a = 0.0004f; 
    }
}

int main() {
    agent myagent;
    myagent.calculate_1 = my_planner;
    array<agent, playercount> myagents;
    myagents[0] = myagent;

    simulationinstance s(myagents, 60.0);
    //s.lidar_noise = 0.02f;
    s.humanmode = false;

    // Shrink the boundary box slightly so it is fully visible on screen
    if (!s.mp.empty()) {
        s.mp.back() = {point(0.95, 0.95), point(0.95, -0.95), point(-0.95, -0.95), point(-0.95, 0.95)};
    }

    // OBSTACLE SPEEDS REDUCED SIGNIFICANTLY
    const ftype T = 6.0f; // Longer period
    for (size_t i = 0; i < s.mp.size() - 1; i++) {
        ftype speed = 0.0002f + (i % 3) * 0.0001f; // ~10x slower than before
        ftype phase = i * 0.7f; 
        
        if (i % 2 == 0) {
            s.movementspecifier[i] = [=](vector<point>& obs, const ftype& curtime) {
                ftype shift_x = cos(curtime * 2.0f * PI / T + phase) * speed; 
                for(auto& p : obs) p.x += shift_x;
            };
        } else {
            s.movementspecifier[i] = [=](vector<point>& obs, const ftype& curtime) {
                ftype shift_y = sin(curtime * 2.0f * PI / T + phase) * speed;
                for(auto& p : obs) p.y += shift_y;
            };
        }
    }

    s.run();
    return 0;
}