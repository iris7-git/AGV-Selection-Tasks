#pragma once

#include "geometry.hpp"

// constants and defines

constexpr ftype screenWidth = 600, screenHeight = 600,
                scale = min(screenHeight, screenWidth) / 2;
const ftype pointradius = 2, lineweight = 2;

// declerations

// returns the screen coordinates of a point
point transfer(const point& a);
// draws a point to the screen
void drawpoint(const point& a, const Color& c = WHITE);
// draws a line segment
void drawline(const point& a, const point& b, const Color& c = WHITE);
// draws an arbitrary collection of points
void drawscatter(const vector<point>& a, const Color& c = WHITE);
// draws a polygon assumed to be given in the order of points
void drawpolygon(const vector<point>& a, const Color& fc = WHITE);
// same as previous but only draws boundary of polygon
void drawpolygonboundary(const vector<point>& a, const Color c = RED);

// implementations

point transfer(const point& a) {
    return {screenWidth / 2 + scale * a.x, screenHeight / 2 - scale * a.y};
}  // the coordinate plane is [-1, 1] X [-1, 1] for a square window and [-1, 1]
   // along the smaller axis otherwise

void drawpoint(const point& a, const Color& c) {
    DrawCircleV(transfer(a), pointradius, c);
}

void drawline(const point& a, const point& b, const Color& c) {
    DrawLineEx(transfer(a), transfer(b), lineweight, c);
}

void drawscatter(const vector<point>& a, const Color& c) {
    for (int i = 0; i < a.size(); i++) drawpoint(a[i], c);
}  // depends on drawpoint

void drawpolygon(const vector<point>& a, const Color& fc) {
    const int n = a.size();
    for (int i = 1; i < n - 1; i++)
        DrawTriangle(transfer(a[0]), transfer(a[i]), transfer(a[i + 1]), fc);
}  // depends on nothing

void drawpolygonboundary(const vector<point>& a, const Color c) {
    const int n = a.size();

    for (int i = 0; i < n - 1; i++)
        DrawLineEx(transfer(a[i]), transfer(a[i + 1]), lineweight, c);
    DrawLineEx(transfer(a[0]), transfer(a[n - 1]), lineweight, c);

    for (int i = 0; i < n; i++) drawpoint(a[i]);
}  // depends on drawpoint
