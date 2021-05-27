size(8cm);

// Draw axes vectors
draw((0, 0) -- (0.5, 0),dashed,Arrow);
draw((0, 0) -- (0, 0.5),dashed,Arrow);
label("x", (0.5, 0), N);
label("y", (0, 0.5), N);

// Draw the triangle
draw((-1, -1) -- (1, -1) -- (-1, 1) -- cycle);

// Draw text labels
pen labelpen = rgb(0, 0, 0.75);
label("$O$",(0,0),SW);
label("(-1, -1)", (-1, -1), SW, labelpen);
label("(1, -1)", (1, -1), SE, labelpen);
label("(-1, 1)", (-1, 1), NW, labelpen);