import three;

currentprojection=orthographic(5,3,2,center=true);

size(8cm);
size3(5cm,5cm,5cm);

// Draw axes vectors
draw((0, 0, 0) -- (2.5, 0, 0),dashed,Arrow3);
draw((0, 0, 0) -- (0, 2.5, 0),dashed,Arrow3);
draw((0, 0, 0) -- (0, 0, 2.5),dashed,Arrow3);
label("x", (2, 0, 0), N);
label("y", (0, 2, 0), N);
label("z", (0, 0, 2), E);

// Draw obstructed edges
draw((-1, -1, -1) -- (1, -1, -1),dotted);
draw((-1, -1, -1) -- (-1, 1, -1),dotted);
draw((-1, -1, -1) -- (-1, -1, 1),dotted);

// Draw unobstructed edges
draw((1, -1, -1) -- (-1, 1, -1) -- (-1, 1, 1) -- (1, -1, 1) -- cycle);
draw((1, -1, 1) -- (-1, 1, 1) -- (-1, -1, 1) -- cycle);

// Draw text labels
pen labelpen = rgb(0, 0, 0.75);
label("$O$",(0,0,0),NW);
label("(-1, -1, -1)", (-1, -1, -1), S, labelpen);
label("(1, -1, -1)", (1, -1, -1), SW, labelpen);
label("(-1, 1, -1)", (-1, 1, -1), E, labelpen);
label("(-1, -1, 1)", (-1, -1, 1), NW, labelpen);
label("(1, -1, 1)", (1, -1, 1), W, labelpen);
label("(-1, 1, 1)", (-1, 1, 1), E, labelpen);