int num_circles = 4;
int r = 80;
// Stochastic
label("Stochastic Gradient Descent",(0, r+20));
for(int i = 0 ; i < num_circles; ++i){
	path c = circle((0, 0), r-i*r/4);
    filldraw(c, palegray);
}
draw((-80, 0)--(-60, 20), EndArrow);
draw((-60, 20)--(-50, -10), EndArrow);
draw((-50, -10)--(-35, 10), EndArrow);
draw((-35, 10)--(-15, 0), EndArrow);
draw((-15, 0)--(-3, 2), EndArrow);
filldraw(circle((0, 0), 1.5), chartreuse);

// Gradient
label("Gradient Descent",(r*2+20, r+20));
for(int i = 0 ; i < num_circles; ++i){
	path c = circle((r*2+20, 0), r-i*r/4);
    filldraw(c, palegray);
}
path[] points = {
  (-69, 40)--(-60, 20),
  (-60, 20)--(-40, 10),
  (-40, 10)--(-20, 5),
  (-20, 5)--(-5, 2),
                };
for(path p: points){
	draw(shift(r*2+20, 0)*p, EndArrow);
}
filldraw(circle((r*2+20, 0), 1.5), chartreuse);