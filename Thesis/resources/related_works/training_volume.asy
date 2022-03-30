import graph;

size(300,175,IgnoreAspect);
scale(Log,Linear);
real f(real t){
	return log10(t)*10+60;
}
draw(graph(f,10,1000), red+linewidth(2));
pen thin=linewidth(0.5*linewidth());
real[] graph_ticks = {10, 100, 1000};
real[] graph_ticks2 = {20, 40, 60, 80, 100};
real[] ticks;
for(real i = 10; i < 1000; i+=20){
	ticks.push(i);
}
ylimits(0, 100);
xaxis("Thousands of Samples",BottomTop,LeftTicks(extend=true, Ticks=graph_ticks, ticks=ticks));
yaxis("Accuracy",LeftRight,RightTicks(extend=true,Ticks=graph_ticks2));