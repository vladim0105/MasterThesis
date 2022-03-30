settings.render = 8;
import three;
import fontsize;
srand(1);
int r=7;
real spacing = 1.5;
triple connectionStart;
currentprojection=orthographic(camera=(0, 30, -100));
currentlight=light(0, -30, 100);
// Top Layer
for(int x = 0; x < 11; ++x){
  for(int z = 0; z < 9; ++z){
    for(int y = 0; y < 4; ++y){
          
          triple pos = (x*r*2*spacing*1.2-z*r,y*r*2*spacing/4,z*r*2*spacing*2.2-y*2);
          path3 u=circle(pos,r);
          pen p=black+linewidth(2);
          pen c = white;
      	  if(x == 6 && z == 0){
          	c = chartreuse;
            if(y == 0){
            	connectionStart = pos;
            }
          }
          if(x == 3 && z == 2){
          	c = chartreuse;
          }
          if(x == 5 && z == 5){
          	c = chartreuse;
          }
          draw(surface(u),c,p);
    }   
  }
}
// Bottom Layer
real height = -200;
for(int x = 0; x < 11; ++x){
	for(int z = 0; z < 9; ++z){
      triple pos = (x*r*2*spacing*1.2-z*r,height,z*r*2*spacing*3);
      path3 b=circle(pos,r);
      pen p=black+linewidth(3);
      pen c = gray;
      if(rand()%10==1){
          c = white;
      }
      bool draw_connection = (x==6 && z == 0) || (x==6 && z == 4) || (x == 8 && z==3) || (x==4 && z==3) || (x==5 && z == 2) || (x==7 && z==1);
      bool dashed_line = (x==6 && z == 0) || (x==6 && z == 4) || (x == 8 && z==3);
      if(draw_connection){
        pen pn = chartreuse+linewidth(2);
        if(dashed_line){
        	pn=lightred+linewidth(2)+dashed;
        }
        path3 connectionLine = shift(0, 0, 5)*connectionStart--shift(0, 0, 2)*pos;
        draw(connectionLine, pn);
      }

      draw(surface(b), c, p);
    }
}
// Box

real x, z;
real padding=r;
path3 m;

x = 4;
z = 0;
m = shift(-padding, 0, -padding)*((x*r*2*spacing*1.2-z*r,height,z*r*2*spacing*3));

x = 9;
z = 0;
m = m--shift(padding, 0, -padding)*((x*r*2*spacing*1.2-z*r,height,z*r*2*spacing*3));

x = 9;
z = 4;
m = m--shift(padding, 0, padding*3)*((x*r*2*spacing*1.2-z*r,height,z*r*2*spacing*3));

x = 4;
z = 4;
m = m--shift(-padding, 0, padding*3)*((x*r*2*spacing*1.2-z*r,height,z*r*2*spacing*3));

m = m--cycle;

//Labels
draw(shift(0, -r, 0)*surface(m),lightgray+opacity(opacity=0.3, blend="Overlay"), black+dashed+linewidth(2));
pen label_pen = black+fontsize(24)+TimesRoman();
label("SP Columns", (50, 180, -90),E, label_pen);
label("Input SDR", (50, 0, -90),E, label_pen);
label("Receptive Field", (165, -200, -90), label_pen);
triple bit_label_pos;

bit_label_pos = (-75, 100, -90);
draw(surface(circle(bit_label_pos, r)), chartreuse, black+linewidth(2));
label("Active Column", shift(-5, 0, 0)*bit_label_pos, E, label_pen);
bit_label_pos = (-75, 68, -90);
draw(surface(circle(bit_label_pos, r)), white, black+linewidth(2));
label("Inactive Column", shift(-5, 0, 0)*bit_label_pos, E, label_pen);

bit_label_pos = (-75, -100, -90);
draw(surface(circle(bit_label_pos, r)), white, black+linewidth(2));
label("Active Bit", shift(-5, 0, 0)*bit_label_pos, E, label_pen);
bit_label_pos = (-75, -132, -90);
draw(surface(circle(bit_label_pos, r)), gray, black+linewidth(2));
label("Inactive Bit", shift(-5, 0, 0)*bit_label_pos, E, label_pen);

//draw(O -- 200X, L=Label("$x$",position=EndPoint));
//draw(O -- 200Y, L=Label("$y$", position=EndPoint));
//draw(O -- 200Z, L=Label("$z$", position=EndPoint));