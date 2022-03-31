settings.render = 8;
import three;
import fontsize;
srand(1);
int r=7;
real spacing = 1.5;
triple connectionStart;
currentprojection=oblique(50);
currentlight=Headlamp;
int max_x = 11;
int max_y = 4;
int max_z = 9;
// Top Layer
for(int x = 0; x < max_x; ++x){
	for(int y = 0; y < max_y; ++y){
  		for(int z = 0; z < max_z; ++z){
          
          triple pos = (x*r*2*spacing*1.2,y*r*2*spacing/4,z*r*2*spacing+1.5*y);
          path3 u=circle(pos,r);
          pen p=black+linewidth(2);
          pen c = white;
      	  if(x == 3 && z == max_z-1){
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
          draw(surface(u),emissive(c),p);
    }   
  }
}
// Bottom Layer
real height = -200;
for(int x = 0; x < max_x; ++x){
	for(int z = 0; z < max_z; ++z){
      triple pos = (x*r*2*spacing*1.2,height,z*r*2*spacing);
      path3 b=circle(pos,r);
      pen p=black+linewidth(3);
      pen c = gray;
      if(rand()%5==1){
          c = white;
      }
      bool draw_connection = (x==3 && z == max_z-1) || (x==2 && z == max_z-5) || (x == 1 && z==max_z-4) || (x==4 && z==max_z-4) || (x==3 && z == max_z-3) || (x==2 && z==max_z-2);
      bool dashed_line = (x==3 && z == max_z-1) || (x==3 && z == max_z-3) || (x == 2 && z==max_z-2);
      if(draw_connection){
        pen pn = chartreuse+linewidth(2);
        if(dashed_line){
        	pn=lightred+linewidth(2)+dashed;
        }
        path3 connectionLine = shift(0, 0, 2)*connectionStart--shift(0, 0, 2)*pos;
        draw(connectionLine, pn);
      }

      draw(surface(b), emissive(c), p);
    }
}

// Box
int box_x=1, box_z=max_z-1, box_size_x = 5, box_size_z = 5;
real padding=r*2;
surface test = 
  shift(box_x*r*2*spacing-padding/2, height-r-padding/2, (box_z-box_size_z)*r*2*spacing-padding/2)*
  scale(box_size_x*r*2*spacing+padding, 0.001*(r*2+padding), box_size_z*r*2*spacing)*
  unitcube;
draw(
  //shift(0, -padding, 0)*surface(m),
  test,
  emissive(gray+opacity(0.5)),
  red+linewidth(2)
);
// Labels
pen label_pen = black+fontsize(20)+TimesRoman();
label("SP Columns", (r*2*spacing*max_x/2, r*2*spacing, 0),N, label_pen);
label("Input SDR", (r*2*spacing*max_x/2, height+r*2*spacing/max_y, 0), N, label_pen);
label("Receptive Field", (r*2*spacing*(box_x+(box_size_x+1)/2), height-r/2, r*2*spacing*max_z), S, label_pen+fontsize(16));


triple bit_label_pos;
bit_label_pos = (r*2*spacing*(max_x+4), r*2*spacing, r*2*spacing*(max_z/2));
draw(surface(circle(bit_label_pos, r)), emissive(chartreuse), black+linewidth(2));
label("Active Column", shift(r, 0, 0)*bit_label_pos, E, label_pen+fontsize(16));
bit_label_pos = (r*2*spacing*(max_x+4), 0, r*2*spacing*(max_z/2));
draw(surface(circle(bit_label_pos, r)), emissive(white), black+linewidth(2));
label("Inactive Column", shift(r, 0, 0)*bit_label_pos, E, label_pen+fontsize(16));

bit_label_pos = (r*2*spacing*(max_x+4), height+r*2*spacing, r*2*spacing*(max_z/2));
draw(surface(circle(bit_label_pos, r)), emissive(white), black+linewidth(2));
label("Active Bit", shift(r, 0, 0)*bit_label_pos, E, label_pen+fontsize(16));
bit_label_pos = (r*2*spacing*(max_x+4), height, r*2*spacing*(max_z/2));
draw(surface(circle(bit_label_pos, r)), emissive(gray), black+linewidth(2));
label("Inactive Bit", shift(r, 0, 0)*bit_label_pos, E, label_pen+fontsize(16));
