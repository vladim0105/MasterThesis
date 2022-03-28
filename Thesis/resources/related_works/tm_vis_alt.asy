settings.render = 8;
import three;

int r=20;
real spacing = 1.1;
currentprojection=orthographic(camera=(0, 30, -100));
currentlight=light(0, -30, 100);
for(int x = 0; x < 14; ++x){
  for(int y = 0; y < 4; ++y){
    for(int z = 0; z < 5; ++z){
          path3 u=circle((x*r*2*spacing*1.2-z*20,y*r*2*spacing/3,z*r*2*spacing*2.2-y*5),r, (80/360, 0, 1));
          pen p=black+linewidth(3);
          pen c = white;
      	  if(rand()%30 == 1){
          	c = blue;
          } else {
          	if(rand()%30==1){
            	c=green;
            }
          }
      	  if(x == 6 && z == 0){
          	c = blue;
          }
          draw(surface(u),c,p);
    }

  }

}