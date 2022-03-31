import three;
import graph3;
import fontsize;
currentprojection = oblique();
currentlight = Headlamp;

void plot_column(triple pos, int max_x, int max_y, int max_z, int layer_size, real spacing){
	for(int x = 0; x < max_x; ++x){
    	for(int y = 0; y < max_y; ++y){
          	for(int z = 0; z < max_z; ++z){
        		draw(
                  shift(pos)*shift(x*spacing, y*spacing, z*spacing)*unitcube,
                  emissive(lightcyan), black+linewidth(0.55)
                );
              	if(x == 0 && z == 0 && y % layer_size == 0){
                	draw(
                      shift(pos)*
                      shift(x*spacing-spacing/4, y*spacing-spacing/8, z*spacing-spacing/4)*
                      scale(max_x*spacing+spacing/2, 0.1, max_z*spacing+spacing/2)*unitcube,
                      emissive(cyan), black+linewidth(0.55)
                    );
                }
            }
        }
    }
}
real spacing = 1.22;
int dim = 3;
int layer_size = 3;
int height = layer_size*6;
plot_column((0, 0, 0), dim, height, dim, layer_size, spacing);
plot_column((7.5, 0, 0), 3, height, 3, layer_size, spacing);
plot_column((20, 0, 0), 3, height, 3, layer_size, spacing);

string[] layer_names = {"L6", "L5", "L4", "L3", "L2", "L1"};
for(int i = 0; i < layer_names.length; ++i){
  label(layer_names[i], (-4, i*layer_size*spacing+0.5, 1), E, fontsize(24));
}
// Mini column
draw(
  shift(-0.3/2, 0, (dim-1)*spacing)*
  scale(1.3, (height*spacing+(1-spacing))+0.1, 1.1)*unitcube, 
  opacity(0), red+linewidth(1.3)+dashed
);

// Cortical column
draw(
  shift(-7.5, -0.5, dim+1)*
  scale(dim*spacing+10.5, height*spacing+(1-spacing)+3.2, 1)*unitplane,
  opacity(0), purple+linewidth(1.3)+dashed
);

// Cortical Region
draw(
  shift(-8, -2.2, dim+1)*
  scale(dim*spacing*3+8+15.5, height*spacing+(1-spacing)+5.5, 1)*
  unitplane,
  opacity(0), black+linewidth(1.3)+dashed
);


label("Mini-column", (0.6, height*spacing+1, dim+1), W, red+fontsize(24));
label("Cortical Column", ((dim*spacing+4)/2-4, -0.5, dim+1), S, purple+fontsize(24));
label("Cortical Region", ((dim*spacing*3+8)/2, -2, dim+1), S, fontsize(24));
label("...", ((dim*spacing*3+4)/2+9, height/2+1, dim+1), fontsize(35));
size(20cm);