pair getPoint(real r, real ang){
	return (r*Cos(ang), r*Sin(ang));
}

real r = 50;
real num_bits = 7*3;
string[] days = {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"};
draw(circle((0,0), r));
int _counter = 0;
for(real i = 0; i < 360; i+=360/num_bits){
  pair current_point = getPoint(r, i);
  path c = circle(current_point, 5);
  pen fill_p = gray;
  pen border_p = black;
  if(i >= 100 && i <= 220){
  	fill_p = white;
  }

  filldraw(c, fill_p, border_p);
   if(_counter % 3 == 0){
  	dot(current_point);
  }
  ++_counter;
}



int counter = 0;
for(real i = 0; i < 360; i+=360/7){
  	pen p = black;
  	if(i > 150 && i< 170){
    	p = blue;
    }
  	if(cos(i) < 0 && sin(i) < 0){
      label(days[counter], getPoint(r*1.1, i), W, p);
    }
    else if(cos(i) < 0 && sin(i) > 0){
      label(days[counter], getPoint(r*1.1, i), N, p);
    }
  	else if(cos(i) > 0 && sin(i) > 0){
      label(days[counter], getPoint(r*1.1, i), E, p);
    }
    else if(cos(i) > 0 && sin(i) < 0){
      label(days[counter], getPoint(r*1.1, i), S, p);
    } 
  	else {
      label(days[counter], getPoint(r*1.1, i), E, p);
    }
  	++counter;
}

pair label_point = (r*2, -r/2);
filldraw(circle(label_point, 3), white);
label("Active Bit", shift(3, 0)*label_point, E);

label_point = (r*2, -r/2-15);
filldraw(circle(label_point, 3), gray);
label("Inactive Bit", shift(3, 0)*label_point, E);
