uint64 timestamp

float64 lat
float64 lon

float64[8] shape_conf 
# should sum to 1
# indices in order:
# circle, semicircle, quarter circle, triangle, rectangle, pentagon, star, cross
float64[8] color_conf
# should sum to 1
# indices in order:
# white, black, red, blue, green, purple, brown, orange