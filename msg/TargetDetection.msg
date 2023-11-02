uint64 timestamp

float64 x
float64 y
float64 z

float64[8] shape_conf 
# should sum to 1
# indices in order:
# circle, semicircle, quarter circle, triangle, rectangle, pentagon, star, cross

float64[36] letter_conf
# should sum to 1
# indices in order:
# ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789

float64[8] shape_color_conf
# should sum to 1
# indices in order:
# white, black, red, blue, green, purple, brown, orange

float64[8] letter_color_conf
# should sum to 1
# indices in order:
# white, black, red, blue, green, purple, brown, orange