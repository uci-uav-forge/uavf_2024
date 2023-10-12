labels are in this format:

```
shape letter shape_color letter_color x y w h
```

for example:
```
1 0 3 6 400 500 20 30
```
shape classes:  
  0: circle   
  1: cross    
  2: heptagon  
  3: hexagon  
  4: octagon  
  5: pentagon  
  6: quartercircle  
  7: rectangle  
  8: semicircle  
  9: square   
  10: star    
  11: trapezoid  
  12: triangle  
  13: person  

letter classes:
ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789

colors:
```
COLORS_TO_RGB = {
    'red': (255, 0, 0),
    'green': (0, 255, 0),
    'blue': (0, 0, 255),
    'orange': (255, 165, 0),
    'purple': (128, 0, 128),
    'white': (255, 255, 255),
    'black': (0, 0, 0),
    'brown': (165, 42, 42),
}
```