url = 'http://10.235.79.12:8080/shot.jpg';
ss  = imread(url);
fh = image(ss);
shouldDraw = 1;
while(shouldDraw)
    ss  = imread(url);
    set(fh,'CData',ss);
    drawnow;
end